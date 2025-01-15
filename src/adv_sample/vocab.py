"""
This module provides a flexible vocabulary class for token embeddings.
We can use phrase embeddings which are multiple tokens long
so our vocabulary can contain words and phrases of different lengths.
"""

import torch
# TODO: uncomment the following line
#from sentence_transformers.util import normalize_embeddings
import logging
logger = logging.getLogger(__name__)

class FlexibleVocab:
    def __init__(self, vocab_tokens, embedding_matrix):
        """
        Create a flexible vocabulary for token embeddings.
        
        Args:
        - vocab_tokens (list of lists): Each sublist is the id of tokenized word/phrase e.g.,  [[1], [2, 40], [201]] coresponding to [["the"], ["New", "York"], ["a"]])
        - embedding_matrix (Tensor): Pre-trained embedding matrix for all tokens (d_vocab, d_model).
        """
        self.vocab_token = {}
        self.vocab = {}
        self.embedding_matrix = embedding_matrix  # Shape: [d_vocab, d_model]
        
        # Process vocabulary
        for tokens in vocab_tokens:
            token_count = len(tokens)
            token_embeddings = self.embedding_matrix[tokens]  # Get embeddings for the tokens
            
            # Normalize embeddings TODO: when implementing compare() function should be switched to normalize_embeddings across the entire vocab
            # TODO: uncomment the following line
            #token_embeddings = normalize_embeddings(token_embeddings)
            
            # Store by token count
            if token_count not in self.vocab:
                self.vocab[token_count] = []
                self.vocab_token[token_count] = []

            self.vocab[token_count].append(token_embeddings)
            self.vocab_token[token_count].append(tokens)
        
        # Convert lists to tensors for efficient comparison
        for token_count in self.vocab:
            self.vocab[token_count] = torch.stack(self.vocab[token_count])  # Shape: [num_words, token_count, embedding_dim]

        logger.debug("FlexibleVocab: __init__(): initialized with %d words/phrases", len(vocab_tokens))

    def compare_strict(self, input_embeddings=None, input_tokens=None):
        """
        Compare input tokens with the vocabulary and compute dot products but only between the the same size tokens.
        
        Args:
        - input_embeddings (torch.Tensor): A tensor of shape [token_count_input, embedding_dim] e.g. [2, 40, 512] (coresponding to ["New", "York"]).
        - input_tokens (torch.Tensor): A tensor of shape [token_count_input] e.g. [2, 40] (coresponding to ["New", "York"]).
        
        Returns:
        - similarities (torch.Tensor): A tensor of shape [num_words] containing dot product similarities for each vocab entry.
        - vocab_token (list): A list of tokenized words/phrases in the vocabulary for this token count.
        
        """
        if input_embeddings is None:
            token_count_input = len(input_tokens)
            # Get embeddings for the input tokens
            input_embeddings = self.embedding_matrix[input_tokens]  # Shape: [token_count_input, embedding_dim]
        
        else:
            token_count_input = len(input_embeddings)
        
        if token_count_input not in self.vocab:
            logger.debug("FlexibleVocab: compare_strict(): no matches for token count %d", token_count_input)

            return torch.empty((0), device=input_tokens.device), self.vocab_token[token_count_input]  # No matches, return empty tensor
        
        
        logger.debug("FlexibleVocab: compare_strict(): input_embeddings shape: %s", input_embeddings.shape)
        # Retrieve the relevant part of the vocabulary
        vocab_embeddings = self.vocab[token_count_input]  # Shape: [num_words, token_count, embedding_dim]
        
        logger.debug("FlexibleVocab: compare_strict(): vocab_embeddings shape: %s", vocab_embeddings.shape)

        # Compute dot product for each word in the vocab
        dot_products = torch.einsum('ntd,td->nt', vocab_embeddings, input_embeddings)  # Shape: [num_words, token_count_input]

        # Sum the dot products across the token count dimension
        dot_products = dot_products.sum(dim=-1)

        logger.debug("FlexibleVocab: compare_strict(): dot_products shape: %s", dot_products.shape)

        return dot_products, self.vocab_token[token_count_input]
    
    def compare_strict_batch(self, 
                             input_embeddings: torch.tensor = None, 
                             input_tokens: torch.tensor = None):
        """
        Compare input tokens with the vocabulary and compute dot products, 
        but only between the same size tokens for the entire batch.
        
        Args:
        - input_embeddings (torch.Tensor): A tensor of shape [batch_size, token_count_input, embedding_dim] 
        where each row is a sequence of embeddings with the same length.
        - input_tokens (torch.Tensor): A tensor of shape [batch_size, token_count_input] 
        where each row is a sequence of token IDs with the same length.
        
        Returns:
        - similarities (torch.Tensor): A tensor of shape [batch_size, num_words] containing 
        dot product similarities for each batch entry and matching vocab entry.
        - vocab_token (list): A list of tokenized words/phrases in the vocabulary for this token count.
        """
        if input_embeddings is None:
            batch_size, token_count_input = input_tokens.shape
            # Get embeddings for the input tokens
            input_embeddings = self.embedding_matrix[input_tokens]  # Shape: [batch_size, token_count_input, embedding_dim]

        else:
            batch_size, token_count_input, _ = input_embeddings.shape

        if token_count_input not in self.vocab:
            logger.debug("FlexibleVocab: compare_strict_batch(): no matches for token count %d", token_count_input)

            return torch.empty((batch_size, 0), device=input_tokens.device), self.vocab_token[token_count_input] # No matches, return empty tensor
        
        
        # Retrieve the relevant part of the vocabulary
        vocab_embeddings = self.vocab[token_count_input]  # Shape: [num_words, token_count_input, embedding_dim]
        
        # Compute dot products for each word in the vocab across the batch
        dot_products = torch.einsum('btd,ntd->bn', vocab_embeddings, input_embeddings)  # Shape: [num_words, batch_size]

        dot_products = dot_products.t() # Shape: [batch_size, num_words]

        logger.debug("FlexibleVocab: compare_strict_batch(): dot_products shape: %s", dot_products.shape)
        
        return dot_products, self.vocab_token[token_count_input]


    def compare(self, input_tokens, diff_length_penalty=0.5):
        #TODO: Experimental function, not used in the current implementation
        """
        Compare input tokens with the vocabulary and compute dot products.
        
        Args:
        - input_tokens (torch.Tensor): A tokenized input phrase e.g. [2, 40] (coresponding to ["New", "York"]).
        
        Returns:
        - similarities (dict): A dictionary with token count as key and dot product similarities as values.
        - vocab_token (list): A list of tokenized words/phrases in the vocabulary for this token count.
        """
        token_count_input = len(input_tokens)
        
        input_embeddings = self.embedding_matrix[input_tokens]  # Shape: [token_count, embedding_dim]

        dot_results = {}

        for token_count in self.vocab:

            vocab_embeddings = self.vocab[token_count] # Shape: [num_words, token_count, embedding_dim]

            if token_count_input > token_count:

                dot_products = torch.einsum('ntd,td->nt', input_embeddings[:token_count], vocab_embeddings)  # Shape: [num_words, token_count_input]

                # Sum the dot products across the token count dimension
                dot_products = dot_products.sum(dim=-1)

            else:

                dot_products = torch.einsum('ntd,td->nt', input_embeddings[:token_count], vocab_embeddings)  # Shape: [num_words, token_count_input]

                # Take into account results that are no longer than the input
                dot_products = dot_products[:, :token_count_input]

                # Sum the dot products across the token count dimension
                dot_products = dot_products.sum(dim=-1)

            # Add penalty for different token counts
            diff_length_penalty = diff_length_penalty * (token_count - token_count_input)
            dot_products -= diff_length_penalty
            dot_results[token_count] = dot_products

        logger.debug("FlexibleVocab: compare(): dot_results shape: %s", dot_results)

        return dot_results, self.vocab_token

            


def main():
    # Example usage

    # Dummy embedding matrix (10 tokens, 5 dimensions)
    embedding_matrix = torch.randn(8, 5)

    # Tokenized vocabulary (words/phrases)
    vocab_string = [["hello"], ["world"], ["New", "York"], ["Los", "Angeles"], ["San", "Francisco"]]
    words_to_ids = {"hello": 0, "world": 1, "New": 2, "York": 3, "Los": 4, "Angeles": 5, "San": 6, "Francisco": 7}
    ids_to_words = {v: k for k, v in words_to_ids.items()}

    vocab_tokens = [[words_to_ids[word] for word in token] for token in vocab_string]

    print("Tokenized list:", vocab_tokens)

    # Create FlexibleVocab object
    flex_vocab = FlexibleVocab(vocab_tokens, embedding_matrix)

    # Compare input tokens
    print("Input tokens: [2, 3] (New York)")
    similarities, token_list = flex_vocab.compare_strict(torch.tensor([2, 3]))
    print("Similarities:", similarities)
    print("Token list:", token_list)
    string_list = [[ids_to_words[token] for token in tokens] for tokens in token_list]
    print("Token list (string):", string_list)

    # Compare input tokens (batch)
    print("Input tokens (batch): [[2, 3], [2, 1]] (New York, New world)")
    similarities, token_list = flex_vocab.compare_strict_batch(torch.tensor([[2, 3], [2, 1]]))
    print("Batch similarities:", similarities)
    print("Token list:", token_list)
    string_list = [[ids_to_words[token] for token in tokens] for tokens in token_list]
    print("Token list (string):", string_list)

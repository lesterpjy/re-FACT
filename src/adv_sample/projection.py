"""
This module contains the function to project the adversarial embeddings
to the closest token in the embedding space of chosen vocabulary.
"""
import torch
import logging
from src.adv_sample.vocab import FlexibleVocab

logger = logging.getLogger(__name__)



def project_embeddings(sample_embeddings: torch.tensor,
                       sample_tokens: torch.tensor, 
                       vocab: FlexibleVocab, 
                       mask: torch.tensor,
                       method: str = 'strict') -> torch.tensor:
    """
    Given a batch of sample adversarial embeddings, project
    them into their closest token in the embedding space.

    Input:
    - `sample_embeddings`: Tensor of shape (batch_size, seq_length, embedding_dim) containing the embeddings.
    - `sample_tokens`: Tensor of shape (batch_size, seq_length) containing the tokens.
    - `vocab`: The vocabulary to use for the adversarial samples.
    - `mask`: Tensor of shape (batch_size, seq_length) containing the mask.
    - `method`: The method to use for the projection.

    Output:
    - `batch_emb`: Tensor of shape (batch_size, seq_length, embedding_dim) containing the projected embeddings.
    - `batch_tokens`: Tensor of shape (batch_size, seq_length) containing the projected tokens.

    """

    batch_emb = sample_embeddings.clone()
    batch_tokens = sample_tokens.clone()

    logger.debug(f"project_embeddings: batch_emb.shape: {batch_emb.shape}")
    for i in torch.unique(mask):
        logger.debug(f"project_embeddings: i: {i}")
        # Get the indices where mask == i
        indices = mask == i

        # Expand indices to match the embedding dimensions
        expanded_indices = indices.unsqueeze(-1).expand_as(batch_emb)

        logger.debug(f"project_embeddings: expanded_indices.shape: {expanded_indices.shape}")
        # Select embeddings where mask == i
        selected_embeddings = batch_emb[expanded_indices].view(-1, i, batch_emb.size(-1))

        # Apply the perturbation function
        #TODO: make optional this function, here we use stict comparison
        if method == 'strict':
            results, token_list = vocab.compare_strict_batch(input_embeddings=selected_embeddings)

            idx_closest = results.argmax(dim=-1)

            token_closest = torch.tensor([token_list[idx] for idx in idx_closest.tolist()]).view(-1)
            
            embedding_closest = vocab.embedding_matrix[token_closest].view(-1) #getting embedding from token_closest as list of token ids
            
            logger.debug(f"project_embeddings: embedding_closest.shape: {embedding_closest.shape}")
        #embedding_closest = embedding_closest.view(*sample_embeddings.size())  #returing to the original shape of selected_embeddings
        #embedding_closest = embedding_closest.view(-1) #flattening the tensor for assignment
        # those 2 lines together can be skipped because we are returning to the shape of embedding_closest, but good for understanding the shape

        # Update the embeddings in batch_emb
        logger.debug(f"project_embeddings: batch_emb[expanded_indices] shape: {batch_emb[expanded_indices].shape}")
        batch_emb[expanded_indices] = embedding_closest
        batch_tokens[indices] = token_closest

    logger.debug(f"project_embeddings: batch_emb.shape: {batch_emb.shape}")
    logger.debug(f"project_embeddings: batch_tokens.shape: {batch_tokens.shape}")

    return batch_emb, batch_tokens    


def main():
    # Test the projection function
    # Dummy embedding matrix (8 tokens, 5 dimensions)
    embedding_matrix = torch.randn(8, 5)

    # Tokenized vocabulary (words/phrases)
    vocab_string = [["hello"], ["world"], ["New", "York"], ["Los", "Angeles"], ["San", "Francisco"]]
    words_to_ids = {"hello": 0, "world": 1, "New": 2, "York": 3, "Los": 4, "Angeles": 5, "San": 6, "Francisco": 7}

    vocab_tokens = [[words_to_ids[word] for word in token] for token in vocab_string]

    print("Tokenized list:", vocab_tokens)

    # Create FlexibleVocab object
    flex_vocab = FlexibleVocab(vocab_tokens, embedding_matrix)

    sentences = [["hello world New York"], ["Los Angeles San Francisco"]]
    print(f"Sentences: {sentences}")
    tokenized_sentences = torch.tensor([[words_to_ids[word] for word in sentence[0].split()] for sentence in sentences])
    print(f"Tokenized sentences: {tokenized_sentences}")
    embedded_sentences = flex_vocab.embedding_matrix[tokenized_sentences.view(-1)].view(tokenized_sentences.shape[0], tokenized_sentences.shape[1], -1)
    print(f"Embedded sentences shape: {embedded_sentences.shape}")

    #add random noise to the embeddings
    perturbed_embeddings = embedded_sentences + torch.randn(embedded_sentences.shape)*0.0001

    batch_emb, batch_tokens  = project_embeddings(sample_embeddings = perturbed_embeddings,
                        sample_tokens = torch.tensor(tokenized_sentences),
                        vocab = flex_vocab,
                        mask = torch.ones_like(tokenized_sentences)+torch.ones_like(tokenized_sentences),
                        method = 'strict')

    print("Batch tokens:", batch_tokens)
    print("Batch embeddings:", batch_emb)

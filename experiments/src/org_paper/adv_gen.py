from sentence_transformers.util import semantic_search, dot_score, normalize_embeddings
import torch
import numpy as np


def AdvMarginLoss(margin=1.0):
    """
    Create the adversarial Margin Loss
    """
    def loss_fn(logits, y, average=True):
        """
        Return the adversarial margin loss used to generate adversarial samples.

        Parameters:
        - `logits`: Tensor of shape (batch_size, num_classes) containing the logits.
        - `y`: Tensor of shape (batch_size,) containing the index of the ground truth.
        """
        # gather the logits of the ground truth
        logits_correct = logits[torch.arange(0, logits.shape[0]), y] # (batch_size,)
        # retrieve the maximum logits of the incorrect answers
        y_incorrect = torch.ones_like(logits, dtype=bool)
        y_incorrect[torch.arange(0, logits.shape[0]), y] = 0
        max_logits_incorrect = logits[y_incorrect].view(logits.shape[0], -1).max(1).values # (batch_size,)
        loss = (logits_correct - max_logits_incorrect + margin)
        loss = torch.where(loss < 0, torch.zeros_like(loss), loss)

        return loss.mean() if average else loss

    return loss_fn


def project_embeddings(sample_tokens, sample_embeddings, embedding_matrix, vocab, mask):
    """
    Given a batch of sample adversarial embeddings, project
    them into their closest token in the embedding space.

    Input:
    - `sample_embeddings`: Tensor of shape (batch_size, seq_len, d_model)
    - `embedding_matrix`: Tensor of shape (d_vocab, d_model)
    - `vocab`: Tensor of shape (n_vocab,), where n_vocab <= d_vocab, containing
        the list of possible tokens when projecting/updating the samples. The reason
        for this is that, for example, when generating acronyms, we might not want to
        change a capital letter to a non-capital letter, i.e. we want to stay inside a
        concrete vocabulary.
    - `mask`: Tensor of shape (seq_len,). mask[i] = 1 means that the i-th token will be
    optimized/changed by the algorithm.

    Returns:

    Two tensors containing both the ids and the embedding vectors of the projected vectors.
    - `projected_tokens`: Tensor of shape (batch_size, seq_len)
    - `projected_embeddings`: Tensor of shape (batch_size, seq_len, d_model)
    """
    vocab_embedding_matrix = normalize_embeddings(embedding_matrix[vocab])
    # project the sample embeddings, i.e. find the closest embedding pertaining to a real token
    result = semantic_search(normalize_embeddings(sample_embeddings.view(-1, sample_embeddings.shape[-1])), # flatten the batch size and pos dimensions and normalize
                vocab_embedding_matrix,
                query_chunk_size=sample_embeddings.shape[0], top_k=1,
                score_function=dot_score)
    projected_tokens = torch.tensor([vocab[x[0]["corpus_id"]] for x in result]).cuda()
    projected_tokens = projected_tokens.view(sample_embeddings.shape[:-1]) # (batch_size, seq_len)
    projected_tokens = torch.where(mask, projected_tokens, sample_tokens) # project only the tokens specified by the mask
    projected_embeddings = embedding_matrix[projected_tokens].clone().detach()
    return projected_tokens, projected_embeddings


def grad_stat_org(mask, gradients):
  mask_cpu = mask.cpu()
  mask_cpu = mask_cpu == 0 

  # Step 1: Flatten all gradients to compute global mean and std
  all_gradients = np.concatenate(
      [np.array(grad.cpu()).reshape(-1, grad.shape[-1]) for grad in gradients], axis=0
  )  # Shape: [total_elements, embedding_dim]

  # Compute global mean and std
  global_mean = np.mean(all_gradients, axis=0, keepdims=True)  # Shape: [1, embedding_dim]
  global_std = np.std(all_gradients, axis=0, keepdims=True) + 1e-8  # Shape: [1, embedding_dim]

  # Step 2: Normalize gradients globally
  masked_means = []
  unmasked_means = []

  masked_maxs = []
  unmasked_maxs = []

  grouped_emb = []

  for it in range(len(gradients)):
      grad = np.array(gradients[it].cpu())  # Convert to numpy for easier manipulation

      # Normalize gradients globally
      grad_norm = (grad - global_mean) / global_std  # Shape: [batch_size, token_length, embedding_dim]

      # Mean over embeddings for each token across the batch
      grad_mean_per_token = np.mean(abs(grad_norm), axis=(0,2))  # Shape: [token_length]

      # Separate masked and unmasked gradients
      masked_grad = grad_mean_per_token[mask_cpu]
      unmasked_grad = grad_mean_per_token[~mask_cpu]

      # # Compute means for masked and unmasked
      masked_mean = np.mean(masked_grad) if masked_grad.size > 0 else 0
      unmasked_mean = np.mean(unmasked_grad) if unmasked_grad.size > 0 else 0

      # Compute means for masked and unmasked
      masked_max = np.max(abs(masked_grad)) if masked_grad.size > 0 else 0
      unmasked_max = np.max(abs(unmasked_grad)) if unmasked_grad.size > 0 else 0

      masked_means.append(masked_mean)
      unmasked_means.append(unmasked_mean)

      masked_maxs.append(masked_max)
      unmasked_maxs.append(unmasked_max)

  return masked_maxs, unmasked_maxs, masked_means, unmasked_means
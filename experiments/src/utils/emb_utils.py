import torch
import pandas as pd
import numpy as np
from src.utils.logits_utils import compute_logit_diff_2

def group_emb(gradients):
    
    # Step 1: Flatten all gradients to compute global mean and std
    all_gradients = np.concatenate(
        [np.array(grad.cpu()).reshape(-1, grad.shape[-1]) for grad in gradients], axis=0
    )  # Shape: [total_elements, embedding_dim]

    # Compute global mean and std
    global_mean = np.mean(all_gradients, axis=0, keepdims=True)  # Shape: [1, embedding_dim]
    global_std = np.std(all_gradients, axis=0, keepdims=True) + 1e-8  # Shape: [1, embedding_dim]

    # Step 2: Normalize gradients globally
    mean_grad = []

    for it in range(len(gradients)):
        grad = np.array(gradients[it].cpu())  # Convert to numpy for easier manipulation

        # Normalize gradients globally
        grad_norm = (grad - global_mean) / global_std  # Shape: [batch_size, token_length, embedding_dim]

        # Mean over embeddings for each token across the batch
        grad_mean_per_token = np.mean(abs(grad_norm), axis=2)  # Shape: [batch_size,token_length]

        mean_grad.append(grad_mean_per_token)
        
    return mean_grad
    
  
def AdvMarginLoss(margin=1.0):
    """
    Create the adversarial Margin Loss
    """
    def loss_fn(logits_all, tokens_all, y, average=True, needed_tokens = [0, 1]):
        """
        Return the adversarial margin loss used to generate adversarial samples.

        Parameters:
        - `logits_all`: Tensor of shape (batch_size, seq_length, num_classes) containing the logits.
        - `y`: Tensor of shape (batch_size,) containing the index of the ground truth.
        """
        # gather the logits of the ground truth
        loss = compute_logit_diff_2(logits_all = logits_all,
                            tokens_all = tokens_all, 
                            correct_answers = y, 
                            needed_tokens = needed_tokens,
                            average=False)
        
        loss = loss + margin 
        loss = torch.where(loss < 0, torch.zeros_like(loss), loss)

        return loss.mean() if average else loss

    return loss_fn



def create_mask(model, sample_tokens, names):
    mask = torch.zeros_like(sample_tokens)

    for j, name in enumerate(names):
      # Find the sequence in the larger tensor
      sequence = model.to_tokens(name, prepend_bos=False)[0]
      large_tensor = sample_tokens[j]
      seq_len = sequence.size(0)
      for i in range(len(large_tensor)- seq_len + 1):
          
          if torch.equal(large_tensor[i:i + seq_len], sequence):
              mask[j,i:i + seq_len] = seq_len  # Update mask with value of seq_len
              break


    #check the correctness of the mask
    for i,mas in enumerate(mask):
        if (mas != 0).any():
           continue
        else:
            print(f"Error: Mask at index {i} is entirely zero.")
            
    return mask
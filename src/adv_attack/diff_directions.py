import torch 
import einops
import logging

logger = logging.getLogger(__name__)

# def get_logit_diff_directions_seq(answer_tokens, wrong_tokens, model, device='cuda'):
#     """
#     Computes the logit difference directions for sequences of tokens, 
#     it is primarily used for tokens we want to generate from the model and the tokens we want to avoid.
    
#     Parameters:
#     -----------
#     - `answer_tokens`: Tensor of shape (batch_size, seq_dim) containing the correct tokens.
#     - `wrong_tokens`: Tensor of shape (batch_size, seq_dim) containing the incorrect tokens.
#     - `model`: The transformer model with the `tokens_to_residual_directions` method.
#     - `device`: The device to use (default: 'cuda').

#     Returns:
#     --------
#     - `logit_diff_directions`: Tensor of shape (batch_size, seq_dim, d_model) containing the logit difference directions.
#     """
#     # Flatten tokens to identify unique tokens across all sequences
#     all_tokens = torch.cat([answer_tokens.flatten(), wrong_tokens.flatten()]).unique()

#     # Get the residual directions for all unique tokens
#     residual_directions = model.tokens_to_residual_directions(all_tokens.to(device))  # (n_unique_tokens, d_model)

#     # Create a mapping from token IDs to their residual directions
#     token_to_direction = {token.item(): residual_directions[i] for i, token in enumerate(all_tokens)}

#     # Match residual directions to tokens for both answer_tokens and wrong_tokens
#     correct_directions, incorrect_directions = [
#         torch.stack([
#             torch.stack([token_to_direction[token.item()] for token in sequence], dim=0) # for each token in a sequence
#             for sequence in token_group # for each sample in a batch
#         ], dim=0)  # (batch_size, seq_dim, d_model)
#         for token_group in (answer_tokens, wrong_tokens) # the same oparation for both to find directions
#     ]

#     # Compute the logit difference directions
#     logit_diff_directions = correct_directions - incorrect_directions  # (batch_size, seq_dim, d_model)

#     return logit_diff_directions

def get_logit_directions(sample_tokens, model):
    """
    Efficiently maps input tokens to their corresponding residual directions by first finding unique tokens.

    Parameters:
    -----------
    - `sample_tokens`: Tensor of shape (batch_size, seq_dim) containing the tokens.
    - `model`: The transformer model, which provides `tokens_to_residual_directions`.

    Returns:
    --------
    - `sample_directions`: Tensor of shape (batch_size, seq_dim, d_model) containing the residual directions for the tokens.
    """
    # Flatten and find unique tokens
    unique_tokens, inverse_indices = torch.unique(sample_tokens, return_inverse=True)
    
    # Get residual directions for unique tokens
    unique_directions = model.tokens_to_residual_directions(unique_tokens)  # (num_unique_tokens, d_model)
    
    # Map back to original shape using inverse indices
    sample_directions = unique_directions[inverse_indices].reshape(*sample_tokens.shape, -1)  # (batch_size, seq_dim, d_model)
    
    return sample_directions



def get_heads_logit_diff(prompts: list[str], 
                         answer_tokens: torch.Tensor, 
                         wrong_tokens: torch.Tensor, 
                         position_of_eos: list[int],
                         model: torch.nn.Module, #TODO: change it 
                         position: int = -1, 
                         ):
    """
    Computes the logit difference directions for sequences of tokens, 
    it is primarily used for tokens we want to generate from the model and the tokens we want to avoid.
    
    Parameters:
    -----------
    - `prompts`: Tensor of shape (batch_size, seq_dim) containing the tokenized prompts.
    - `answer_tokens`: Tensor of shape (batch_size, seq_dim) containing the correct tokens.
    - `wrong_tokens`: Tensor of shape (batch_size, seq_dim) containing the incorrect tokens.
    - `position_of_eos`: The position of the EOS token in the input sample, in our case they differ across the batch (because of padding)
    - `model`: The transformer model with the `tokens_to_residual_directions` method.
    - 'position': The position of the (main) token in the sequence, 
      function get_logit_directions outputs shape (batch_size, seq_dim, d_model) so we need to select the position of the token we are interested in for the difference.

    Returns:
    --------
    - `per_head_logit_diffs`: Tensor of shape (n_layers, n_heads) containing the dot score between the residual directions and the logit difference directions.
    """


    _, cache = model.run_with_cache(prompts)

    correct_directions, incorrect_directions = [get_logit_directions(sample_tokens, model)
                                                for sample_tokens in (answer_tokens, wrong_tokens)]
        

    logit_diff_directions = correct_directions - incorrect_directions  # (batch_size, seq_dim, d_model)

    logit_diff_directions = logit_diff_directions[:, position]  # (batch_size, d_model)

    per_head_residual = cache.stack_head_results(layer=-1, return_labels=False)

    indices = position_of_eos.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)  # Shape: (1, batch_size, 1, 1)
    indices = indices.expand(per_head_residual.shape[0], -1, 1, per_head_residual.shape[-1])  # Shape: (lay x head, batch_size, 1, d_model)

    # Gather specific positions
    per_head_residual = per_head_residual.gather(dim=2, index=indices)  # Gather along the seq_len dimension
    
    per_head_residual = per_head_residual.squeeze(2)
    
    logger.debug(f"get_heads_logit_diff: per_head_residual shape: {per_head_residual.shape}")

    per_head_residual = einops.rearrange(
        per_head_residual,
        "(layer head) ... -> layer head ...",
        layer=model.cfg.n_layers
    )

    per_head_logit_diffs = residual_stack_to_logit_diff(per_head_residual, cache, logit_diff_directions).mean(-1)

    return per_head_logit_diffs


def residual_stack_to_logit_diff(residual_stack, cache, logit_diff_directions):
    '''
    Gets the avg logit difference between the correct and incorrect answer for a given
    stack of components in the residual stream.
    '''
    scaled_residual_stack = cache.apply_ln_to_stack(residual_stack, layer=-1, pos_slice=-1)

    return einops.einsum(
        scaled_residual_stack, logit_diff_directions,
        "... batch d_model, batch d_model -> ... batch"
    )

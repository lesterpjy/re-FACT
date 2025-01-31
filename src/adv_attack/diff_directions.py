import torch 
import einops
import logging
import pickle

logger = logging.getLogger(__name__)


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
                         batch_size: int = 2,
                         save_folder: str = "results",
                         save_prefix: str = "full"
                         ):
    """
    Computes the logit differences between the correct and incorrect answers for each head in the transformer model.

    Parameters:
    -----------
    - `prompts`: Tensor of shape (batch_size, seq_dim) containing the tokenized prompts.
    - `answer_tokens`: Tensor of shape (batch_size, seq_dim) containing the correct tokens.
    - `wrong_tokens`: Tensor of shape (batch_size, seq_dim) containing the incorrect tokens.
    - `position_of_eos`: The position of the EOS token in the input sample, in our case they differ across the batch (because of padding)
    - `model`: The transformer model with the `tokens_to_residual_directions` method.
    - 'position': The position of the (main) token in the sequence, 
      function get_logit_directions outputs shape (batch_size, seq_dim, d_model) so we need to select the position of the token we are interested in for the difference.
    - `batch_size`: The batch size to use for the computation.
    - `save_folder`: The folder to save the results.
    - `save_prefix`: The prefix to use for saving the results.

    Returns:
    --------
    """
    
    for i in range(0,len(prompts), batch_size):

        _, cache = model.run_with_cache(prompts[i:i+batch_size])

        correct_directions, incorrect_directions = [get_logit_directions(sample_tokens, model)
                                                    for sample_tokens in (answer_tokens[i:i+batch_size], wrong_tokens[i:i+batch_size])]


        logit_diff_directions = correct_directions - incorrect_directions  # (batch_size, seq_dim, d_model)

        logit_diff_directions = logit_diff_directions[:, position]  # (batch_size, d_model)

        per_head_residual = cache.stack_head_results(layer=-1, return_labels=False)

        indices = position_of_eos[i:i+batch_size].unsqueeze(0).unsqueeze(-1).unsqueeze(-1)  # Shape: (1, batch_size, 1, 1)
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

        with open(f"{save_folder}/{save_prefix}_per_head_logit_diffs_{i}.pkl", "wb") as handle:
            pickle.dump(per_head_logit_diffs, handle, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"Saved per_head_logit_diffs_{i}.pkl")


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

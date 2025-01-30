"""
This module contains the code to generate adversarial samples
for a given model, samples and vocabulary.
"""

import torch
from tqdm import tqdm
import logging
from src.adv_sample.projection import project_embeddings
from src.adv_sample.utils import AdvMarginLoss
from src.adv_sample.vocab import FlexibleVocab

logger = logging.getLogger(__name__)




def generate_adversarial_samples(model: torch.nn.Module, #TODO: change typing its from hooked transformer
                                sample_tokens: torch.tensor, 
                                y_sample: torch.tensor,
                                sample_embeddings: torch.tensor, 
                                vocab: FlexibleVocab, 
                                mask: torch.tensor,
                                #compare_function: str,
                                needed_tokens: list,
                                iterations: int = 10,
                                lr: float = 1e-1, 
                                weight_decay: float = 1e-1, 
                                margin: int = 4,
                                thresh: float = 0.0):
    """
    Generate adversarial samples for a given model, samples and vocabulary.

    Parameters:
    - `model`: The model to use for the generation.
    - `sample_tokens`: Tensor of shape (batch_size, seq_length) containing the tokens.
    - `y_sample`: Tensor of shape (batch_size,) containing the index of the ground truth.
    - `sample_embeddings`: Tensor of shape (batch_size, seq_length, embedding_dim) containing the embeddings.
    - `vocab`: The vocabulary to use for the adversarial samples.
    - `mask`: Tensor of shape (batch_size, seq_length) containing the mask, it can have values from 0 to n depending on the number of tokens specific phrase will be switched to.
    - `needed_tokens`: The tokens needed to compute the adversarial margin loss, in our example tokens for 'true' and 'false'.
    # `compare_function`: The function to use to compare the embeddings.
    - `iterations`: The number of iterations to run the optimization.
    - `lr`: The learning rate to use for the optimization.
    - `weight_decay`: The weight decay to use for the optimization.
    - `margin`: The margin to use for the adversarial margin loss.

    Output:
    - `adv_samples`: List of tensors containing the adversarial samples.
    - `original_samples`: List of tensors containing the original samples.
    - `losses`: List of losses for each iteration.
    """
    sample_embeddings.requires_grad = True

    # Initialize the adversarial margin loss
    loss_fn = AdvMarginLoss(margin=margin)

    input_optimizer = torch.optim.AdamW([sample_embeddings], lr=lr, weight_decay=weight_decay)

    # Additional mask for where to zero the gradient
    mask_0_1 = mask != 0

    # we will collect the adversarial samples: samples that are incorrectly classified by the model
    adv_samples = []
    # we also collect the original sample associated to each adversarial sample
    original_samples = []
    losses = []
    losses_2 = []
    gradients = []
    
    projected_tokens, projected_embeddings = sample_tokens.clone().detach(), sample_embeddings.clone().detach()

    # OPTIMIZE
    for _ in tqdm(range(iterations), disable=False):

        tmp_embeddings = sample_embeddings.detach().clone()
        tmp_embeddings.data = projected_embeddings.data
        tmp_embeddings.requires_grad = True

        if hasattr(model, 'pos_embed'):
            # GPT-2 style positional embeddings
            embeddings_with_pos = tmp_embeddings + model.pos_embed(projected_tokens)
        else:
            # LLaMA-2 style: RoPE is applied internally
            embeddings_with_pos = tmp_embeddings

        output_logits = model.forward(embeddings_with_pos, start_at_layer=0)
        #output_logits = model.forward(projected_tokens)


        loss = loss_fn(logits_all = output_logits, 
                       tokens_all = projected_tokens, 
                       y = y_sample,
                       needed_tokens = needed_tokens,
                       average = True)
 
        #logger.info(f"generate_adversarial_samples: loss: {loss.item()}")

        sample_embeddings.grad, = torch.autograd.grad(loss, [tmp_embeddings])
        # set the gradient of elements outside the mask to zero
        
        gradients.append(sample_embeddings.grad)
        
        sample_embeddings.grad = torch.where(mask_0_1[ ..., None], sample_embeddings.grad, 0.)
        input_optimizer.step()
        input_optimizer.zero_grad()

        losses.append(loss.item())

        with torch.no_grad():
            # Project the embeddings
            _, projected_tokens = project_embeddings(sample_embeddings = sample_embeddings,
                                                                        sample_tokens = sample_tokens,
                                                                        vocab = vocab,
                                                                        mask = mask,
                                                                        #compare_function = compare_function
                                                                       )
            
            projected_embeddings = vocab.embedding_matrix[projected_tokens]

            logger.debug(f"generate_adversarial_samples: projected_embeddings.shape: {projected_embeddings.shape}")
            logger.debug(f"generate_adversarial_samples: projected_tokens.shape: {projected_tokens.shape}")
            logger.debug(f"generate_adversarial_samples: projected_tokens: {projected_tokens}")

            # check if there are adversarial samples
            # Take the logits of the subspace
            if hasattr(model, 'pos_embed'):
                # GPT-2 style positional embeddings
                embeddings_2_with_pos = projected_embeddings + model.pos_embed(projected_tokens)
            else:
                # LLaMA-2 style: RoPE is applied internally
                embeddings_2_with_pos = projected_embeddings

        
            
            output_logits_2 = model.forward(embeddings_2_with_pos.float(), start_at_layer=0)

            loss_i = loss_fn(logits_all = output_logits_2, 
                                tokens_all = projected_tokens, 
                                y = y_sample,
                                needed_tokens = needed_tokens,
                                average = False)

            adv_samples.append(projected_tokens[loss_i < margin+thresh]) # a loss lower than margin implies that the sample is incorrectly classified (logits difference is equal 0); if our threshold for classifing a sample based on logit difference is different than 0 it should be switched  
            original_samples.append(sample_tokens[loss_i < margin+thresh])
            losses_2.append(loss_i[loss_i < margin+thresh])

    return adv_samples, original_samples, losses, losses_2, gradients


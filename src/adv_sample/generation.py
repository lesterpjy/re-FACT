"""
This module contains the code to generate adversarial samples
for a given model, samples and vocabulary.
"""

import torch
from tqdm import tqdm
import logging
from src.adv_sample.projection import project_embeddings
from src.adv_sample.utils import AdvMarginLoss

logger = logging.getLogger(__name__)




def generate_adversarial_samples(model, 
                                sample_tokens, 
                                y_sample,
                                sample_embeddings, 
                                vocab, 
                                mask, 
                                needed_tokens = [0, 1],
                                iterations = 10,
                                lr=1e-1, 
                                weight_decay=1e-1, 
                                margin=4):
    """
    Generate adversarial samples for a given model, samples and vocabulary.

    Parameters:
    - `model`: The model to use for the generation.
    - `sample_tokens`: Tensor of shape (batch_size, seq_length) containing the tokens.
    - `y_sample`: Tensor of shape (batch_size,) containing the index of the ground truth.
    - `sample_embeddings`: Tensor of shape (batch_size, seq_length, embedding_dim) containing the embeddings.
    - `vocab`: The vocabulary to use for the adversarial samples.
    - `mask`: Tensor of shape (batch_size, seq_length) containing the mask, it can have values from 0 to n depending on the number of tokens specific phrase will be switched to.
    - `needed_tokens`: The tokens needed to compute the adversarial margin loss.
    - `iterations`: The number of iterations to run the optimization.
    - `lr`: The learning rate to use for the optimization.
    - `weight_decay`: The weight decay to use for the optimization.
    - `margin`: The margin to use for the adversarial margin loss.

    Output:
    - `adv_samples`: List of tensors containing the adversarial samples.
    - `original_samples`: List of tensors containing the original samples.
    - `losses`: List of losses for each iteration.
    """

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

    projected_tokens, projected_embeddings = sample_tokens.clone(), sample_embeddings.clone()

    # OPTIMIZE
    for _ in tqdm(range(iterations), disable=False):

        tmp_embeddings = sample_embeddings.detach().clone()
        tmp_embeddings.data = projected_embeddings.data
        tmp_embeddings.requires_grad = True

        output_logits = model.forward(tmp_embeddings + model.pos_embed(projected_tokens), start_at_layer=0)

        loss = loss_fn(logits_all = output_logits, 
                       tokens_all = projected_tokens, 
                       y = y_sample,
                       needed_tokens = needed_tokens,
                       average = True)
 
        logger.info(f"generate_adversarial_samples: loss: {loss.item()}")

        sample_embeddings.grad, = torch.autograd.grad(loss, [tmp_embeddings])
        # set the gradient of elements outside the mask to zero
        sample_embeddings.grad = torch.where(mask_0_1[None, ..., None], sample_embeddings.grad, 0.)
        input_optimizer.step()
        input_optimizer.zero_grad()

        losses.append(loss.item())

    with torch.no_grad():
        # Project the embeddings
        projected_tokens, projected_embeddings = project_embeddings(sample_embeddings = sample_embeddings,
                                                                    sample_tokens = sample_tokens,
                                                                    vocab = vocab,
                                                                    mask = mask)

        
        #TODO: comment 1st line of debugger for the final version
        logger.debug(f"generate_adversarial_samples: is sample_tokens equal to projected_tokens: {torch.equal(sample_tokens, projected_tokens)}")
        logger.debug(f"generate_adversarial_samples: projected_embeddings.shape: {projected_embeddings.shape}")
        logger.debug(f"generate_adversarial_samples: projected_tokens.shape: {projected_tokens.shape}")
        logger.debug(f"generate_adversarial_samples: projected_tokens: {projected_tokens}")

        # check if there are adversarial samples
        # Take the logits of the subspace
        output_logits = model.forward(projected_embeddings + model.pos_embed(projected_tokens), start_at_layer=0)

        loss_i = loss_fn(logits_all = output_logits, 
                            tokens_all = projected_tokens, 
                            y = y_sample,
                            needed_tokens = needed_tokens,
                            average = False)
        
        adv_samples.append(projected_tokens[loss_i < margin]) # a loss lower than margin implies that the sample is incorrectly classified
        original_samples.append(sample_tokens[loss_i < margin])

    return adv_samples, original_samples, losses


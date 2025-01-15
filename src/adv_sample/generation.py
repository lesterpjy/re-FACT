"""
This module contains the code to generate adversarial samples
for a given model, samples and vocabulary.
"""
#TODO: not ready yet, REDO THE CODE

import torch
from tqdm import tqdm

#vocab = model.to_tokens(word_list, prepend_bos=False)[:, 0]
vocab = 
mask = 
sample_embeddings = 


lr=1e-1
weight_decay=1e-1
margin = 4
loss_fn = AdvMarginLoss(margin=margin)

input_optimizer = torch.optim.AdamW([sample_embeddings], lr=lr, weight_decay=weight_decay)

# we will collect the adversarial samples: samples that are incorrectly classified by the model
adv_samples = []
# we also collect the original sample associated to each adversarial sample
original_samples = []
losses = []


# OPTIMIZE
for iter in tqdm(range(10), disable=False):

    # Project the embeddings
    #projected_tokens, projected_embeddings = project_embeddings(sample_embeddings, embedding_matrix, vocab, mask)
    # Expand mask to match the last dimension of sample_embeddings
    #CHECK
    projected_tokens, projected_embeddings = project_embeddings(sample_embeddings, vocab, mask)

    print(f'iter: {iter}')
    projected_sample_tokens_from_embed = torch.argmax(torch.matmul(
        normalize_embeddings(sample_embeddings),
        normalize_embeddings(model.W_E.T)  # Assuming model.W_E is the embedding matrix
    ), dim=-1)
    print(f'equal: {torch.equal(projected_tokens, projected_sample_tokens_from_embed)}')
    print(f'original: {[model.to_string(x) for x in projected_sample_tokens_from_embed]}')
    print(f'projected tokens: {[model.to_string(x) for x in projected_tokens]}')
    # BRUH this is causing a high bottleneck. Optimize when everything works right

    sample_y_idx = torch.tensor([token_to_idx[model.to_tokens(model.to_string(x[indices_letters[letter]])[1], prepend_bos=False).item()] for x in projected_tokens], dtype=torch.long).cuda()
    print(f'first: {[model.to_string(x[indices_letters[letter]]) for x in projected_tokens ]}')
    print(f'second: {[model.to_string(x[indices_letters[letter]])[1] for x in projected_tokens ]}')

    print(f'sample_y_idx: {sample_y_idx}')
    tmp_embeddings = sample_embeddings.detach().clone()
    tmp_embeddings.data = projected_embeddings.data
    tmp_embeddings.requires_grad = True

    # Take the logits of the subspace
    projected_tokens_from_embed = torch.argmax(torch.matmul(
        normalize_embeddings(tmp_embeddings),
        normalize_embeddings(model.W_E.T)  # Assuming model.W_E is the embedding matrix
    ), dim=-1)

    # Print the tokens as strings
    print(f'tmp_embeddings: {[model.to_string(x) for x in projected_tokens_from_embed]}')

    projected_tokens_from_embed = torch.argmax(torch.matmul(
        normalize_embeddings(tmp_embeddings + model.pos_embed(projected_tokens)),
        normalize_embeddings(model.W_E.T)  # Assuming model.W_E is the embedding matrix
    ), dim=-1)

    # Print the tokens as strings
    print(f'into model forward: {[model.to_string(x) for x in projected_tokens_from_embed]}')
    logits_vocab = model.forward(tmp_embeddings + model.pos_embed(projected_tokens), start_at_layer=0)[:, indices_logits[letter], cap_tokens]

    loss = loss_fn(logits_vocab, sample_y_idx, average=True)
    print(f"loss: {loss.item()}")

    sample_embeddings.grad, = torch.autograd.grad(loss, [tmp_embeddings])
    # set the gradient of elements outside the mask to zero
    sample_embeddings.grad = torch.where(mask[None, ..., None], sample_embeddings.grad, 0.)
    input_optimizer.step()
    input_optimizer.zero_grad()
    #print(loss.item())
    #print(model.to_string(projected_tokens))
    losses.append(loss.item())

    with torch.no_grad():
        # Re-project the embeddings
        projected_tokens, projected_embeddings = project_embeddings(sample_embeddings, vocab, mask)

        sample_y_idx = torch.tensor([token_to_idx[model.to_tokens(model.to_string(x[indices_letters[letter]])[1], prepend_bos=False).item()] for x in projected_tokens], dtype=torch.long).cuda()
        # check if there are adversarial samples
        # Take the logits of the subspace
        logits_vocab = model.forward(projected_embeddings + model.pos_embed(projected_tokens), start_at_layer=0)[:, indices_logits[letter], cap_tokens]

        loss_i = loss_fn(logits_vocab, sample_y_idx, average=False)
        adv_samples.append(projected_tokens[loss_i < margin]) # a loss lower than margin implies that the sample is incorrectly classified
        original_samples.append(sample_tokens[loss_i < margin])
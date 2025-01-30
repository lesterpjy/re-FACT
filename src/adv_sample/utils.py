import torch

def get_all_logits(sampled_prompts, model, run_with_cache = True):
    tokens_all = model.to_tokens(sampled_prompts, prepend_bos = False)
    if run_with_cache:
        logits_all, cache_all = model.run_with_cache(tokens_all)
        return logits_all, tokens_all, cache_all
    logits_all = model(tokens_all)
    return logits_all, tokens_all

def find_bos_token(tokens_all, bos_token = 128009):
    """
    
    """
    x = []
    
    for tokens in tokens_all:
        bos = (tokens == bos_token).nonzero(as_tuple=True)[0]
        if len(bos) == 0:
            x.append(len(tokens) - 1)
        else:
            x.append(bos[0] - 1)
    return x

def find_bos_token(tokens_all, bos_token=128009):
    """
    Finds the position of the BOS token in each sequence of a batch of tokenized sentences.

    Parameters:
    -----------
    tokens_all : Tensor
        A batch of tokenized sentences, shape (batch_size, seq_len).
    bos_token : int, optional
        The ID of the BOS token to find. Default is 128009.

    Returns:
    --------
    List[int]
        A list of positions for the BOS token in each sentence. If not found,
        the position is set to len(tokens) - 1.
    """
    # Check where the BOS token exists in each sentence
    bos_positions = (tokens_all == bos_token).nonzero(as_tuple=False)

    # Create a result array initialized with len(tokens) - 1 for each sentence
    result = [len(tokens) - 1 for tokens in tokens_all]
    
    # Update with BOS token positions
    for sentence_idx, token_idx in reversed(bos_positions):
        if token_idx == 0:
          continue
        result[sentence_idx] = token_idx.item() - 1
    
    return result



def filter_answer_logits(logits_all, tokens_all, needed_tokens):
    
    x = find_bos_token(tokens_all)

    x = torch.tensor(x, device=logits_all.device, dtype=torch.long)
    logits_answer = torch.stack([logits[idx, needed_tokens] for logits, idx in zip(logits_all, x)])

    return logits_answer

def compute_logit_diff_2(logits_all, tokens_all, correct_answers: int, needed_tokens,average=True):
    logits = filter_answer_logits(logits_all, tokens_all, needed_tokens)
    logit_diffs = ((logits[:, 0] - logits[:, 1])*torch.tensor(correct_answers).to(logits.device))
    return logit_diffs.mean() if average else logit_diffs


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
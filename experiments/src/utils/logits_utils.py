import torch 


def get_all_logits(sampled_prompts: list[str],
                   model: torch.nn.Module, 
                   run_with_cache: bool = True
                   ):
    """
    Generates logits for a list of sampled prompts.

    Parameters:
    -----------
    sampled_prompts : List[str]
        A list of sampled prompts.
    model : Module  
        A LLM model.
    run_with_cache : bool, optional
        If True, runs the model with cache. Default is True.

    Returns:
    --------
    Tensor
        A tensor of logits for each token in each sentence, shape (batch_size, seq_len, vocab_size).
    Tensor
        A tensor of tokens for each sentence, shape (batch_size, seq_len).
    Tensor
        A tensor of cache for each sentence, shape (batch_size, num_layers, num_heads, seq_len, head_size).
    """

    tokens_all = model.to_tokens(sampled_prompts, prepend_bos = False)
    if run_with_cache:
        logits_all, cache_all = model.run_with_cache(tokens_all)
        return logits_all, tokens_all, cache_all
    logits_all = model(tokens_all)
    return logits_all, tokens_all



def find_bos_token(tokens_all: torch.Tensor,
                   bos_token: int = 128009):
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



def filter_answer_logits(logits_all: torch.Tensor,
                         tokens_all: torch.Tensor,
                         needed_tokens: list[int] = [0, 1]
                         ):
    """
    Filters the logits for the answer tokens in each sentence.
    
    Parameters:
    -----------
    logits_all : Tensor
        A batch of logits, shape (batch_size, seq_len, vocab_size).
    tokens_all : Tensor
        A batch of tokenized sentences, shape (batch_size, seq_len).
    needed_tokens : List[int]
        A list of token positions to extract logits from.
        
    Returns:
    --------
    Tensor
        A tensor of logits for the answer tokens in each sentence, shape (batch_size, len(needed_tokens)).
    """
    
    x = find_bos_token(tokens_all)

    x = torch.tensor(x, device=logits_all.device, dtype=torch.long)
    logits_answer = torch.stack([logits[idx, needed_tokens] for logits, idx in zip(logits_all, x)])

    return logits_answer


def compute_logit_diff_2(logits_all: torch.Tensor,
                         tokens_all: torch.Tensor, 
                         correct_answers: list[int], 
                         needed_tokens: list[int] = [0, 1],
                         average: bool = True
                         ):
    """
    Computes the difference between the logits of the correct and incorrect answers.

    Parameters:
    -----------
    logits_all : Tensor
        A batch of logits, shape (batch_size, seq_len, vocab_size).
    tokens_all : Tensor
        A batch of tokenized sentences, shape (batch_size, seq_len).
    correct_answers : List[int]
        A list of correct answer positions in the vocab.
    needed_tokens : List[int], optional
        A list of token positions to extract logits from. Default is [0, 1].
    average : bool, optional
        If True, returns the average logit difference. Default is True.
    
    Returns:
    --------
    Tensor
        A tensor of logit differences for the correct and incorrect answers in each sentence.
    """
    
    logits = filter_answer_logits(logits_all, tokens_all, needed_tokens)
    logit_diffs = ((logits[:, 0] - logits[:, 1])*torch.tensor(correct_answers).to(logits.device))
    return logit_diffs.mean() if average else logit_diffs


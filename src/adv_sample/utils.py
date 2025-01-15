import torch

def get_all_logits(sampled_prompts, model, run_with_cache = True):
  tokens_all = model.to_tokens(sampled_prompts)

  if run_with_cache:
    logits_all, cache_all = model.run_with_cache(tokens_all)
    return logits_all, tokens_all, cache_all
  
  logits_all = model(tokens_all)
  return logits_all, tokens_all

def filter_answer_logits(logits_all, tokens_all, needed_tokens):
  x = []
  for tokens in tokens_all:
    bos = (tokens == 128009).nonzero(as_tuple=True)[0]
    if len(bos) == 0:
      x.append(torch.tensor(-1).to("cuda"))
    else:
      x.append(bos[0]-1)

  logits_answer = []
  for i, logits in enumerate(logits_all):
    logits_answer.append(logits[x[i], needed_tokens].cpu().detach().numpy())


  return torch.tensor(logits_answer)

def compute_logit_diff_2(logits_all, tokens_all, correct_answers: list[int], needed_tokens,average=True):

  logits = filter_answer_logits(logits_all, tokens_all, needed_tokens)

  logit_diffs = ((logits[:, 0] - logits[:, 1])*torch.tensor(correct_answers))
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
from engine import Value
import numpy as np

def mse(gt, pred):
    '''
    mean squared error
    '''
    gt = Value._to_value(gt)

    gt.require_grad = False     # for safety
    return (gt - pred) ** 2



def bce(target, pred):
    '''
    binary cross entropy loss
    '''
    target = Value._to_value(target)
    target.require_grad = False     # for safety

    eps = 1e-12
    return -(target * (pred+eps).log() + ((1 - target) * ((1-pred +eps).log())))


def bce_with_logits(target, logits):
    '''
    achieving numerical stability by passing logits
    '''
    target = Value._to_value(target)
    target.require_grad = False     # for safety

    max_value = logits.relu()          # apply max(x,0)
    abs_logits = (-logits).relu() + logits.relu()       #  |x|
    log_term = (1 + (-abs_logits).exp()).log()          # log(1 + e**-|logits|)
    return max_value - target * logits + log_term



def softmax(logits):
    '''
    softmax function
    '''
    max_logit = logits[0]
    for logit in logits[1:]:
        if logit.data > max_logit.data:
            max_logit = logit

    exps = []
    exp_sum = Value(0.0)
    for logit in logits:
        e = (logit - max_logit).exp()
        exps.append(e)
        exp_sum += e

    softs = [e / exp_sum for e in exps]
    return softs


def cross_entropy(targets, probs):
    '''
    cross entropy with probabilities
    '''
    if not isinstance(targets, np.ndarray):
        targets = np.array(targets)

    target_idx = targets.argmax()
    return -probs[target_idx].log()



def cross_entropy_with_logits(targets, logits):
    '''
    achieving numerical stability by passing logits
    '''

    if not isinstance(targets, np.ndarray):
        targets = np.array(targets)

    # get target index
    target_idx = targets.argmax()

    # get max logit for shifting logits
    max_logit = logits[0]
    for logit in logits[1:]:
        if logit.data > max_logit.data:
            max_logit = logit

    # shift each logit , apply exp and calculate sum
    exp_sum = 0.0       # if exp_sum is initiated to be Value(0.0) this causes an error that confused me.
    for logit in logits:
        exp_sum += (logit - max_logit).exp()

    log_term = exp_sum.log() + max_logit

    # loss
    loss = log_term - logits[target_idx]
    return loss
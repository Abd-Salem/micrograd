from engine import Value


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
    max_value = logits.relu()          # apply max(x,0)
    abs_logits = (-logits).relu() + logits.relu()       #  |x|
    log_term = (1 + (-abs_logits).exp()).log()          # log(1 + e**-|logits|)
    return max_value - target * logits + log_term




def mse(gt, pred):
    '''
    mean squared error
    '''
    gt = Value._to_value(gt)

    gt.require_grad = False     # for safety
    return (gt - pred) ** 2

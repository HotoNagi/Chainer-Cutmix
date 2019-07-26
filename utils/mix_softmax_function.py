import chainer.functions as F

def mix_softmaxcross_entropy(y, t):
    cross_entropy = - F.sum(t * F.log_softmax(y))
    return cross_entropy / y.shape[0]
import torch
import torch.nn.functional as F


# taken from https://github.com/pytorch/examples/blob/master/imagenet/main.py
def accuracy_topk(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

# idea and code got from https://github.com/fastai/fastai/blob/master/fastai/metrics.py#L23
def accuracy(y_pred: torch.Tensor, y_true: torch.Tensor):
    bsize = y_pred.size(0)
    y_pred = y_pred.argmax(dim=-1).view(bsize, -1)
    y_true = y_true.view(bsize, -1)
    acc = y_pred==y_true
    return acc.float().mean()

# idea and code  got from https://github.com/fastai/fastai/blob/master/fastai/metrics.py#L30
def accuracy_threshold(y_pred:torch.Tensor, y_true:torch.Tensor, thresh:float=0.5, sigmoid:bool=False):
    if sigmoid: y_pred = F.sigmoid(y_pred)
    y_thresh = y_pred > thresh
    acc = y_thresh==y_true.byte()
    return acc.float().mean()





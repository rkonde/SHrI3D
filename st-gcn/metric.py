import numpy as np
import torch


def accuracy(preds, target):
    topk = torch.topk(preds, 1)[1]
    if np.isin(topk.cpu().numpy(), target.item()).any():
        return 1
    else:
        return 0

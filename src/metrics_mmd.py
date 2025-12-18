import torch
import numpy as np

def compute_mmd(x, y, sigma=1.0):
    def rbf(a, b):
        d = ((a[:, None] - b[None, :]) ** 2).sum(-1)
        return torch.exp(-d / (2 * sigma ** 2))

    xx = rbf(x, x).mean()
    yy = rbf(y, y).mean()
    xy = rbf(x, y).mean()
    return (xx + yy - 2 * xy).item()

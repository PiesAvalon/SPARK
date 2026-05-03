import numpy as np


def MAEDistance(x, y):
    x = x.detach().cpu().numpy()
    y = y.detach().cpu().numpy()
    return np.mean(np.abs(x - y))

def ChebyshevDistance(x, y):
    x = x.detach().cpu().numpy()
    y = y.detach().cpu().numpy()
    return np.max(np.abs(x - y))
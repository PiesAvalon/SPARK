import numpy as np
import torch


def ChebyshevDistance(x, y):
    x = x.detach().cpu().numpy()
    y = y.detach().cpu().numpy()
    return np.max(np.abs(x - y))


def ManhattanDistance(x, y):
    return torch.sum(torch.abs(torch.sub(x, y)))


def EuclideanDistance(x, y):
    # 欧式距离
    # print("x.type", type(x))
    # print("y.type", type(y))
    out = torch.sqrt(torch.sum(torch.square(torch.sub(x, y))))
    return out.detach().cpu().numpy()

distance_MODE = {
    "ManhattanDistance": ManhattanDistance,
    # ChebyshevDistance调用numpy实现，速度可能会相对较慢
    "ChebyshevDistance": ChebyshevDistance,
    "EuclideanDistance": EuclideanDistance,
}

def distance(x1, x2):
    distance_real = distance_MODE["EuclideanDistance"]
    dis = distance_real(x1, x2)
    return dis
import numpy as np

def MAEDistance(x, y):
    x = x.asnumpy()  # 将 MindSpore 张量转换为 NumPy 数组
    y = y.asnumpy()
    return np.mean(np.abs(x - y))

def ChebyshevDistance(x, y):
    x = x.asnumpy()  # 将 MindSpore 张量转换为 NumPy 数组
    y = y.asnumpy()
    return np.max(np.abs(x - y))

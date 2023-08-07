import numpy as np
from dezero import Variable

gpu_enable = True
try:
    import cupy as cp

    cupy = cp
except ImportError:
    gpu_enable = False


# 返回参数x的对应模块，x为Variable或ndarray类型(numpy或cupy)
def get_array_module(x):
    if isinstance(x, Variable):
        x = x.data

    if not gpu_enable:
        return np
    xp = cp.get_array_module(x)
    return xp


# 将x转化为numpy
def as_numpy(x):
    if isinstance(x, Variable):
        x = x.data

    if np.isscalar(x):
        return np.array(x)
    elif isinstance(x, np.ndarray):
        return x
    return cp.asnumpy(x)


# 将x转换为cupy
def as_cupy(x):
    if isinstance(x, Variable):
        x = x.data

    if not gpu_enable:
        raise Exception('Cupy cannot be loaded. Install Cupy!')
    return cp.asarray(x)

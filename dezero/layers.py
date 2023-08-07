import os.path
import weakref

import numpy as np

import dezero.functions as F
from dezero import cuda
from dezero.core import Parameter


# 基类
class Layer:
    def __init__(self):
        self._params = set()

    # 该方法在属性被赋值时调用
    def __setattr__(self, key, value):
        if isinstance(value, (Parameter, Layer)):
            self._params.add(key)
        super().__setattr__(key, value)

    def __call__(self, *inputs):
        outputs = self.forward(*inputs)
        if not isinstance(outputs, tuple):
            outputs = (outputs,)
        self.inputs = [weakref.ref(x) for x in inputs]
        self.outputs = [weakref.ref(y) for y in outputs]
        return outputs if len(outputs) > 1 else outputs[0]

    def forward(self, inputs):
        raise NotImplementedError()

    # 取出Layer实例所持有的Parameter实例
    # 提供给cleargrads()函数使用
    def params(self):
        for name in self._params:
            obj = self.__dict__[name]

            # yield使用方法与return相同
            # return会直接返回结果，而yield则是暂停处理并返回值，之后会恢复回来
            if isinstance(obj, Layer):
                yield from obj.params()
            else:
                yield obj

    def cleargrads(self):
        for param in self.params():
            param.cleargrad()

    def _flatten_params(self, params_dict, parent_key=''):
        for name in self._params:
            obj = self.__dict__[name]
            # 一开始都是进行else的处理，如果递归了，才会进行else前面的处理
            key = parent_key + '/' + name if parent_key else name

            # 递归
            if isinstance(obj, Layer):
                obj._flatten_params(params_dict, key)
            else:
                params_dict[key] = obj

    def save_weights(self, path):
        self.to_cpu()

        params_dcit = {}
        self._flatten_params(params_dcit)
        array_dict = {key: param.data for key, param in params_dcit.items()
                      if param is not None}
        try:
            np.savez_compressed(path, **array_dict)
        except(Exception, KeyboardInterrupt) as e:
            if os.path.exists(path):
                os.remove(path)
            raise

    def load_weights(self, path):
        npz = np.load(path)
        params_dict = {}
        # 模型里面是存在参数数据的，但是值是随机的
        # 将保存的参数赋值给模型中参数
        self._flatten_params(params_dict)
        for key, param in params_dict.items():
            param.data = npz[key]

    def to_cpu(self):
        # 每一个param都是Parameter类，而Parameter类继承了Variable类，因此param也具有to_gpu和to_cpu方法
        for param in self.params():
            param.to_cpu()

    def to_gpu(self):
        for param in self.params():
            param.to_gpu()


class Linear(Layer):
    def __init__(self, out_size, nobias=False, dtype=np.float32, in_size=None):
        super().__init__()

        self.in_size = in_size
        self.out_size = out_size
        self.dtype = dtype

        self.W = Parameter(None, name='W')
        # 如果没有指定in_size，则在forward代码中才初始化权重矩阵，这样的好处是不用用户指定，代码自动获取输入的形状
        if self.in_size is not None:
            self._init_W()

        if nobias:
            self.b = None
        else:
            self.b = Parameter(np.zeros(out_size, dtype=dtype), name='b')

    def _init_W(self, xp=np):
        I, O = self.in_size, self.out_size
        W_data = xp.random.randn(I, O).astype(self.dtype) * np.sqrt(1 / I)
        self.W.data = W_data

    def forward(self, x):
        # 在传播数据时初始化权重，自动获取
        if self.W.data is None:
            self.in_size = x.shape[1]
            xp = cuda.get_array_module(x)
            self._init_W(xp)

        y = F.linear(x, self.W, self.b)
        return y

import contextlib
import weakref

import numpy as np


class Config:
    enable_backprop = True


class Variable:
    __array_priority__ = 200

    def __init__(self, data, name=None):
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError(f"{type(data)} is not Support")

        self.data = data
        self.name = name
        # 导数值
        # 在计算图中，数据x对应的是导数值
        self.grad = None
        self.creator = None
        # 代表哪一代(即优先级)，用户的输入为父级，往后则为子级
        self.generation = 0

    # 从变量的角度来看，除了用户自己输入的变量，每一个变量都有自己的一个creator(函数)，即通过这个creator创建出了这个变量
    def set_creator(self, func):
        self.creator = func
        self.generation = func.generation + 1

    def cleargrad(self):
        self.grad = None

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        if self.data is None:
            return 'variable(None)'
        p = str(self.data).replace('\n', '\n' + ' ' * 9)
        return 'variable(' + p + ')'

    @property
    def shape(self):
        return self.data.shape

    @property
    def ndim(self):
        return self.data.ndim

    @property
    def size(self):
        return self.data.size

    @property
    def dtype(self):
        return self.data.dtype

    # 递归写法：消耗的内存空间太大了
    # 循环写法
    def backward(self, retain_grad=False):
        # 方便用户的操作，不用每次定义y.grad = np.array(1)
        if self.grad is None:
            self.grad = np.ones_like(self.data)

        funcs = []
        # 防止同一个函数被多次存放到funcs列表中
        seen_set = set()

        def add_func(f):
            if f not in seen_set:
                funcs.append(f)
                seen_set.add(f)
                # x为列表中的元素，利用x.generation进行排序，从小至大排序
                funcs.sort(key=lambda x: x.generation)

        add_func(self.creator)

        while funcs:
            f = funcs.pop()
            gys = [output().grad for output in f.outputs]
            gxs = f.backward(*gys)

            if not isinstance(gxs, tuple):
                gxs = (gxs,)

            for x, gx in zip(f.inputs, gxs):
                if x.grad is None:
                    x.grad = gx
                else:
                    x.grad = x.grad + gx

                if x.creator is not None:
                    add_func(x.creator)

            # 不要留中间变量的导数
            # 在前面全部处理完后再清除导数
            if not retain_grad:
                for y in f.outputs:
                    # 弱引用
                    y().grad = None


class Function:
    # 一些函数的输入和输出可能会有多个，因此要利用列表存储起来
    def __call__(self, *inputs):
        inputs = [as_variable(x) for x in inputs]

        xs = [x.data for x in inputs]
        # 具体的计算在forward方法中实现
        # *是解包的意思
        ys = self.forward(*xs)
        if not isinstance(ys, tuple):
            ys = (ys,)
        # 保证y也是np.ndarray实例后再放入Variable中
        outputs = [(Variable(as_array(y))) for y in ys]

        if Config.enable_backprop:
            self.generation = max([x.generation for x in inputs])
            # 让输出变量保存创造者的信息
            # 理解：这个输出是由该函数创造的，将self(即该函数的类本身)传递为creator
            for output in outputs:
                output.set_creator(self)
            # 反向传播时要用到正向传播中的变量，因此要保存起来
            self.inputs = inputs
            # 同时保存输出变量，方便实现自动反向传播
            # 弱引用，去除循环引用
            self.outputs = [weakref.ref(output) for output in outputs]

        return outputs if len(outputs) > 1 else outputs[0]

    # 正向传播
    def forward(self, xs):
        raise NotImplementedError()

    # 反向传播
    def backward(self, gys):
        raise NotImplementedError()


# 继承了Function类的同时，也继承了__call__方法
class Add(Function):
    def forward(self, x0, x1):
        y = x0 + x1
        return y

    def backward(self, gy):
        return gy, gy


class Mul(Function):
    def forward(self, x0, x1):
        y = x0 * x1
        return y

    def backward(self, gy):
        x0, x1 = self.inputs[0].data, self.inputs[1].data
        return x1 * gy, x0 * gy;


class Neg(Function):
    def forward(self, x):
        return -x

    def backward(self, gy):
        return -gy


class Sub(Function):
    def forward(self, x0, x1):
        y = x0 - x1
        return y

    def backward(self, gy):
        return gy, -gy


class Div(Function):
    def forward(self, x0, x1):
        y = x0 / x1
        return y

    def backward(self, gy):
        x0, x1 = self.inputs[0].data, self.inputs[1].data
        gx0 = gy / x1
        gx1 = gy * (-x0 / x1 ** 2)
        return gx0, gx1


class Pow(Function):
    def __init__(self, c):
        self.c = c

    def forward(self, x):
        y = x ** self.c
        return y

    def backward(self, gy):
        x, c = self.inputs[0].data, self.c
        gx = c * x ** (c - 1) * gy
        return gx


# 转换类型
def as_array(x):
    if np.isscalar(x):
        return np.array(x)
    return x


def as_variable(obj):
    if isinstance(obj, Variable):
        return obj
    return Variable(obj)


# 简化操作
def add(x0, x1):
    x1 = as_array(x1)
    return Add()(x0, x1)


def mul(x0, x1):
    x1 = as_array(x1)
    return Mul()(x0, x1)


def neg(x):
    return Neg()(x)


def sub(x0, x1):
    x1 = as_array(x1)
    return Sub()(x0, x1)


# rsub第一个参数是减号右边的，第二个参数是减号左边的(为常数)
def rsub(x0, x1):
    x1 = as_array(x1)
    return Sub()(x1, x0)


def div(x0, x1):
    x1 = as_array(x1)
    return Div()(x0, x1)


def rdiv(x0, x1):
    x1 = as_array(x1)
    return Div()(x1, x0)


def pow(x, c):
    return Pow(c)(x)


def set_variable():
    Variable.__add__ = add
    Variable.__radd__ = add
    Variable.__mul__ = mul
    Variable.__rmul__ = mul
    Variable.__neg__ = neg
    Variable.__sub__ = sub
    Variable.__rsub__ = rsub
    Variable.__truediv__ = div
    Variable.__rtruediv__ = rdiv
    Variable.__pow__ = pow


# with语句有前处理和后处理，调用后先关闭反向传播模式。处理结束后，自动打开反向传播模式
@contextlib.contextmanager
def using_config(name, value):
    old_value = getattr(Config, name)
    setattr(Config, name, value)

    try:
        yield
    finally:
        setattr(Config, name, old_value)


def no_grad():
    return using_config('enable_backprop', False)

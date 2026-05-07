import numpy as np

class Value:
    """ stores a single scalar value and its gradient """

    def __init__(self, data, _children=(), _op='', label='', require_grad=False):
        self.data = data
        self.grad = 0.0
        self.require_grad = require_grad    # gradient check
        # internal variables used for autograd graph construction
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op # the op that produced this node, for graphviz / debugging / etc
        self.label = label # for graphviz


    @staticmethod
    def _to_value(x):
        '''
        type checking for wrapping
        '''
        if isinstance(x ,Value):
            return x
        try:
            return Value(x, label=f'')
            # return Value(x, label=f'{x:.3f}') # this will show all labels in the graph
        except Exception:
            raise TypeError(f"Unsupported type: {type(x)}")


    def __add__(self, other):
        other = Value._to_value(other)
        out = Value(self.data + other.data, _children=(self, other),_op='+',
                    label=f'{self.label}+{other.label}' if self.label and other.label else '')

        out.require_grad = self.require_grad or other.require_grad

        def _backward():
            if self.require_grad:
                self.grad += out.grad

            if other.require_grad:
                other.grad += out.grad
        out._backward = _backward

        return out

    def __mul__(self, other):
        other = Value._to_value(other)
        out = Value(self.data * other.data, _children=(self, other),_op='*',
                    label=f'{self.label}*{other.label}' if self.label and other.label else '')

        out.require_grad = self.require_grad or other.require_grad

        def _backward():
            if self.require_grad:
                self.grad += other.data * out.grad

            if other.require_grad:
                other.grad += self.data * out.grad
        out._backward = _backward

        return out

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only supporting int/float powers for now"
        out = Value(self.data**other, _children=(self,), _op=f'**{other}',
                    label=f'{self.label}**{other}' if self.label else 'power')

        out.require_grad = self.require_grad

        def _backward():
            if self.require_grad:
                self.grad += (other * self.data**(other-1)) * out.grad
        out._backward = _backward

        return out

    def exp(self):
        data = np.exp(np.clip(self.data, -100, 100))

        out = Value(float(data), _children=(self, ), _op='exp',
                    label=f'exp({self.label})' if self.label else 'exp')

        out.require_grad = self.require_grad

        def _backward():
            if self.require_grad:
                self.grad += out.data * out.grad

        out._backward = _backward
        return out


    def log(self):
        eps = 1e-12
        data = np.log(self.data + eps)

        out = Value(float(data), _children=(self, ), _op='log',
                    label=f'log({self.label})' if self.label else 'log')
        out.require_grad = self.require_grad

        def _backward():
            if self.require_grad:
                self.grad += (1/ (self.data + eps)) * out.grad

        out._backward = _backward
        return out

    def relu(self):
        out = Value(0.0 if self.data < 0 else self.data, _children=(self,), _op='ReLU',
                    label=f'ReLU({self.label})' if self.label else 'ReLU')
        out.require_grad = self.require_grad

        def _backward():
            if self.require_grad:
                self.grad += (out.data > 0) * out.grad
        out._backward = _backward

        return out


    def tanh(self):
        t = np.tanh(self.data)

        out = Value(float(t), _children=(self, ), _op='tanh', label=f'tanh({self.label})' if self.label else '')
        out.require_grad = self.require_grad

        def _backward():
            if self.require_grad:
                self.grad += (1 - t**2) * out.grad

        out._backward = _backward
        return out

    def sigmoid(self):
        s = 1 / (1 + np.exp(-self.data))

        out = Value(float(s), _children=(self,), _op='sigmoid', label=f'sigmoid({self.label})' if self.label else '')
        out.require_grad = self.require_grad

        def _backward():
            if self.require_grad:
                self.grad += s * (1 - s) * out.grad

        out._backward = _backward
        return out

    def backward(self):

        # topological order all of the children in the graph
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        # go one variable at a time and apply the chain rule to get its gradient
        self.grad = 1
        for v in reversed(topo):
            v._backward()

    def __neg__(self): # -self
        return self * -1

    def __radd__(self, other): # other + self
        return self + other

    def __iadd__(self, other):
        return self + other

    def __sub__(self, other): # self - other
        return self + (-other)

    def __rsub__(self, other): # other - self
        return other + (-self)

    def __isub__(self, other):
        return self + (-other)

    def __rmul__(self, other): # other * self
        return self * other

    def __imul__(self, other):
        return self * other

    def __truediv__(self, other): # self / other
        return self * other**-1

    def __rtruediv__(self, other): # other / self
        return other * self**-1

    def __itruediv__(self, other):
        return self * (other**-1)

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"
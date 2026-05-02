
class Value:
    """ stores a single scalar value and its gradient """

    def __init__(self, data, _children=(), _op='', label=''):
        self.data = data
        self.grad = 0
        # internal variables used for autograd graph construction
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op # the op that produced this node, for graphviz / debugging / etc
        self.label = label # for graphviz
        self.require_grad = False


    @staticmethod
    def _to_value(x):
        '''
        type checking for wrapping
        '''
        if isinstance(x ,Value):
            return x
        try:
            return Value(x, label=f'({float(x):.2f})')
        except Exception:
            raise TypeError(f"Unsupported type: {type(x)}")


    def __add__(self, other):
        other = Value._to_value(other)
        out = Value(self.data + other.data, _children=(self, other),_op='+',
                    label=f'{self.label}+{other.label}' if self.label and other.label else '')

        out.require_grad = self.require_grad or other.require_grad
        if not out.require_grad:
            return out

        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward

        return out

    def __mul__(self, other):
        other = Value._to_value(other)
        out = Value(self.data * other.data, _children=(self, other),_op='*',
                    label=f'{self.label}*{other.label}' if self.label and other.label else '')

        out.require_grad = self.require_grad or other.require_grad
        if not out.require_grad:
            return out

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward

        return out

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only supporting int/float powers for now"
        out = Value(self.data**other, _children=(self,), _op=f'**{other}',
                    label=f'{self.label}**{other}' if self.label else 'power')

        out.require_grad = self.require_grad
        if not out.require_grad:
            return out

        def _backward():
            self.grad += (other * self.data**(other-1)) * out.grad
        out._backward = _backward

        return out

    def relu(self):
        out = Value(0.0 if self.data < 0 else self.data, _children=(self,), _op='ReLU',
                    label=f'ReLU({self.label})' if self.label else 'ReLU')

        out.require_grad = self.require_grad
        if not out.require_grad:
            return out

        def _backward():
            self.grad += (out.data > 0) * out.grad
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

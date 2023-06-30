import copy
import numpy as np

from initializations import _zero


class Optimizer(object):
    
    def __init__(self, lr=0.001, clip=-1, decay=0., lr_min=0., lr_max=np.inf):
        self.lr = lr
        self.clip = clip
        self.decay = decay
        self.lr_min = lr_min
        self.lr_max = lr_max

        self.iterations = 0

    def update(self, params, grads):
        
        self.iterations += 1

        self.lr *= (1. / 1 + self.decay * self.iterations)
        self.lr = np.clip(self.lr, self.lr_min, self.lr_max)

    def __str__(self):
        return self.__class__.__name__




#去掉继承类
class Adam(Optimizer):
    

    def __init__(self, beta1=0.9, beta2=0.999, epsilon=1e-8, *args, **kwargs):
        super(Adam, self).__init__(*args, **kwargs)

        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

        self.ms = None
        self.vs = None

    def update(self, params, grads):
        # init
        self.iterations += 1
        a_t = self.lr * np.sqrt(1 - np.power(self.beta2, self.iterations)) / \
              (1 - np.power(self.beta1, self.iterations))
        if self.ms is None:
            self.ms = [_zero(p.shape) for p in params]
        if self.vs is None:
            self.vs = [_zero(p.shape) for p in params]

        # update parameters
        for i, (m, v, p, g) in enumerate(zip(self.ms, self.vs, params, grads)):
            m = self.beta1 * m + (1 - self.beta1) * g
            v = self.beta2 * v + (1 - self.beta2) * np.power(g, 2)
            p -= a_t * m / (np.sqrt(v) + self.epsilon)

            self.ms[i] = m
            self.vs[i] = v





def npdl_clip(grad, boundary):
    if boundary > 0:
        return np.clip(grad, -boundary, boundary)
    else:
        return grad


def get(optimizer):
    if optimizer.__class__.__name__ == 'str':
        
        if optimizer in ['adam', 'Adam']:
            return Adam()
        
        raise ValueError('Unknown optimizer name: {}.'.format(optimizer))

    elif isinstance(optimizer, Optimizer):
        return copy.deepcopy(optimizer)

    else:
        raise ValueError("Unknown type: {}.".format(optimizer.__class__.__name__))
    
    
    
    

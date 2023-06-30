# -*- coding: utf-8 -*-


import numpy as np
import activations
from initializations import _zero
from initializations import get as get_init


#神经网络层
class Layer(object):
    """
    The :class:`Layer` class represents a single layer of a neural network. It
    should be subclassed when implementing new types of layers.

    Because each layer can keep track of the layer(s) feeding into it, a
    network's output :class:`Layer` instance can double as a handle to the full
    network.

    """

    first_layer = False

    def forward(self, input, *args, **kwargs):
        """ Calculate layer output for given input (forward propagation). """
        raise NotImplementedError

    def backward(self, pre_grad, *args, **kwargs):
        """ calculate the input gradient """
        raise NotImplementedError

    def connect_to(self, prev_layer):
        """Propagates the given input through this layer (and only this layer).

        Parameters
        ----------
        prev_layer : previous layer
            The previous layer to propagate through this layer.

        """
        raise NotImplementedError

    def to_json(self):
        """ To configuration """
        raise NotImplementedError

    @classmethod
    def from_json(cls, config):
        """ From configuration """
        return cls(**config)

    @property
    def params(self):
        """ Layer parameters. 
        
        Returns a list of numpy.array variables or expressions that
        parameterize the layer.

        Returns
        -------
        list of numpy.array variables or expressions
            A list of variables that parameterize the layer

        Notes
        -----
        For layers without any parameters, this will return an empty list.
        """
        return []

    @property
    def grads(self):
        """ Get layer parameter gradients as calculated from backward(). """
        return []

    @property
    def param_grads(self):
        """ Layer parameters and corresponding gradients. """
        return list(zip(self.params, self.grads))

    def __str__(self):
        return self.__class__.__name__





_rng = np.random
_dtype = 'float32'
def get_dtype():
    """Get data dtype ``numpy.dtype``.

    Returns
    -------
    str or numpy.dtype
    """
    return _dtype

def get_rng():
    """Get the package-level random number generator.

    Returns
    -------
    :class:`numpy.random.RandomState` instance
        The :class:`numpy.random.RandomState` instance passed to the most
        recent call of :func:`set_rng`, or ``numpy.random`` if :func:`set_rng`
        has never been called.
    """
    return _rng



class Linear(Layer):
    """A fully connected layer implemented as the dot product of inputs and
    weights.

    Parameters
    ----------
    n_out : (int, tuple)
        Desired size or shape of layer output
    n_in : (int, tuple) or None
        The layer input size feeding into this layer
    init : (Initializer, optional)
        Initializer object to use for initializing layer weights
    """

    def __init__(self, n_out, n_in=None, init='glorot_uniform'):
        self.n_out = n_out
        self.n_in = n_in
        self.out_shape = (None, n_out)
        self.init = get_init(init)

        self.W = None
        self.b = None
        self.dW = None
        self.db = None
        self.last_input = None

    def connect_to(self, prev_layer=None):
        if prev_layer is None:
            assert self.n_in is not None
            n_in = self.n_in
        else:
            assert len(prev_layer.out_shape) == 2
            n_in = prev_layer.out_shape[-1]

        self.W = self.init((n_in, self.n_out))
        self.b = _zero((self.n_out,))

    def forward(self, input, *args, **kwargs):
        """ Apply the forward pass transformation to the input data.
        
        Parameters
        ----------
        input : numpy.array
            input data
        
        Returns
        -------
        numpy.array
            output data
        """
        self.last_input = input
        return np.dot(input, self.W) + self.b

    def backward(self, pre_grad, *args, **kwargs):
        """Apply the backward pass transformation to the input data.
        
        Parameters
        ----------
        pre_grad : numpy.array
            deltas back propagated from the adjacent higher layer
            
        Returns
        -------
        numpy.array
            deltas to propagate to the adjacent lower layer
        """
        self.dW = np.dot(self.last_input.T, pre_grad)
        self.db = np.mean(pre_grad, axis=0)
        if not self.first_layer:
            return np.dot(pre_grad, self.W.T)

    @property
    def params(self):
        return self.W, self.b

    @property
    def grads(self):
        return self.dW, self.db


class Dense(Layer):
    """A fully connected layer implemented as the dot product of inputs and
    weights. Generally used to implemenent nonlinearities for layer post activations.

    Parameters
    ----------
    n_out : int
        Desired size or shape of layer output
    n_in : int, or None
        The layer input size feeding into this layer
    activation : str, or npdl.activatns.Activation
        Defaults to ``Tanh``
    init : str, or npdl.initializations.Initializer
        Initializer object to use for initializing layer weights
    """

    def __init__(self, n_out, n_in=None, init='glorot_uniform', activation='tanh'):
        self.n_out = n_out
        self.n_in = n_in
        self.out_shape = (None, n_out)
        self.init = get_init(init)
        self.act_layer = activations.get(activation)

        self.W, self.dW = None, None
        self.b, self.db = None, None
        self.last_input = None

    def connect_to(self, prev_layer=None):
        if prev_layer is None:
            assert self.n_in is not None
            n_in = self.n_in
        else:
            assert len(prev_layer.out_shape) == 2
            n_in = prev_layer.out_shape[-1]

        self.W = self.init((n_in, self.n_out))
        self.b = _zero((self.n_out,))

    def forward(self, input, *args, **kwargs):
        """ Apply the forward pass transformation to the input data.

        Parameters
        ----------
        input : numpy.array
            input data

        Returns
        -------
        numpy.array
            output data
        """
        self.last_input = input
        linear_out = np.dot(input, self.W) + self.b
        act_out = self.act_layer.forward(linear_out)
        
        # print(f"softmax-{act_out.shape}")
        return act_out

    def backward(self, pre_grad, *args, **kwargs):
        """Apply the backward pass transformation to the input data.

        Parameters
        ----------
        pre_grad : numpy.array
            deltas back propagated from the adjacent higher layer

        Returns
        -------
        numpy.array
            deltas to propagate to the adjacent lower layer
        """
        act_grad = pre_grad * self.act_layer.derivative()
        self.dW = np.dot(self.last_input.T, act_grad)
        self.db = np.mean(act_grad, axis=0)
        if not self.first_layer:
            return np.dot(act_grad, self.W.T)

    @property
    def params(self):
        return self.W, self.b

    @property
    def grads(self):
        return self.dW, self.db


class Softmax(Dense):
    """A fully connected layer implemented as the dot product of inputs and
    weights.

    Parameters
    ----------
    n_out : int
        Desired size or shape of layer output
    n_in : int, or None
        The layer input size feeding into this layer
    init : str, or npdl.initializations.Initializer
        Initializer object to use for initializing layer weights
    """

    def __init__(self, n_out, n_in=None, init='glorot_uniform'):
        super(Softmax, self).__init__(n_out, n_in, init, activation='softmax')


class Dropout(Layer):
    """A dropout layer.

    Applies an element-wise multiplication of inputs with a keep mask.

    A keep mask is a tensor of ones and zeros of the same shape as the input.

    Each :meth:`forward` call generates an new keep mask stochastically where there
    distribution of ones in the mask is controlled by the keep param.

    Parameters
    ----------
    p : float
        fraction of the inputs that should be stochastically kept.
    
    """

    def __init__(self, p=0.):
        self.p = p

        self.last_mask = None
        self.out_shape = None

    def connect_to(self, prev_layer):
        self.out_shape = prev_layer.out_shape

    def forward(self, input, train=True, *args, **kwargs):
        """Apply the forward pass transformation to the input data.

        Parameters
        ----------
        input : numpy.array
            input data
        train : bool
            is inference only

        Returns
        -------
        numpy.array
            output data
        """
        if 0. < self.p < 1.:
            if train:
                self.last_mask = get_rng().binomial(1, 1 - self.p, input.shape) / (1 - self.p)
                return input * self.last_mask
            else:
                return input * (1 - self.p)
        else:
            return input

    def backward(self, pre_grad, *args, **kwargs):
        if 0. < self.p < 1.:
            return pre_grad * self.last_mask
        else:
            return pre_grad




#形状
class Flatten(Layer):
    def __init__(self, outdim=2):
        self.outdim = outdim
        if outdim < 1:
            raise ValueError('Dim must be >0, was %i', outdim)

        self.last_input_shape = None
        self.out_shape = None

    def connect_to(self, prev_layer):
        assert len(prev_layer.out_shape) > 2

        to_flatten = np.prod(prev_layer.out_shape[self.outdim - 1:])
        flattened_shape = prev_layer.out_shape[:self.outdim - 1] + (to_flatten,)

        self.out_shape = flattened_shape

    def forward(self, input, *args, **kwargs):
        self.last_input_shape = input.shape

        # to_flatten = np.prod(self.last_input_shape[self.outdim-1:])
        # flattened_shape = input.shape[:self.outdim-1] + (to_flatten, )
        flattened_shape = input.shape[:self.outdim - 1] + (-1,)
        
        # print(f"flatten-{np.reshape(input, flattened_shape).shape}")
        return np.reshape(input, flattened_shape)

    def backward(self, pre_grad, *args, **kwargs):
        return np.reshape(pre_grad, self.last_input_shape)


class DimShuffle(Layer):
    def __init__(self, axis=1):
        self.axis = axis
        if axis < 0:
            raise ValueError('Dim must be > 0, bug get {}'.format(axis))

        self.last_input_shape = None
        self.out_shape = None

    def connect_to(self, prev_layer):
        assert len(prev_layer.out_shape) >= self.axis
        self.out_shape = prev_layer.out_shape[:self.axis] + (1,) + prev_layer.out_shape[self.axis:]

    def forward(self, input, *args, **kwargs):
        self.last_input_shape = input.shape
        return np.expand_dims(input, axis=self.axis)

    def backward(self, pre_grad, *args, **kwargs):
        return np.reshape(pre_grad, self.last_input_shape)


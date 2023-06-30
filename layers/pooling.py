# -*- coding: utf-8 -*-
import numpy as np

from layers.core import Layer
from initializations import _zero


class MeanPooling(Layer):
    
    def __init__(self, pool_size):
        self.pool_size = pool_size

        self.out_shape = 0
        self.out_shape = None
        self.input_shape = None

    def connect_to(self, prev_layer):
        assert 5 > len(prev_layer.out_shape) >= 3

        # print(f"Input shape: {prev_layer.out_shape}")
        old_h, old_w = prev_layer.out_shape[-2:]
        pool_h, pool_w = self.pool_size
        new_h, new_w = old_h // pool_h, old_w // pool_w

        assert old_h % pool_h == old_w % pool_w == 0

        self.out_shape = prev_layer.out_shape[:-2] + (new_h, new_w)

    def forward(self, input, *args, **kwargs):

        # shape
        self.input_shape = input.shape
        pool_h, pool_w = self.pool_size
        new_h, new_w = self.out_shape[-2:]

        # forward
        outputs = _zero(self.input_shape[:-2] + self.out_shape[-2:])

        if np.ndim(input) == 4:
            nb_batch, nb_axis, _, _ = input.shape

            for a in np.arange(nb_batch):
                for b in np.arange(nb_axis):
                    for h in np.arange(new_h):
                        for w in np.arange(new_w):
                            outputs[a, b, h, w] = np.mean(input[a, b, h:h + pool_h, w:w + pool_w])

        elif np.ndim(input) == 3:
            nb_batch, _, _ = input.shape

            for a in np.arange(nb_batch):
                for h in np.arange(new_h):
                    for w in np.arange(new_w):
                        outputs[a, h, w] = np.mean(input[a, h:h + pool_h, w:w + pool_w])

        else:
            raise ValueError()

        # print(f"pooling-{outputs.shape}")
        return outputs

    def backward(self, pre_grad, *args, **kwargs):
        new_h, new_w = self.out_shape[-2:]
        pool_h, pool_w = self.pool_size
        length = np.prod(self.pool_size)

        layer_grads = _zero(self.input_shape)

        if np.ndim(pre_grad) == 4:
            nb_batch, nb_axis, _, _ = pre_grad.shape

            for a in np.arange(nb_batch):
                for b in np.arange(nb_axis):
                    for h in np.arange(new_h):
                        for w in np.arange(new_w):
                            h_shift, w_shift = h * pool_h, w * pool_w
                            layer_grads[a, b, h_shift: h_shift + pool_h, w_shift: w_shift + pool_w] = \
                                pre_grad[a, b, h, w] / length

        elif np.ndim(pre_grad) == 3:
            nb_batch, _, _ = pre_grad.shape

            for a in np.arange(nb_batch):
                for h in np.arange(new_h):
                    for w in np.arange(new_w):
                        h_shift, w_shift = h * pool_h, w * pool_w
                        layer_grads[a, h_shift: h_shift + pool_h, w_shift: w_shift + pool_w] = \
                            pre_grad[a, h, w] / length

        else:
            raise ValueError()

        return layer_grads



import numpy as np
import theano.tensor as T

from lasagne import init
from lasagne.init import Initializer
from lasagne.layers.base import Layer
from lasagne.utils import floatX
from lasagne.random import get_rng


__all__ = [
    "InvBatchNormLayer",
]

class InvBatchNormLayer(Layer):
    """
    """
    def __init__(self, incoming, axes = 'auto', beta=init.Constant(0), gamma=init.Constant(1),
                 mean=init.Constant(0), inv_std=init.Constant(1),
                 **kwargs):
        super(InvBatchNormLayer, self).__init__(incoming, **kwargs)

        if axes == 'auto':
            # default: normalize over all but the second axis
            axes = (0,) + tuple(range(2, len(self.input_shape)))
        elif isinstance(axes, int):
            axes = (axes,)
        self.axes = axes

        shape = [size for axis, size in enumerate(self.input_shape)
                 if axis not in self.axes]
        self.beta = self.add_param(beta, shape, 'beta', trainable=True, regularizable=False)
        self.gamma = self.add_param(gamma, shape, 'gamma', trainable=True, regularizable=True)
        self.mean = self.add_param(mean, shape, 'mean', trainable=False, regularizable=False)
        self.inv_std = self.add_param(inv_std, shape, 'inv_std', trainable=False, regularizable=False)

    def get_output_shape_for(self, input_shape):
        return input_shape

    def get_output_for(self, input, **kwargs):
        eps = 1e-9
        q = (input - self.beta)/(self.gamma+eps * self.inv_std+eps) + self.mean
        return q + eps

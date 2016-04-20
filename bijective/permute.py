import numpy as np
import theano.tensor as T

from lasagne import init
from lasagne.init import Initializer
from lasagne.layers.base import Layer
from lasagne.utils import floatX
from lasagne.random import get_rng


__all__ = [
    "PermuteLayer",
] 

class Eye(Initializer):
    """Initialize weights with constant value.

    Parameters
    ----------
     val : float
        Constant value for weights.
    """
    def __init__(self, val=0.0):
        self.val = val

    def sample(self, shape):
        return floatX(np.eye(*shape))

class UniformLen(Initializer):
    """Sample initial weights from the uniform distribution.

    Parameters are sampled from U(a, b).

    Parameters
    ----------
    """
    def sample(self, shape):
        nelem = shape[0]
        return floatX(get_rng().uniform(
            low=0, high=nelem, size=shape))

class Arange(Initializer):
    """Sample initial weights from the uniform distribution.

    Parameters are sampled from U(a, b).

    Parameters
    ----------
    """
    def sample(self, shape):
        nelem = shape[0]
        return floatX(np.arange(nelem))

class PermuteLayer(Layer):
    """
    lasagne.layers.PermuteLayer(incoming, num_units,
    W=lasagne.init.GlorotUniform(), b=lasagne.init.Constant(0.),
    nonlinearity=lasagne.nonlinearities.rectify, **kwargs)

    A fully connected layer.

    Parameters
    ----------
    incoming : a :class:`Layer` instance or a tuple
        The layer feeding into this layer, or the expected input shape

    num_units : int
        The number of units of the layer

    W : Theano shared variable, expression, numpy array or callable
        Initial value, expression or initializer for the weights.
        These should be a matrix with shape ``(num_inputs, num_units)``.
        See :func:`lasagne.utils.create_param` for more information.

    Notes
    -----
    If the input to this layer has more than two axes, it will flatten the
    trailing axes. This is useful for when a dense layer follows a
    convolutional layer, for example. It is not necessary to insert a
    :class:`FlattenLayer` in this case.
    """
    def __init__(self, incoming, p=UniformLen(), k=1,
                 **kwargs):
        super(PermuteLayer, self).__init__(incoming, **kwargs)
        self.x_len = np.prod(self.input_shape[1:])
        self.p = self.add_param(p, (self.x_len,), name = "p", regularizable=False)
        self.k = k

    def get_output_shape_for(self, input_shape):
        # Permutation does not change shape
        return input_shape

    def get_output_for(self, input, **kwargs):
        p = self.p
        k = self.k
        nbatches = input.shape[0]
        x_len = self.x_len
        # x_len = 30
        # x = input.reshape((nbatches, x_len))
        x = input.reshape((nbatches, x_len))

        p_floor = T.floor(p)
        p_ceil = T.ceil(p)
        
        # Deltas
        p_delta = p - p_floor
        ep_delta = T.exp(k*-p_delta)

        p2_delta = 1 - p_delta
        ep2_delta = T.exp(k*-p2_delta)

        p0_delta = 1 + p_delta
        ep0_delta = T.exp(k*-p0_delta)

        ep_sum = ep_delta + ep2_delta + ep0_delta

        perm1 = x[:, (T.cast(p_floor, 'int32'))%x_len]
        perm2 = x[:, (T.cast(p_ceil, 'int32')+1)%x_len]
        perm0 = x[:, (T.cast(p_floor, 'int32')-1)%x_len]

        perm1_factor = ep_delta * perm1
        perm2_factor = ep2_delta * perm2
        perm3_factor = ep0_delta * perm0
        res = (perm1_factor + perm2_factor + perm3_factor) / ep_sum
        return res.reshape(input.shape)
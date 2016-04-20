import lasagne
from permute import PermuteLayer
from theano import function
import numpy as np
import theano
theano.config.optimizer = 'None'

x = lasagne.layers.InputLayer((None,3,3))
y = PermuteLayer(x,k=10)
weights = lasagne.layers.get_all_param_values(y)
f = function([x.input_var], lasagne.layers.get_output(y))
data = np.arange(9).reshape(1,3,3)
f(data)
# f(np.random.rand(4,10,3))
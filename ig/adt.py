
# abstract data types

import theano
import theano.tensor as T
import lasagne
from lasagne.layers import InputLayer
from lasagne.layers import DenseLayer
from lasagne.layers import NonlinearityLayer
from lasagne.layers import ConcatLayer
from lasagne.layers import DropoutLayer
from lasagne.layers import Pool2DLayer as PoolLayer
from lasagne.layers import Conv1DLayer
# from lasagne.layers.dnn import Conv2DDNNLayer as ConvLayer
from lasagne.nonlinearities import softmax

from theano import shared
from theano import function
theano.config.optimizer = 'None'



def mse(a, b):
    eps = 1e-9
    return (T.maximum(eps, (a - b)**2)).mean()

# Square error
def dist(a, b):
    eps = 1e-9
    return T.sum(T.maximum(eps, (a - b)**2))

# Push(Stack, Item) -> Stack
def push(input_stack, input_item):
  push_net = {}
  push_net['combine'] = ConcatLayer([input_stack, input_item])
  push_net['l1'] = DenseLayer(push_net['combine'], nstack_reals,
    nonlinearity = lasagne.nonlinearities.rectify, W=lasagne.init.HeNormal(gain='relu'))
  push_net['l2'] = DenseLayer(push_net['l1'] , nstack_reals,
    nonlinearity = lasagne.nonlinearities.rectify, W=lasagne.init.HeNormal(gain='relu'))
  push_net['l3'] = DenseLayer(push_net['l2'], nstack_reals,
    nonlinearity = lasagne.nonlinearities.rectify, W=lasagne.init.HeNormal(gain='relu'))
  return push_net, push_net['l3']

# Push(Stack, Item) -> Stack
def pop(input_stack):
  pop_net = {}
  pop_net['l1'] = DenseLayer(input_stack, nstack_reals,
    nonlinearity = lasagne.nonlinearities.rectify, W=lasagne.init.HeNormal(gain='relu'))
  pop_net['l2'] = DenseLayer(pop_net['l1'] , nstack_reals+nitem_reals,
    nonlinearity = lasagne.nonlinearities.rectify, W=lasagne.init.HeNormal(gain='relu'))
  pop_net['l3'] = DenseLayer(pop_net['l2'], nstack_reals+nitem_reals,
    nonlinearity = lasagne.nonlinearities.rectify, W=lasagne.init.HeNormal(gain='relu'))
  return pop_net, pop_net['l3']

nitem_reals = 1
nbatch = 512
nstack_reals = 1000
# empty_stack = shared(np.random.rand(nstack_reals))

input_stack = InputLayer((nbatch, nstack_reals))
input_item = InputLayer((nbatch, 1))
## Push
push_net, push_net_last_layer = push(input_stack, input_item)
push_net_op = lasagne.layers.get_output(push_net_last_layer)

## Specification
## =============

# is_empty(empty_stack) = true
# is_empty_loss = dist(is_empty(empty_stack), empty_stack)

## Axiom 2: Forall stacks, items, pop(push(stack, item)) = stack, items
pop_net, pop_net_last_layer = pop(push_net_last_layer)
pop_net_op = lasagne.layers.get_output(pop_net_last_layer)
pop_op_stack = pop_net_op[:, 0:nstack_reals]
pop_op_item = pop_net_op[:, nstack_reals:]

ax2_loss1 = mse(pop_op_stack, input_stack.input_var)
ax2_loss2 = mse(pop_op_item, input_item.input_var)
# loss = ax2_loss1
## Axiom: L

loss = ax2_loss2 + ax2_loss1
params =  lasagne.layers.get_all_params(pop_net_last_layer,  trainable=True)
updates = lasagne.updates.adam(loss, params, learning_rate = 0.0001)
# updates = lasagne.updates.momentum(loss, params, learning_rate = 0.01)
f = function([input_stack.input_var, input_item.input_var], [loss, ax2_loss1, ax2_loss2], updates = updates)


# for i in range(100):
import numpy as np
input_stack_data = np.random.rand(nbatch, nstack_reals)
input_item_data = np.random.randn(nbatch, 1)
f(input_stack_data, input_item_data)

# for i in range(10000):
#     input_stack_data = np.random.rand(nbatch, nstack_reals)
#     input_item_data = np.random.randn(nbatch, 1)
#     [loss, loss1, loss2] = f(input_stack_data, input_item_data)
#     print i, loss, loss1, loss2
#
# push_func = function([input_stack.input_var, input_item.input_var], push_net_op)
# pop_func = function([input_stack.input_var, ])

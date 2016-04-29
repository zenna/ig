
# abstract data types

from __future__ import print_function

import time

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
from lasagne.layers import batch_norm
from lasagne.utils import floatX

# def batch_norm(x): return x


# from lasagne.layers.dnn import Conv2DDNNLayer as ConvLayer
from lasagne.nonlinearities import softmax, rectify, sigmoid
from theano import shared
from theano import function
from theano import config

import numpy as np
from mnist import *


config.optimizer = 'None'
# config.optimizer = 'fast_compile'

def sigmoid(y, tnp = T):
  return tnp.minimum(tnp.maximum(0, y),1)

def bound_loss(x, tnp = T) :
  eps = 1e-9
  loss = tnp.maximum(tnp.maximum(eps,x-1), tnp.maximum(eps,-x)) + eps
  return tnp.maximum(loss, eps) + eps

def mse(a, b):
    eps = 1e-9
    return (T.maximum(eps, (a - b)**2)).mean()

# Square error
def dist(a, b):
    eps = 1e-9
    return T.sum(T.maximum(eps, (a - b)**2))

# Push(Stack, Item) -> Stack
def push(input_stack, input_item, stack_size, n_blocks = 7, block_size = 2):
  push_net = {}
  push_net['combine'] = prev_layer = ConcatLayer([input_stack, input_item])
  layer_width = stack_size
  print("pop layer_width", layer_width)
  wx = batch_norm(DenseLayer(prev_layer, layer_width, nonlinearity = rectify, W=lasagne.init.HeNormal(gain='relu')))
  for j in range(n_blocks):
    for i in range(block_size):
      push_net['res2d%s_%s' % (j,i)] = prev_layer = batch_norm(DenseLayer(prev_layer, layer_width,
            nonlinearity = rectify, W=lasagne.init.HeNormal(gain='relu')))
    push_net['block%s' % j] = prev_layer = wx = lasagne.layers.ElemwiseSumLayer([prev_layer, wx])

  # push_net['final'] = prev_layer = wx = lasagne.layers.NonlinearityLayer(prev_layer, nonlinearity=rectify)
  return push_net, wx


# Push(Stack, Item) -> Stack
def pop(input_stack, stack_size, item_size, n_blocks = 7, block_size = 2):
  pop_net = {}
  prev_layer = x = input_stack
  layer_width = stack_size+item_size
  print("layer_width", layer_width)
  wx = batch_norm(DenseLayer(x, layer_width, nonlinearity = rectify, W=lasagne.init.HeNormal(gain='relu')))

  for j in range(n_blocks):
    for i in range(block_size):
      pop_net['res2d%s_%s' % (j,i)] = prev_layer = batch_norm(DenseLayer(prev_layer, layer_width,
            nonlinearity = rectify, W=lasagne.init.HeNormal(gain='relu')))
    pop_net['block%s' % j] = prev_layer = wx = lasagne.layers.ElemwiseSumLayer([prev_layer, wx])

  # pop_net['final'] = prev_layer = wx = lasagne.layers.NonlinearityLayer(prev_layer, nonlinearity=rectify)
  return pop_net, wx

def main(X_train, stack_size = 100, nbatch = 256, item_size = 28*28, num_epochs = 100):
    # empty_stack = shared(np.random.rand(stack_size))
    global params, push_net, push_net_last_layer, push_func, pop_net, pop_net_last_layer
    input_stack = InputLayer((nbatch, stack_size))
    input_item = InputLayer((nbatch, item_size))
    ## Push
    push_net, push_net_last_layer = push(input_stack, input_item, stack_size)
    push_net_op = lasagne.layers.get_output(push_net_last_layer)
    push_func = function([input_stack.input_var, input_item.input_var], push_net_op)

    ## Specification
    ## =============

    # is_empty(empty_stack) = true
    # is_empty_loss = dist(is_empty(empty_stack), empty_stack)

    ## Axiom 2: Forall stacks, items, pop(push(stack, item)) = stack, items
    pop_net, pop_net_last_layer = pop(push_net_last_layer, stack_size, item_size)
    pop_net_op = lasagne.layers.get_output(pop_net_last_layer)
    pop_op_stack = pop_net_op[:, 0:stack_size]
    pop_op_item = pop_net_op[:, stack_size:]

    ax2_loss1 = mse(pop_op_stack, input_stack.input_var)
    ax2_loss2 = mse(pop_op_item, input_item.input_var)
    # loss = ax2_loss1
    ## Axiom: L

    # Boundary losses
    b_loss1 = bound_loss(push_net_op, tnp=T).mean()
    b_loss2 = bound_loss(pop_op_stack, tnp=T).mean()

    loss = ax2_loss2 + ax2_loss1 + b_loss1 + b_loss2
    params =  lasagne.layers.get_all_params(pop_net_last_layer,  trainable=True)
    updates = lasagne.updates.adam(loss, params, learning_rate = 0.001)
    # updates = lasagne.updates.momentum(loss, params, learning_rate = 0.1)
    # updates = lasagne.updates.rmsprop(loss, params, learning_rate = 0.001)
    f_train = function([input_stack.input_var, input_item.input_var], [loss, ax2_loss1, ax2_loss2, b_loss1, b_loss2], updates = updates)

    print("Starting training...")
    for epoch in range(num_epochs):
        # In each epoch, we do a full pass over the training data:
        train_err = 0
        train_batches = 0
        start_time = time.time()
        for batch in iterate_minibatches(X_train, nbatch, shuffle=True):
            input_item_data = batch.reshape(nbatch, item_size)
            input_stack_data = np.array(np.random.rand(nbatch, stack_size), dtype=config.floatX)
            losses = f_train(input_stack_data, input_item_data)
            print(losses)
            train_err += losses[0]
            train_batches += 1


def iterate_minibatches(inputs, batchsize, shuffle=False):
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt]

def testing():
    ## Testing
    push_func = function([input_stack.input_var, input_item.input_var], push_net_op)

    op_net_inp, pop_net_last_layer_inp = pop(input_stack)
    pop_net_op_inp = lasagne.layers.get_output(pop_net_last_layer_inp)
    pop_op_stack_inp = pop_net_op_inp[:, 0:stack_size]
    pop_op_item_inp  = pop_net_op_inp[:, stack_size:]

    pop_param_values = lasagne.layers.get_all_param_values(pop_net_last_layer)
    npop_parmas = len(lasagne.layers.get_all_params(pop_net_last_layer_inp))
    pop_inp_param_values = pop_param_values[-npop_parmas:]
    lasagne.layers.set_all_param_values(pop_net_last_layer_inp, pop_inp_param_values)
    pop_func = function([input_stack.input_var], [pop_op_stack_inp,pop_op_item_inp])

    ## Testing
    new_stack_data =  push_func(input_stack_data, input_item_data)
    shoud_be_old_stack, should_be_old_item = pop_func(new_stack_data)

X_train, y_train, X_val, y_val, X_test, y_test = load_dataset()
main(X_train, stack_size = 100, nbatch = 256, item_size = 28*28)

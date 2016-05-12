from __future__ import print_function

import time

from templates import *
from ig.io import *
from ig.util import *

import theano
import theano.tensor as T
import lasagne
# def batch_norm(x, **kwargs): return x
from lasagne.utils import floatX

# from lasagne.layers.dnn import Conv2DDNNLayer as ConvLayer
from theano import shared
from theano import function
from theano import config

import numpy as np

# Mean square error
def mse(a, b, tnp = T):
    eps = 1e-9
    return (tnp.maximum(eps, (a - b)**2)).mean()

def infinite_samples(sampler, batchsize, shape):
    while True:
        to_sample_shape = (batchsize,)+shape
        yield lasagne.utils.floatX(sampler(*to_sample_shape))

def infinite_minibatches(inputs, batchsize, shuffle=False):
    start_idx = 0
    nelements = len(inputs)
    indices = np.arange(nelements)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    while True:
        end_idx = start_idx + batchsize
        if end_idx > nelements:
            diff = end_idx - nelements
            excerpt = np.concatenate([indices[start_idx:nelements], indices[0:diff]])
            start_idx = diff
            if shuffle:
                indices = np.arange(len(inputs))
                np.random.shuffle(indices)
        else:
            excerpt = indices[start_idx:start_idx + batchsize]
            start_idx = start_idx + batchsize
        yield inputs[excerpt]

def iterate_minibatches(inputs, batchsize, shuffle=False):
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_iiteratedx + batchsize)
        yield inputs[excerpt]

class Type():
    def __init__(self, shape, dtype = T.config.floatX, name = ''):
        self.shape = shape
        self.dtype = dtype
        self.tensor_id = 0
        self.name = name

    def tensor(self, name = None, add_batch=False):
        if name == None:
            name = "%s_%s" % (self.name, self.tensor_id)
            self.tensor_id += 1
        # Create a tensor for this shape
        ndims = len(self.shape)
        if add_batch:
            ndims += 1
        return T.TensorType(dtype=self.dtype, broadcastable=(False,)*ndims)(name)

    def get_shape(self, add_batch = False, batch_size = None):
        if add_batch:
            return (batch_size,) + self.shape
        else:
            return self.shape

class Interface():
    def __init__(self, lhs, rhs, func_space, **func_space_kwargs):
        self.lhs = lhs
        self.rhs = rhs
        self.func_space = func_space
        self.func_space_kwargs = func_space_kwargs
        self.inputs = [type.tensor(add_batch=True) for type in lhs]
        print(self.inputs[0].ndim)
        params = Params()
        self.inp_shapes = [type.get_shape(add_batch=True) for type in lhs]
        self.out_shapes = [type.get_shape(add_batch=True) for type in rhs]
        outputs, params = func_space(*self.inputs, deterministic = True, params = params, inp_shapes = self.inp_shapes, out_shapes = self.out_shapes, **self.func_space_kwargs)
        self.params = params
        self.outputs = outputs

    def __call__(self, *raw_args):
        args = [arg.input_var if hasattr(arg, 'input_var') else arg for arg in raw_args]
        print("Calling", args)
        shapes = [type.get_shape(add_batch=True) for type in self.lhs]
        outputs, params = self.func_space(*args, deterministic = False, params = self.params, inp_shapes = self.inp_shapes, out_shapes = self.out_shapes, **self.func_space_kwargs)
        return outputs

    def load_params(self, param_values):
        params = self.params.get_params()
        lasagne.layers.set_all_param_values(params, param_values)

    def load_params_fname(self, fname):
        params_file = np.load(fname)
        param_values = npz_to_array(params_f)
        return load_params(param_values)

    def save_params(self, fname):
        params = self.params.get_params()
        param_values = [param.get_value() for param in params]
        np.savez_compressed(fname, *param_values)

    def compile(self):
        print("Compiling interface")
        call_fn = function(self.inputs, self.outputs)
        return call_fn

class ForAllVar():
    "Universally quantified variable"
    def __init__(self, type):
        self.type = type
        self.input_var = type.tensor(add_batch=True)

class Axiom():
    def __init__(self, lhs, rhs, name=''):
        assert len(lhs) == len(rhs)
        self.lhs = lhs
        self.rhs = rhs

    def get_losses(self, dist = mse):
        losses = [dist(self.lhs[i], self.rhs[i]) for i in range(len(self.lhs))]
        return losses

def bound_loss(x, tnp = T) :
  eps = 1e-9
  loss = tnp.maximum(tnp.maximum(eps,x-1), tnp.maximum(eps,-x)) + eps
  return tnp.maximum(loss, eps) + eps

class BoundAxiom():
    "Constraints a type to be within specifiec bounds"
    def __init__(self, type, name = 'bound_loss'):
        self.input_var = type

    def get_losses(self):
        return [bound_loss(self.input_var).mean()]

# class Constant():
#     def __init__(self, value):
#         self.value = shared(value)

class Params():
    def __init__(self):
        self.params = {}
        self.is_locked = False

    def lock(self):
        self.is_locked = True

    def check(self, params):
        # FIXME, implement check to see all parameters are there
        return True

    def __getitem__(self, key_default_value):
        key, default_value = key_default_value
        return self.get(key, default_value)

    def get(self, key, default_value):
        if self.params.has_key(key):
            return self.params[key]
        else:
            assert not self.is_locked, "Attempted to create parameter from locked params"
            param = default_value
            self.params[key] = param
            return param

    def set(self, key, value):
        if self.params.has_key(key):
            self.params[key] = value
        else:
            print("Setting value before generated")
            exit(1)

    def add_tagged_params(self, tagged_params):
        self.tagged_params = tagged_params

    def get_params(self, **tags):
        result = list(self.tagged_params.keys())
        only = set(tag for tag, value in tags.items() if value)
        if only:
            # retain all parameters that have all of the tags in `only`
            result = [param for param in result
                      if not (only - self.tagged_params[param])]

        exclude = set(tag for tag, value in tags.items() if not value)
        if exclude:
            # retain all parameters that have none of the tags in `exclude`
            result = [param for param in result
                      if not (self.tagged_params[param] & exclude)]

        return lasagne.utils.collect_shared_vars(result)

def get_updates(loss, params, options):
    updates = {}
    print("Params",params)
    if options['update'] == 'momentum':
        updates = lasagne.updates.momentum(loss, params, learning_rate=options['learning_rate'], momentum=options['momentum'])
    elif options['update'] == 'adam':
        updates = lasagne.updates.adam(loss, params, learning_rate=options['learning_rate'])
    elif options['update'] == 'rmsprop':
        updates = lasagne.updates.rmsprop(loss, params, learning_rate=options['learning_rate'])
    return updates

def get_losses(axioms):
    losses = []
    for axiom in axioms:
        for loss in axiom.get_losses():
            losses.append(loss)
    return losses

def get_params(interfaces, options, **tags):
    params = []
    for interface in interfaces:
        for param in interface.params.get_params(**tags):
            params.append(param)

    return params

def compile_fns(interfaces, forallvars, axioms, options):
    print("Compiling training fn...")
    losses = get_losses(axioms)
    params = get_params(interfaces, options, trainable=True)
    loss = sum(losses)
    updates = get_updates(loss, params, options)
    train_fn = function([forallvar.input_var for forallvar in forallvars], losses, updates = updates)
    # Compile the interface for use
    if options['compile_fns']:
        print("Compiling interface fns...")
        call_fns = [interface.compile() for interface in interfaces]
    else:
        call_fns = []
    #FIXME Trainable=true, deterministic = true/false
    return train_fn, call_fns

def train(train_fn, generators, num_epochs = 1000, summary_gap = 100):
    """One epoch is one pass through the data set"""
    print("Starting training...")
    for epoch in range(num_epochs):
        train_err = 0
        train_batches = 0
        start_time = time.time()
        for i in range(summary_gap):
            inputs = [gen.next() for gen in generators]
            losses = train_fn(*inputs)
            print("epoch: ", epoch, "losses: ", losses)
            train_err += losses[0]
            train_batches += 1
        print("epoch: ", epoch, " Total loss per epoch: ", train_err)

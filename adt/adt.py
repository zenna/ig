from __future__ import print_function

import time

from templates import *
from ig.io import *
from ig.util import *
from ig.distances import *

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
import sys
sys.setrecursionlimit(40000)

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
        # output_args = {'batch_norm_update_averages' : True, 'batch_norm_use_averages' : True}
        output_args = {'deterministic' : True}
        outputs, params = func_space(*self.inputs, output_args = output_args, params = params, inp_shapes = self.inp_shapes, out_shapes = self.out_shapes, **self.func_space_kwargs)
        params.lock()
        self.params = params
        self.outputs = outputs

    def __call__(self, *raw_args):
        args = [arg.input_var if hasattr(arg, 'input_var') else arg for arg in raw_args]
        print("Calling", args)
        # shapes = [type.get_shape(add_batch=True) for type in self.lhs]
        # output_args = {'batch_norm_update_averages' : True, 'batch_norm_use_averages' : False}
        output_args = {'deterministic' : False}
        outputs, params = self.func_space(*args, output_args = output_args, params = self.params, inp_shapes = self.inp_shapes, out_shapes = self.out_shapes, **self.func_space_kwargs)
        return outputs

    def get_params(self, **tags):
        return self.params.get_params(**tags)

    def load_params(self, param_values):
        params = self.params.get_params()
        assert len(param_values) == len(params), "Tried to load invalid param file"
        for i in range(len(params)):
            params[i].set_value(param_values[i])

    def load_params_fname(self, fname):
        params_file = np.load(fname)
        param_values = npz_to_array(params_file)
        return self.load_params(param_values)

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

class BoundAxiom():
    "Constraints a type to be within specifiec bounds"
    def __init__(self, type, name = 'bound_loss'):
        self.input_var = type

    def get_losses(self):
        return [bound_loss(self.input_var).mean()]

class Constant():
    def __init__(self, type, spec=lasagne.init.GlorotUniform(), name = 'C'):
        self.type = type
        shape = type.get_shape(add_batch=True, batch_size=1)
        arr = spec(shape)
        arr = floatX(arr)
        assert arr.shape == shape
        broadcastable = (True,) + (False,) * (len(shape) - 1)
        # broadcastable = None
        self.input_var = theano.shared(arr, name=name, broadcastable=broadcastable)

    def get_params(self, **tags):
        return [self.input_var]

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
            # print("Retrieving Key")
            return self.params[key]
        else:
            assert not self.is_locked, "Attempted to create parameter from locked params"
            # print("Creating new key")
            param = default_value
            self.params[key] = param
            return param

    def set(self, key, value):
        if self.is_locked:
            # print("Not Setting, locked")
            return
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

class AbstractDataType():
    def __init__(self, interfaces, constants, forallvars, axioms, name = ''):
        self.interfaces = interfaces
        self.constants = constants
        self.forallvars = forallvars
        self.axioms = axioms
        self.name = name

class ProbDataType():
    """ A probabilistic data type gives a function (space) to each interfaces,
        a value to each constant and a random variable to each diti=rbution"""
    def __init__(self, adt, train_fn, call_fns, generators, gen_to_inputs, intermediates):
        self.adt = adt
        self.train_fn = train_fn
        self.call_fns = call_fns
        self.generators = generators
        self.gen_to_inputs = gen_to_inputs

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
        for param in interface.get_params(**tags):
            params.append(param)

    return params

def compile_fns(interfaces, constants, forallvars, axioms, intermediates, options):
    print("Compiling training fn...")
    losses = get_losses(axioms)
    interface_params = get_params(interfaces, options, trainable=True)
    constant_params = get_params(constants, options)
    params = interface_params + constant_params
    loss = sum(losses)
    outputs = intermediates + losses
    updates = get_updates(loss, params, options)
    train_fn = function([forallvar.input_var for forallvar in forallvars], outputs, updates = updates)
    # Compile the interface for use
    if options['compile_fns']:
        print("Compiling interface fns...")
        call_fns = [interface.compile() for interface in interfaces]
    else:
        call_fns = []
    #FIXME Trainable=true, deterministic = true/false
    return train_fn, call_fns

def train(train_fn, generators, gen_to_inputs = identity, nintermediates,
          num_epochs = 1000, summary_gap = 100):
    """One epoch is one pass through the data set"""
    print("Starting training...")
    for epoch in range(num_epochs):
        train_err = 0
        train_batches = 0
        start_time = time.time()
        intermediates = None
        [gen.next() for gen in generators]
        for i in range(summary_gap):
            gens = [gen.send(intermediates) for gen in generators]
            inputs = gen_to_inputs(gens)
            intermediates_losses = train_fn(*inputs)
            intermediates = intermediates_losses[0:nintermediates]
            losses = intermediates_losses[nintermediates:]
            print("epoch: ", epoch, "losses: ", losses)
            train_err += losses[0]
            train_batches += 1
            gens = [gen.next() for gen in generators]
        print("epoch: ", epoch, " Total loss per epoch: ", train_err)

def train_pbt(pbt, **kwargs):
    train(pbt.train_fn, pbt.generators, pbt.gen_to_inputs, len(pbt.intermediates),
          **kwargs)

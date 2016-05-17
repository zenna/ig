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
    def __init__(self, shape, dtype=T.config.floatX, name=''):
        self.shape = shape
        self.dtype = dtype
        self.tensor_id = 0
        self.name = name

    def tensor(self, name=None, add_batch=False):
        if name is None:
            name = "%s_%s" % (self.name, self.tensor_id)
            self.tensor_id += 1
        # Create a tensor for this shape
        ndims = len(self.shape)
        if add_batch:
            ndims += 1
        return T.TensorType(dtype=self.dtype,
                            broadcastable=(False,)*ndims)(name)

    def get_shape(self, add_batch=False, batch_size=None):
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
        # output_args = {'batch_norm_update_averages' : True,
        #                'batch_norm_use_averages' : True}
        output_args = {'deterministic': True}
        outputs, params = func_space(*self.inputs, output_args=output_args,
                                     params=params, inp_shapes=self.inp_shapes,
                                     out_shapes=self.out_shapes,
                                     **self.func_space_kwargs)
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
        print("Compiling func")
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

    def get_losses(self, dist=mse):
        print("lhs", self.lhs)
        print("rhs", self.rhs)
        losses = [dist(self.lhs[i], self.rhs[i]) for i in range(len(self.lhs))]
        return losses


class BoundAxiom():
    "Constraints a type to be within specifiec bounds"
    def __init__(self, type, name='bound_loss'):
        self.input_var = type

    def get_losses(self):
        return [bound_loss(self.input_var).mean()]


class Constant():
    def __init__(self, type, spec=lasagne.init.GlorotUniform(), name='C'):
        self.type = type
        shape = type.get_shape(add_batch=True, batch_size=1)
        arr = spec(shape)
        arr = floatX(arr)
        assert arr.shape == shape
        broadcastable = (True,) + (False,) * (len(shape) - 1)
        # broadcastable = None
        self.input_var = theano.shared(arr, name=name,
                                       broadcastable=broadcastable)

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
        if key in self.params:
            # print("Retrieving Key")
            return self.params[key]
        else:
            assert not self.is_locked, "Cant create param when locked"
            # print("Creating new key")
            param = default_value
            self.params[key] = param
            return param

    def set(self, key, value):
        if self.is_locked:
            # print("Not Setting, locked")
            return
        if key in self.params:
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
    def __init__(self, funcs, consts, forallvars, axioms, name=''):
        self.funcs = funcs
        self.consts = consts
        self.forallvars = forallvars
        self.axioms = axioms
        self.name = name


class ProbDataType():
    """ A probabilistic data type gives a function (space) to each funcs,
        a value to each constant and a random variable to each diti=rbution"""
    def __init__(self, adt, train_fn, call_fns, generators, gen_to_inputs,
                 train_outs):
        self.adt = adt
        self.train_fn = train_fn
        self.call_fns = call_fns
        self.generators = generators
        self.gen_to_inputs = gen_to_inputs
        self.train_outs = train_outs

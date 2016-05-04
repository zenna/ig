## The ways these problems differ
## 1. Representation of the type, basically what kind of tensor.
## 2. function space for each interface
## 3. generator for particular, types
## 5. May have dependent function spaces, e.g. parameter sharing.


from __future__ import print_function

import time
from collections import OrderedDict

from ig.io import *
from ig.util import *

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
# def batch_norm(x, **kwargs): return x
from lasagne.utils import floatX

from lasagne.init import HeNormal, Constant

# from lasagne.layers.dnn import Conv2DDNNLayer as ConvLayer
from lasagne.nonlinearities import softmax, rectify, sigmoid
from theano import shared
from theano import function
from theano import config

import numpy as np
from mnist import *

# theano.config.optimizer = 'fast_compile'
theano.config.optimizer = 'None'

# Mean square error
def mse(a, b):
    eps = 1e-9
    return (T.maximum(eps, (a - b)**2)).mean()

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

def handle_batch_norm(params, suffix, bn_layer):
  params.set("W_%s" % suffix, bn_layer.input_layer.input_layer.W)
  params.set("beta_%s" % suffix, bn_layer.input_layer.beta)
  params.set("gamma_%s" % suffix, bn_layer.input_layer.gamma)
  params.set("inv_std_%s" % suffix, bn_layer.input_layer.inv_std)

def res_net(*inputs, **kwargs):
  """A residual network of n inputs and m outputs"""
  inp_shapes = kwargs['inp_shapes']
  out_shapes = kwargs['out_shapes']
  params = kwargs['params']
  layer_width = kwargs['layer_width']

  input_width = np.sum([in_shape[1] for in_shape in inp_shapes])
  output_width = np.sum([out_shape[1] for out_shape in out_shapes])
  n_blocks = 5
  block_size = 2
  print("Building resnet with: %s residual blocks of size %s inner width: %s from: %s inputs to %s outputs" %
        (n_blocks, block_size, layer_width, input_width, output_width))
  input_layers = [InputLayer(inp_shapes[i], input_var = inputs[i]) for i in range(len(inputs))]

  net = {}
  net['concat'] = prev_layer = ConcatLayer(input_layers)
  # Projet inner layer down/up to hidden layer width only if necessary
  if layer_width != input_width:
      print("Doing input projection, layer_width: %s input_width: %s" % (layer_width, input_width))
      wx = batch_norm(DenseLayer(prev_layer, layer_width, nonlinearity = rectify, W=params['W_wx', HeNormal(gain='relu')]),
                                 beta=params['beta_wx', Constant(0)], gamma=params['gamma_wx', Constant(1)],
                                 mean=params['mean_wx', Constant(0)], inv_std=params['inv_std_wx', Constant(1)])
      handle_batch_norm(params, 'wx', wx)
  else:
      print("Skipping input weight projection, layer_width: %s input_width: %s" % (layer_width, input_width))
      wx = prev_layer
  for j in range(n_blocks):
    for i in range(block_size):
      sfx = "%s_%s" % (j,i)
      net['res2d%s_%s' % (j,i)] = prev_layer = batch_norm(DenseLayer(prev_layer, layer_width,
            nonlinearity = rectify, W=params['W_%s' % sfx, HeNormal(gain='relu')]),
            beta=params['beta_%s' % sfx, Constant(0)], gamma=params['gamma_%s' % sfx, Constant(1)],
            mean=params['mean_%s' % sfx, Constant(0)], inv_std=params['inv_std_%s' % sfx, Constant(1)])
      handle_batch_norm(params, sfx, prev_layer)
    net['block%s' % j] = prev_layer = wx = lasagne.layers.ElemwiseSumLayer([prev_layer, wx])

  if layer_width != output_width:
      print("Doing output projection, layer_width: %s output_width: %s" % (layer_width, output_width))
      net['output'] = batch_norm(DenseLayer(prev_layer, output_width, nonlinearity = rectify, W=params['W_out', HeNormal(gain='relu')]),
                                 beta=params['beta_out', Constant(0)], gamma=params['gamma_out', Constant(0)],
                                 mean=params['mean_out', Constant(1)], inv_std=params['inv_std_out', Constant(1)])
      handle_batch_norm(params, 'out', net['output'])
  else:
      print("Skipping output projection, layer_width: %s output_width: %s" % (layer_width, output_width))
      net['output'] = prev_layer

  # Split up the final layer into necessary parts
  output_product = lasagne.layers.get_output(net['output'])
  outputs = []
  lb = 0
  for out_shape in out_shapes:
      ub = lb + out_shape[1]
      print("lbub", lb," ", ub)
      out = output_product[:, lb:ub]
      outputs.append(out)
      lb = ub

  params.add_tagged_params(get_layer_params(lasagne.layers.get_all_layers(net['output'])))
  params.check(lasagne.layers.get_all_params(prev_layer))
  return outputs, params

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
        outputs, params = func_space(*self.inputs, params = params, inp_shapes = self.inp_shapes, out_shapes = self.out_shapes, **self.func_space_kwargs)
        self.params = params
        self.outputs = outputs

    def __call__(self, *raw_args):
        args = [arg.input_var if hasattr(arg, 'input_var') else arg for arg in raw_args]
        print("Calling", args)
        shapes = [type.get_shape(add_batch=True) for type in self.lhs]
        outputs, params = self.func_space(*args, params = self.params, inp_shapes = self.inp_shapes, out_shapes = self.out_shapes, **self.func_space_kwargs)
        return outputs

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

def get_layer_params(layers):
    params = OrderedDict()
    for layer in layers:
        params.update(layer.params)
    return params

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

def compile_fns(interfaces, forallvars, axioms):
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

def train(train_fn, generators, num_epochs = 1000):
    print("Starting training...")
    for epoch in range(num_epochs):
        train_err = 0
        train_batches = 0
        start_time = time.time()
        for i in range(100):
            inputs = [gen.next() for gen in generators]
            losses = train_fn(*inputs)
            print("epoch: ", epoch, "losses: ", losses)
            train_err += losses[0]
            train_batches += 1
        print("epoch: ", epoch, " Total loss per epoch: ", train_err)

## A stack
def stack_example(train_data, stack_shape = (100,), item_shape = (28*28,), batch_sizes = (256, 256)):
    Stack = Type(stack_shape)
    Item = Type(item_shape)
    push = Interface([Stack, Item],[Stack], res_net, layer_width=100)
    pop = Interface([Stack],[Stack, Item], res_net, layer_width=884)
    stack1 = ForAllVar(Stack)
    item1 = ForAllVar(Item)
    global axiom1
    generators = [infinite_samples(np.random.rand, batch_sizes[0], stack_shape),
                  infinite_minibatches(train_data, batch_sizes[1], True)]
    axiom1 = Axiom(pop(*push(stack1.input_var, item1.input_var)), (stack1.input_var, item1.input_var))
    train_fn, call_fns = compile_fns([push, pop], [stack1, item1], [axiom1])
    train(train_fn, generators)

def scalar_field_example(field_shape = (100,), batch_size=256):
    Field = Type(field_shape)
    Point = Type((3,))
    Scalar = Type((1,))

    s = Interface([Field, Point], [Scalar], res_net, layer_width = 100)
    union = Interface([Field, Field], [Field], res_net, layer_width = 100)
    intersection = Interface([Field, Field], [Field], res_net, layer_width = 100)
    interfaces = [s, union, intersection]

    field1 = ForAllVar(Field)
    field2 = ForAllVar(Field)
    point1 = ForAllVar(Point)
    generators = [infinite_samples(np.random.rand, batch_size, field_shape),
                  infinite_samples(np.random.rand, batch_size, field_shape),
                  infinite_samples(np.random.rand, batch_size, (3,))]
    global forallvars
    forallvars = [field1, field2, point1]
    # Boolean structure on scalar field
    # ForAll p in R3, (f1 union f2)(p) = f1(p) f2(p)
    axiom1 = Axiom(s(*(union(field1, field2) + [point1])), [s(field1, point1)[0] * s(field2, point1)[0]])
    axiom2 = Axiom(s(*(intersection(field1, field2) + [point1])), [s(field1, point1)[0] + s(field2, point1)[0]])
    axioms = [axiom1, axiom2]
    train_fn, call_fns = compile_fns(interfaces, forallvars, axioms)
    train(train_fn, generators)

def binary_tree(train_data, binary_tree_shape = (500,), item_shape = (28*28,),  batch_size = 256):
    BinTree = Type(binary_tree_shape)
    Item = Type(item_shape)
    make = Interface([BinTree, Item, BinTree],[BinTree], res_net, layer_width=500)
    left_tree = Interface([BinTree], [BinTree], res_net, layer_width=500)
    right_tree = Interface([BinTree], [BinTree], res_net, layer_width=500)
    get_item = Interface([BinTree], [Item], res_net, layer_width=500)
    # is_empty = Interface([BinTree], [BoolType])

    bintree1 = ForAllVar(BinTree)
    bintree2 = ForAllVar(BinTree)
    item1 = ForAllVar(Item)
    # error = Constant(np.random.rand(item_shape))

    # axiom1 = Axiom(left_tree(create), error)
    make_stuff = make(bintree1.input_var, item1.input_var, bintree2.input_var)
    axiom2 = Axiom(left_tree(*make_stuff), (bintree1.input_var,))
    # axiom3 = Axiom(right_tree(create), error)
    axiom4 = Axiom(right_tree(*make_stuff), (bintree2.input_var,))
    # axiom5 = Axiom(item(create), error) # FIXME, how to handle True
    axiom6 = Axiom(get_item(*make_stuff), (item1.input_var,))
    # axiom7 = Axiom(is_empty(create), True)
    # axiom8 = Axiom(is_empty(make(bintree1.input_var, item1, bintree2)), False)
    interfaces = [make, left_tree, right_tree, get_item]
    # axioms = [axiom1, axiom2, axiom3, axiom4, axiom5, axiom6, axiom6, axiom7. axiom8]
    axioms = [axiom2, axiom4, axiom6]
    forallvars = [bintree1, bintree2, item1]
    generators = [infinite_samples(np.random.rand, batch_size, binary_tree_shape),
                 infinite_samples(np.random.rand, batch_size, binary_tree_shape),
                 infinite_minibatches(train_data, batch_size, True)]
    train_fn, call_fns = compile_fns(interfaces, forallvars, axioms)
    train(train_fn, generators)
#
# def associative_array(keyed_table_shape = (100,)):
#     # Types
#     KeyedTable = Type(keyed_table_shape)
#     Key = Type(key_shape)
#     Value = Type(val_shape)
#     # Interface
#     update = Interface([KeyedTable, Key, Value], [KeyedTable])
#     delete = Interface([KeyedTable Key], [KeyedTable])
#     find = Interface([KeyedTable, Key], [Value])
#     # is_in = Interface([KeyedTable Key], [BoolType])
#     # is_empty = Interface([KeyedTable], [BoolType])
#     # interface = [store, delete, find, is_in, is_empty]
#     interface = [update, delete, find]
#
#     # Variables
#     item1 = ForAllVar(Value)
#     key1 = ForAllVar(Key)
#     key2 = ForAllVar(Key)
#     kt1 = ForAllVar(KeyedTable)
#
#     axiom1 = find(update(kt1, )))
#
#
#     Item = Type(item_shape)
#     make = Interface([BinTree, Item, BinTree],[BinTree], res_net, layer_width=500)
#     left_tree = Interface([BinTree], [BinTree], res_net, layer_width=500)
#     right_tree = Interface([BinTree], [BinTree], res_net, layer_width=500)
#     get_item = Interface([BinTree], [Item], res_net, layer_width=500)
# #
# # def hierarhical_concept():
# #     ...
#
# def hierarhical_concept():
#     a = 3
#
# def turing_machine(state_shape=(10,), Symbol(1,)):
#     State = Type(state_shape)
#     Symbol = Type(symbol)
#
#     Q_s = Constant(0)



def main(argv):
    ## Args
    global options
    global test_files, train_files
    global net, output_layer, cost_f, cost_f_dict, val_f, call_f, call_f_dict
    global views, shape_params, outputs, net

    options = handle_args(argv)
    nepochs = options['nepochs'] = 10000
    options['compile_fns'] = False

    print(options)

    X_train, y_train, X_val, y_val, X_test, y_test = load_dataset()
    # stack_example(X_train.reshape(50000,28*28))
    # binary_tree(X_train.reshape(50000,28*28))
    scalar_field_example()


if __name__ == "__main__":
   main(sys.argv[1:])

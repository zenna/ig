import numpy as np

from common import *
import theano
import lasagne

from lasagne.layers import InputLayer
from lasagne.layers import ReshapeLayer
from lasagne.layers import DenseLayer
from lasagne.layers import NonlinearityLayer
from lasagne.layers import ConcatLayer
from lasagne.layers import DropoutLayer
from lasagne.layers import Pool2DLayer as PoolLayer
from lasagne.layers import Conv1DLayer
from lasagne.layers import batch_norm
from lasagne.init import HeNormal, Constant
from lasagne.nonlinearities import softmax, rectify, sigmoid

import theano.sandbox.cuda.dnn
if theano.sandbox.cuda.dnn.dnn_available():
    from lasagne.layers.dnn import Conv2DDNNLayer as ConvLayer
    from lasagne.layers.dnn import Conv3DDNNLayer as Conv3DLayer
else:
    from lasagne.layers import Conv2DLayer as ConvLayer


def res_net(*inputs, **kwargs):
  """A residual network of n inputs and m outputs"""
  inp_shapes = kwargs['inp_shapes']
  out_shapes = kwargs['out_shapes']
  params = kwargs['params']
  layer_width = kwargs['layer_width']
  nblocks = kwargs['nblocks']
  block_size = kwargs['block_size']
  output_args = kwargs['output_args']
  ninputs = len(inp_shapes)
  noutputs = len(out_shapes)

  input_width = np.sum([in_shape[1] for in_shape in inp_shapes])
  flat_output_shapes = [np.prod(out_shape[1:]) for out_shape in out_shapes]
  output_width = np.sum(flat_output_shapes)
  print("Building resnet with: %s residual blocks of size %s inner width: %s from: %s inputs to %s outputs" %
        (nblocks, block_size, layer_width, input_width, output_width))
  input_layers = [InputLayer(inp_shapes[i], input_var = inputs[i]) for i in range(len(inputs))]

  ## Flatten the input
  reshaped = [ReshapeLayer(inp, ([0], -1)) for inp in input_layers]

  net = {}
  net['concat'] = prev_layer = ConcatLayer(reshaped)
  # Projet inner layer down/up to hidden layer width only if necessary
  if layer_width != input_width:
      print("Input projection, layer_width: %s input_width: %s" % (layer_width, input_width))
      wx_sfx = 'wxinpproj'
      wx = batch_norm_params(DenseLayer(prev_layer, layer_width, nonlinearity = rectify,
        W=params['W_%s' % wx_sfx, HeNormal(gain='relu')],
        b=params['b_%s' % wx_sfx, Constant(0)]), wx_sfx, params)
  else:
      print("Skipping input weight projection, layer_width: %s input_width: %s" % (layer_width, input_width))
      wx = prev_layer

  ## Residual Blocks
  for j in range(nblocks):
    for i in range(block_size):
      sfx = "%s_%s" % (j,i)
      net['res2d%s_%s' % (j,i)] = prev_layer = batch_norm_params(
        DenseLayer(prev_layer, layer_width, nonlinearity = rectify,
        W=params['W_%s' % sfx, HeNormal(gain='relu')],
        b=params['b_%s' % sfx, Constant(0)]), sfx, params)
    net['block%s' % j] = prev_layer = wx = lasagne.layers.ElemwiseSumLayer([prev_layer, wx])

  ## Project output to correct width
  if layer_width != output_width:
      print("Output projection, layer_width: %s output_width: %s" % (layer_width, output_width))
      wx_sfx = 'wxoutproj'
      net['output'] = wx = batch_norm_params(DenseLayer(prev_layer, output_width, nonlinearity = rectify,
        W=params['W_%s' % wx_sfx, HeNormal(gain='relu')],
        b=params['b_%s' % wx_sfx, Constant(0)]), wx_sfx, params)
  else:
      print("Skipping output projection, layer_width: %s output_width: %s" % (layer_width, output_width))
      net['output'] = prev_layer

  # Split up the final layer into necessary parts and reshape
  output_product = lasagne.layers.get_output(net['output'], **output_args)
  outputs = []
  lb = 0
  for i in range(noutputs):
      ub = lb + flat_output_shapes[i]
      out = output_product[:, lb:ub]
      rout = out.reshape((out.shape[0],) + (out_shapes[i][1:]))
      outputs.append(rout)
      lb = ub

  params.add_tagged_params(get_layer_params(lasagne.layers.get_all_layers(net['output'])))
  params.check(lasagne.layers.get_all_params(prev_layer))
  return outputs, params

## ADT Templates
import numpy as np

import theano
import lasagne
from collections import OrderedDict

from lasagne.layers import InputLayer
from lasagne.layers import ReshapeLayer
from lasagne.layers import DenseLayer
from lasagne.layers import NonlinearityLayer
from lasagne.layers import ConcatLayer
from lasagne.layers import DropoutLayer
from lasagne.layers import Pool2DLayer as PoolLayer
from lasagne.layers import Conv1DLayer
from lasagne.layers import batch_norm

if theano.sandbox.cuda.dnn.dnn_available():
    from lasagne.layers.dnn import Conv2DDNNLayer as ConvLayer
    from lasagne.layers.dnn import Conv3DDNNLayer as Conv3DLayer
else:
    from lasagne.layers import Conv2DLayer as ConvLayer

from lasagne.init import HeNormal, Constant

from lasagne.nonlinearities import softmax, rectify, sigmoid

def get_layer_params(layers):
    params = OrderedDict()
    for layer in layers:
        params.update(layer.params)
    return params

def handle_batch_norm(params, suffix, bn_layer):
  params.set("W_%s" % suffix, bn_layer.input_layer.input_layer.W)
  params.set("b_%s" % suffix, bn_layer.input_layer.input_layer.b)
  params.set("beta_%s" % suffix, bn_layer.input_layer.beta)
  params.set("gamma_%s" % suffix, bn_layer.input_layer.gamma)
  params.set("inv_std_%s" % suffix, bn_layer.input_layer.inv_std)
  params.set("mean_%s" % suffix, bn_layer.input_layer.mean)

def batch_norm_params(input, sfx, params):
    l = batch_norm(input,
        beta=params['beta_%s' % sfx, Constant(0)], gamma=params['gamma_%s' % sfx, Constant(1)],
        mean=params['mean_%s' % sfx, Constant(0)], inv_std=params['inv_std_%s' % sfx, Constant(1)])
    handle_batch_norm(params, sfx, l)
    return l

# def batch_norm_params(input, sfx, params):
#     params.set("W_%s" % sfx, input.W)
#     params.set("b_%s" % sfx, input.b)
#     return input

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
      print("lbub", lb," ", ub)
      out = output_product[:, lb:ub]
      rout = out.reshape((out.shape[0],) + (out_shapes[i][1:]))
      outputs.append(rout)
      lb = ub

  params.add_tagged_params(get_layer_params(lasagne.layers.get_all_layers(net['output'])))
  params.check(lasagne.layers.get_all_params(prev_layer))
  return outputs, params

def conv_res_net(*inputs, **kwargs):
  """A residual convolutional network of n inputs and m outputs.

     One way to do this is to have the concatenate the layers into different channels"""
  inp_shapes = kwargs['inp_shapes']
  out_shapes = kwargs['out_shapes']
  params = kwargs['params']
  width, height = kwargs['width'], kwargs['height']
  deterministic = kwargs['deterministic']
  nblocks = kwargs['nblocks']
  block_size = kwargs['block_size']
  nfilters = kwargs['nfilters']
  npixels = width * height
  ninputs = len(inp_shapes)
  noutputs = len(out_shapes)
  output_args = kwargs['output_args']

  input_width = np.sum([in_shape[1] for in_shape in inp_shapes])
  output_width = np.sum([out_shape[1] for out_shape in out_shapes])

  print("Building convnet with: %s residual blocks of size %s" %
      (nblocks, block_size))

  # Each input is projected to a channel of the input image.
  # May need to project
  input_channels = []
  for i in range(len(inputs)):
      inp_ndim = len(inp_shapes[i])
      if inp_ndim == 4 and inp_shapes[i][2] == width and inp_shapes[i][3] == height:
          print("input ", i, " does not need reshaping or projecting")
          i = InputLayer(inp_shapes[i], input_var = inputs[i])
          input_channels.append(i)
      else:
          raise ValueError("Inputs must all be the right, same, size")
        #   print("input ", i, "of shape", inp_shapes[i][1:], " needs projecting to ", layer_shape)
        #   i = Input_layer(inp_shapes[i], input_var = inputs[i])
        #   r = ReshapeLayer(i, ([0], 1, 1, inp_size))
        #   c = ConvLayer(i, num_filters=npixels, filter_size=1, nonlinearity = rectify, W=lasagne.init.HeNormal(gain='relu'), pad='same' ))
        #   d = DenseLayer(i, npixels)
        #   r = ReshapeLayer(d, (1, width, height))
        #   input_channels.append(r)

  net = {}
  # input_Shape = (batch_size, ninputs, width, height)
  net['input_img'] = prev_layer = ConcatLayer(input_channels, axis = 1) # concatenate over channels
  #
  # ## Convolutional Layers
  wx_sfx = 'wx'

  wx = batch_norm_params(ConvLayer(prev_layer,
    num_filters=nfilters, filter_size=1, nonlinearity = rectify,
    W=params['W_%s' % wx_sfx, HeNormal(gain='relu')],
    b=params['b_%s' % wx_sfx, Constant(0)],
    pad='same'), wx_sfx, params)
  # net['resizeblock'] = prev_layer = x = lasagne.layers.ElemwiseSumLayer([wx, prev_layer])

  # 2d convolutional blocks
  for j in range(nblocks):
      for i in range(block_size):
          sfx = "%s_%s" % (j,i)
          net['conv2d%s_%s' % (j,i)] = prev_layer = batch_norm_params(ConvLayer(prev_layer,
            num_filters=nfilters, filter_size=3, nonlinearity = rectify,
            W=params['W_%s' % sfx, HeNormal(gain='relu')],
            b=params['b_%s' % sfx, Constant(0)],
            pad='same'), sfx, params)
      if nblocks > 1:
          net['block2d%s' % j] = wx = prev_layer = lasagne.layers.ElemwiseSumLayer([prev_layer, wx])

  sfx = 'final_conv'
  net['final_conv'] = prev_layer = batch_norm_params(ConvLayer(prev_layer,
      num_filters=noutputs, filter_size=3, nonlinearity = rectify,
      W=params['W_%s' % sfx, HeNormal(gain='relu')],
      b=params['b_%s' % sfx, Constant(0)],
      pad='same'), sfx, params)

  ## Output Projection
  net['output'] = prev_layer
  output_product = lasagne.layers.get_output(net['output'], **output_args)
  outputs = []
  for i in range(noutputs):
    outputs.append(output_product[:, i:i+1])

  all_params = get_layer_params(lasagne.layers.get_all_layers(net['output']))
  print("ALL_PARAMS", all_params)
  print("PARAMS", params)
  params.add_tagged_params(all_params)
  params.check(lasagne.layers.get_all_params(prev_layer))
  return outputs, params
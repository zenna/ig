import theano
import lasagne
import theano.tensor as T
import numpy as np
from theano import function
from numpy.linalg import inv
from theano.tensor.nlinalg import matrix_inverse

# from lasagne.layers import batch_norm
def batch_norm(x): return x

def inv_leaky_rectify(y):
  return T.minimum(y, 10*y)

def np_inv_leaky_rectify(y):
  return np.minimum(y, 10*y)

def leaky_rectify(x):
  return T.maximum(0.1*x, x)

def np_leaky_rectify(x):
  return np.maximum(x, 0.1*x)


ninputs = 3
nbatch = 200
nlayers = 2
layers = [lasagne.layers.InputLayer((None, ninputs))]
input = layers[0]

def bound_loss(x, tnp = np) :
  eps = 1e-9
  loss = tnp.maximum(tnp.maximum(eps,x-1), tnp.maximum(eps,-x)) + eps
  return tnp.maximum(loss, eps) + eps

layer = layers[0]
for i in range(nlayers):
  layer = batch_norm(lasagne.layers.DenseLayer(layer, ninputs, nonlinearity=leaky_rectify))
  layers.append(layer)

op = lasagne.layers.get_output(layers[-1])
ops = [lasagne.layers.get_output(layer) for layer in layers]

def fwd_kinematics(inp, np = T):
  phi1 = inp[:,0]
  phi2 = inp[:,1]
  phi3 = inp[:,2]
  x = np.cos(phi1) + np.cos(phi1+phi2) + np.cos(phi1+phi2+phi3)
  y = np.sin(phi1) + np.sin(phi1+phi2) + np.sin(phi1+phi2+phi3)
  return np.stack([x,y],1)

fwd_kin_out = fwd_kinematics(input.input_var)
dist = ((op[:,0:2] - fwd_kin_out[:,0:2])**2) + 1e-9
# dist = op.sum()
loss = dist.mean()

params = lasagne.layers.get_all_params(layers[-1], trainable=True)
updates = lasagne.updates.adam(loss, params, 1e-4)

f_train = function([input.input_var], [op, fwd_kin_out, loss], updates = updates)
f = function([input.input_var], [op, fwd_kin_out, loss])

for i in range(1000):
  data = np.random.rand(nbatch, ninputs)
  op1, op2, loss_d = f_train(data)
  print "loss:", loss_d

def invert_weight_matrix(w):
  invw = []
  for i in range(len(w)):
    # layer_weight = w[-(i+1)]
    if i%2 == 1:
      layer_weight = w[-(i+1)]
      print "inv val", -(i+1+1), "of length", len(w)
      invw.append(inv(layer_weight))
    else:
      layer_weight = w[-(i+1)]
      print "bias inv val", -(i+1-1), "of length", len(w)
      invw.append(-layer_weight)

  return invw

def invert_weight_matrix_symb(w):
  invw = []
  for i in range(len(w)):
    # layer_weight = w[-(i+1)]
    if i%2 == 1:
      layer_weight = w[-(i+1)]
      print "inv val", -(i+1+1), "of length", len(w)
      invw.append(matrix_inverse(layer_weight))
    else:
      layer_weight = w[-(i+1)]
      print "bias inv val", -(i+1-1), "of length", len(w)
      invw.append(-layer_weight)

  return invw




## make inverse network
# inv_layers = [lasagne.layers.InputLayer((None, ninputs))]
# inv_layers = [lasagne.layers.InputLayer((None, ninputs - 2))]

def ok(input_layer, symb_inv_weights, trainable):
  inv_layers = [input_layer]

  inv_layers.append(lasagne.layers.NonlinearityLayer(inv_layers[0], nonlinearity=inv_leaky_rectify))

  invlayer = inv_layers[1]
  j = 0
  for i in range(nlayers - 1):
    invlayer = batch_norm(lasagne.layers.BiasLayer(invlayer, b = symb_inv_weights[j]))
    inv_layers.append(invlayer)
    invlayer = batch_norm(lasagne.layers.DenseLayer(invlayer, ninputs, nonlinearity=inv_leaky_rectify, b = None, W = symb_inv_weights[j+1]))
    inv_layers.append(invlayer)
    j = j + 2

  invlayer = batch_norm(lasagne.layers.BiasLayer(invlayer, b = symb_inv_weights[j]))
  j = j + 1
  inv_layers.append(invlayer)
  invlayer = batch_norm(lasagne.layers.DenseLayer(invlayer, ninputs, nonlinearity=None, b = None, W = symb_inv_weights[j]))
  inv_layers.append(invlayer)
  return inv_layers

symb_inv_weights = invert_weight_matrix_symb(params)
# inv_layers_train = ok(layers[-1], symb_inv_weights, True)
# inv_op = lasagne.layers.get_output(inv_layers_train[-1])
# inv_ops = [lasagne.layers.get_output(layer) for layer in inv_layers_train]

# # param_weights = lasagne.layers.get_all_param_values(layers[-1])
# # inv_weights = invert_weight_matrix(param_weights)
# # lasagne.layers.set_all_param_values(inv_layers[-1], inv_weights)

# # Inversion should be within the training set bounds
# bl1 = bound_loss(inv_op, tnp = T).mean()

# # Parameter should be within some reasonable bounds.
# bl2 = bound_loss(op[:,2], tnp = T).mean()

# reverse_loss = bl1 + bl2 + loss
# inv_params = lasagne.layers.get_all_params(inv_layers_train[-1], trainable=True)
# updates = lasagne.updates.adam(reverse_loss, inv_params, 1e-4)
# f2_train = function([input.input_var], [op, inv_op, loss, reverse_loss], updates = updates)

# inv_input = inv_layers[0].input_var
# f_inv = function([inv_input], inv_op)

# for i in range(10000):
#   data = np.random.rand(nbatch, ninputs)
#   d_op, d_inv_op, d_loss, reverse_loss = f2_train(data)
#   print "losses:", d_loss, reverse_loss

# inverse function
inv_inp = lasagne.layers.InputLayer((None, ninputs)) 
inv_layers_call = ok(inv_inp, symb_inv_weights, False)
inv_op2 = lasagne.layers.get_output(inv_layers_call[-1])
f2 = function([inv_inp.input_var], inv_op2)

## Sampling Parameter Space
output_inp = lasagne.layers.InputLayer((None, 2), input_var = op[:,0:2])
param_inp = lasagne.layers.InputLayer((None, ninputs - 2))
concat = lasagne.layers.ConcatLayer([output_inp, param_inp])

concat_inv_layers_train = ok(concat, symb_inv_weights, True)
concat_inv_op = lasagne.layers.get_output(concat_inv_layers_train[-1])
# Inversion should be within the training set bounds
bl1 = bound_loss(concat_inv_op, tnp = T).mean()

# Parameter should be within some reasonable bounds.
bl2 = bound_loss(op[:,2], tnp = T).mean()

reverse_loss_concat = bl1 + bl2 + loss
inv_params_concat = lasagne.layers.get_all_params(concat_inv_layers_train[-1], trainable=True)
updates = lasagne.updates.adam(reverse_loss_concat, inv_params_concat, 1e-4)
f2_train_concat = function([input.input_var, param_inp.input_var], [op, concat_inv_op, loss, reverse_loss_concat], updates = updates)

for i in range(10000):
  data = np.random.rand(nbatch, ninputs)
  param = np.random.rand(nbatch, ninputs - 2)
  d_op, d_concat_inv_op, d_loss, d_reverse_loss_concat = f2_train_concat(data, param)
  print "losses:", d_loss, d_reverse_loss_concat

# Testing
## ======
data = np.random.rand(1, ninputs)
output = f(data)[0]
# f_inv(f(data)[0]) - data

def fwd_kinematics2(phi1, phi2, phi3):
  x = np.cos(phi1) + np.cos(phi1+phi2) + np.cos(phi1+phi2+phi3)
  y = np.sin(phi1) + np.sin(phi1+phi2) + np.sin(phi1+phi2+phi3)
  return x,y

import theano
import theano.tensor as T
from theano import shared
from theano import function
from theano import config
import lasagne
import numpy as np
# config.optimizer='None'
from theano.compile.nanguardmode import NanGuardMode


eps = 1e-9

def inspect_inputs(i, node, fn):
    print i, node, "input(s) value(s):", [input[0] for input in fn.inputs]

def inspect_outputs(i, node, fn):
    print " output(s) value(s):", [output[0] for output in fn.outputs]

mode = None
# mode = NanGuardMode(nan_is_error=True, inf_is_error=True, big_is_error=True)
# mode = theano.compile.MonitorMode(pre_func=inspect_inputs,post_func=inspect_outputs)

# from theano.printing import Print
# Null Print
def Print(x):
  return lambda x: x

nbatch = 1
l1 = l2 = l3 = 1.0
phi1 = T.vector('phi1')
phi2 = T.vector('phi2')
phi3 = T.vector('phi3')

x = T.vector('x')
y = T.vector('y')

def invplus(x, params, tnp = T):
  c = x - params.sum(axis=1) + eps
  return tnp.concatenate([params, tnp.reshape(c, (c.shape[0],1))], axis = 1)

# def singular(vals):
#   s = T.stack(vals)
#   eps = 1e-9
#   return T.mean(s, axis=0), T.maximum(T.var(s, axis=0), eps) # Should eps be here?!

# Mean square error
def mse(a, b, name = ''):
    a = Print('a--%s' % name)(a)
    b = Print('b--%s' % name)(b)
    ab = Print('a-b--%s' % name)(a-b + eps)
    absab = Print('abs_(ab)--%s' % name)(T.sqr(ab))
    return (absab + eps).sum() + eps

def singular(vals, name = ''):
  # s = T.stack(vals)
  # eps = 1e-9
  # mean = T.mean(s, axis=0)
  mean = Print('vals[0]--%s' % name)(vals[0]) + eps
  dists = [Print('mse(mean, val)--%s' % name)(mse(mean, val, name = name)) for val in vals]
  variance = Print('sum(dists)--%s' % name)(sum(dists) + eps)
  return vals[0], variance # T.maximum(T.var(s, axis=0), eps) # Should eps be here?!
  # return T.mean(s, axis=0), variance # T.maximum(T.var(s, axis=0), eps) # Should eps be here?!

def bound_loss(x, tnp = np) :
  eps = 1e-9
  loss = tnp.maximum(tnp.maximum(eps,x-1), tnp.maximum(eps,-x)) + eps
  return tnp.maximum(loss, eps) + eps

params_plus1 = shared(np.random.rand(nbatch, 2))
params_plus2 = shared(np.random.rand(nbatch, 2))

a = Print('invplus(x, params_plus1')(invplus(x, params_plus1))
b = Print('invplus(x, params_plus2')(invplus(y, params_plus2))

eps = 1e-9
a2 = T.arccos(T.clip(a, eps, 1-eps))
b2 = T.arcsin(T.clip(b, eps, 1-eps))
bl1 = bound_loss(a, tnp = T)
bl2 = bound_loss(b, tnp = T)

phi1_group = []
phi2_group = []
phi3_group = []
phi1_group.append(a2[:, 0])
phi1_group.append(b2[:, 0])

delta_group = []
delta_group.append(a2[:, 1])
delta_group.append(b2[:, 1])
delta_single, delta_var = singular(delta_group, name = 'delta')

params_plus3 = shared(np.random.rand(nbatch, 1))
q = invplus(delta_single, params_plus3)
phi1_group.append(q[:, 0])
phi2_group.append(q[:, 1])

gamma_group = []
gamma_group.append(a2[:, 2])
gamma_group.append(b2[:, 2])
gamma_single, gamma_var = singular(gamma_group, name = 'gamma')

params_plus4 = shared(np.random.rand(nbatch, 2))
q = invplus(gamma_single, params_plus4)
phi1_group.append(q[:, 0])
phi2_group.append(q[:, 1])
phi3_group.append(q[:, 2])

phi1_mean, phi1_var = singular(phi1_group, name = 'phi1')
phi2_mean, phi2_var = singular(phi2_group, name = 'phi2')
phi3_mean, phi3_var = singular(phi3_group, name = 'phi3')

# losses = [phi1_var]
losses = [delta_var.mean(), gamma_var.mean(), phi1_var.mean(), phi2_var.mean(), phi3_var.mean(), bl1.mean(), bl2.mean()]


# 0.246740111827
total_loss = T.mean(losses)
# total_loss = phi1_var

params = [params_plus1, params_plus2, params_plus3, params_plus4]
updates = lasagne.updates.adam(total_loss, params)
momentum = lasagne.updates.adam(total_loss, params, 0.1)


  # print "Printing Grad"
  # theano.printing.pydotprint(T.grad(total_loss, params_plus1], outfile="graph1.png", var_with_name_simple=True)

print "Building function"
f = function([x, y], [total_loss, phi1_mean, phi2_mean, phi3_mean] + [gamma_var], updates = updates, mode = mode)
print "Printing"
# theano.printing.pydotprint(f, outfile="graph.png", var_with_name_simple=True)  
# output = f([0.5,1.5],[1.0,1.4])
# phi1_mean_d, phi2_mean_d, phi3_mean_d, delta_var_d, gamma_var_d, phi1_var_d, phi2_var_d, phi3_var_d, bl1_d, bl2_d, tl, a2_d, b2_d = output

def fwd_kinematics(phi1, phi2, phi3):
  x = np.cos(phi1) + np.cos(phi1+phi2) + np.cos(phi1+phi2+phi3)
  y = np.sin(phi1) + np.sin(phi1+phi2) + np.sin(phi1+phi2+phi3)
  return x,y


for i in range(10000):
  # output = f([0.5],[1.0])
  output = f([2.7289220051879965], [1.2076257779006494])
  # phi1_mean_d, phi2_mean_d, phi3_mean_d, delta_var_d, gamma_var_d, phi1_var_d, phi2_var_d, phi3_var_d, bl1_d, bl2_d, tl, a2_d, b2_d = output
  tl, phi1_mean_d, phi2_mean_d, phi3_mean_d = output[0:4]
  print "angles ", phi1_mean_d, phi2_mean_d, phi3_mean_d
  # gamma_var_d = output[4]
  # print "gamma", gamma_var_d
  print "loss", tl
  print "validation loss", fwd_kinematics(phi1_mean_d, phi2_mean_d, phi3_mean_d)
  # print output
  print "params"
  for i in params:
    print i.get_value()

# fwd_kinematics(0.3, 0.1, 0.15)

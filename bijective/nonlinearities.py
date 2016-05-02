import theano.tensor as T
import numpy as np

def inv_leaky_rectify(y, alpha = 0.8, tnp = T):
  return tnp.minimum(y, 1/alpha*y)

def leaky_rectify(x, alpha = 0.8, tnp = T):
  return tnp.maximum(alpha*x, x)

def s_rectify(y, tnp = T):
  return tnp.minimum(tnp.maximum(0, y),1)

def logit(y, tnp = T):
    return -tnp.log(1/y-1)

def contract(x, tnp = T):
    return 0.5 * tnp.tanh(x)+0.5

def expand(y, tnp = T):
    return tnp.arctanh(2*y - 1)

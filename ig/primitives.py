## Primitive Parametric Inverses
import theano.tensor as T

def inv_mul(z, params, add_eps = True, tnp = T):
    assert params.ndim == z.ndim, "%s, %s" % (params.ndim, z.ndim)
    eps  = 1e-9
    if add_eps:
        params = params + eps
    return params, (z/params)

def inv_plus(z, params, tnp = T):
  c = z - params.sum(axis=1) + eps
  return tnp.concatenate([params, tnp.reshape(c, (c.shape[0],1))], axis = 1)

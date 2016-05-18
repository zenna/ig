import numpy as np
import lasagne


def stringy(ls):
    out = ""
    for l in ls:
        out = out + str(l) + "_"
    return out


def stringy_dict(d):
    out = ""
    for (key, val) in d.items():
        if val is not None and val is not '':
            out = out + "%s:%s_" % (str(key), str(val))
    return out


def rand_rotation_matrix(deflection=1.0, randnums=None, floatX='float32'):
    """
    Creates a random rotation matrix.

    deflection: the magnitude of the rotation. For 0, no rotation; for 1,
    competely random
    rotation. Small deflection => small perturbation.
    randnums: 3 random numbers in the range [0, 1]. If `None`, they will be
    auto-generated.
    """
    # from realtimerendering.com/resources/GraphicsGems/gemsiii/rand_rotation.c

    if randnums is None:
        randnums = np.random.uniform(size=(3,))

    theta, phi, z = randnums

    theta = theta * 2.0*deflection*np.pi  # Rotation about the pole (Z).
    phi = phi * 2.0*np.pi  # For direction of pole deflection.
    z = z * 2.0*deflection  # For magnitude of pole deflection.

    # Compute a vector V used for distributing points over the sphere
    # via the reflection I - V Transpose(V).  This formulation of V
    # will guarantee that if x[1] and x[2] are uniformly distributed,
    # the reflected points will be uniform on the sphere.  Note that V
    # has length sqrt(2) to eliminate the 2 in the Householder matrix.

    r = np.sqrt(z)
    Vx, Vy, Vz = V = (
        np.sin(phi) * r,
        np.cos(phi) * r,
        np.sqrt(2.0 - z)
        )

    st = np.sin(theta)
    ct = np.cos(theta)

    R = np.array(((ct, st, 0), (-st, ct, 0), (0, 0, 1)))

    # Construct the rotation matrix  ( V Transpose(V) - I ) R.

    M = (np.outer(V, V) - np.eye(3)).dot(R)
    return np.array(M, dtype=floatX)


# n random matrices
def rand_rotation_matrices(n, floatX='float32'):
    return np.stack([rand_rotation_matrix(floatX=floatX) for i in range(n)])


def named_outputs(func, names):
    def dict_func(*args):
        return dict(zip(names, func(*args)))

    return dict_func


def identity(x):
    return x


def circular_indices(lb, ub, thresh):
    indices = []
    while True:
        stop = min(ub, thresh)
        ix = np.arange(lb, stop)
        indices.append(ix)
        if stop != ub:
            diff = ub - stop
            lb = 0
            ub = diff
        else:
            break

    return np.concatenate(indices)


# Minibatching
def infinite_samples(sampler, batchsize, shape):
    while True:
        to_sample_shape = (batchsize,)+shape
        yield lasagne.utils.floatX(sampler(*to_sample_shape))


def infinite_batches(inputs, batchsize, f=identity, shuffle=False):
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
        yield f(inputs[excerpt])

#
# def infinite_batches(inputs, batchsize, f, shuffle=False):
#     start_idx = 0
#     nelements = len(inputs)
#     indices = np.arange(nelements)
#     if shuffle:
#         indices = np.arange(len(inputs))
#         np.random.shuffle(indices)
#     while True:
#         data = yield
#         end_idx = start_idx + batchsize
#         if end_idx > nelements:
#             diff = end_idx - nelements
#             excerpt = np.concatenate([indices[start_idx:nelements],
#                                       indices[0:diff]])
#             start_idx = diff
#             if shuffle:
#                 indices = np.arange(len(inputs))
#                 np.random.shuffle(indices)
#         else:
#             excerpt = indices[start_idx:start_idx + batchsize]
#             start_idx = start_idx + batchsize
#         yield f(inputs[excerpt], data)


def constant_batches(x, f):
    while True:
        data = yield
        yield f(x, data)


def iterate_batches(inputs, batchsize, shuffle=False):
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_iiteratedx + batchsize)
        yield inputs[excerpt]


def upper_div(x, y):
    a, b = divmod(x, y)
    if b == 0:
        return a
    else:
        return a + 1

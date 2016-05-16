import numpy as np

def stringy(ls):
    out = ""
    for l in ls:
        out = out + str(l) + "_"
    return out

def stringy_dict(d):
    out = ""
    for (key,val) in d.items():
        if val is not None and val is not '':
            out = out + "%s:%s_" % (str(key), str(val))
    return out

def rand_rotation_matrix(deflection=1.0, randnums=None, floatX = 'float32'):
    """
    Creates a random rotation matrix.

    deflection: the magnitude of the rotation. For 0, no rotation; for 1, competely random
    rotation. Small deflection => small perturbation.
    randnums: 3 random numbers in the range [0, 1]. If `None`, they will be auto-generated.
    """
    # from http://www.realtimerendering.com/resources/GraphicsGems/gemsiii/rand_rotation.c

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
def rand_rotation_matrices(n, floatX = 'float32'):
    return np.stack([rand_rotation_matrix(floatX = floatX) for i in range(n)])


def named_outputs(func, names):
    def dict_func(*args):
        return dict(zip(names, func(*args)))

    return dict_func

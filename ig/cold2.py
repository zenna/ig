## Volume Raycasting
from theano import function, config, shared, printing
import numpy as np
from scipy.sparse import csr_matrix
try:
    from mayavi import mlab
except:
    print "couldnt import"
from mayavi import mlab


## Extract features from an image
import lasagne
import theano
import theano.sandbox.cuda.dnn
from lasagne.layers import InputLayer, DenseLayer, DropoutLayer
if theano.sandbox.cuda.dnn.dnn_available():
    from lasagne.layers.dnn import Conv2DDNNLayer as ConvLayer
else:
    from lasagne.layers import Conv2DLayer as ConvLayer

from lasagne.layers import MaxPool2DLayer as PoolLayer
from lasagne.layers import LocalResponseNormalization2DLayer as NormLayer
from lasagne.utils import floatX
from theano import tensor as T
from theano import function, config, shared
import pickle

config.exception_verbosity='high'
config.optimizer = 'None'

def rand_rotation_matrix(deflection=1.0, randnums=None):
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
    return M


# Genereate values in raster space, x[i,j] = [i,j]
def gen_fragcoords(width, height):
    """Create a (width * height * 2) matrix, where element i,j is [i,j]
       This is used to generate ray directions based on an increment"""
    raster_space = np.zeros([width, height, 2], dtype=config.floatX)
    for i in range(width):
        for j in range(height):
            raster_space[i,j] = np.array([i,j], dtype=config.floatX) + 0.5
    return raster_space

# Append an image filled with scalars to the back of an image.
def stack(intensor, width, height, scalar):
    scalars = np.ones([width, height, 1], dtype=config.floatX) * scalar
    return np.concatenate([intensor, scalars], axis=2)

def symbolic_render(r, raster_space, width, height):
    """Symbolically render rays starting with raster_space according to geometry
      e  defined by """
    resolution = np.array([width, height], dtype=config.floatX)
    # Normalise it to be bound between 0 1
    norm_raster_space = raster_space / resolution
    # Put it in NDC space, -1, 1
    screen_space = -1.0 + 2.0 * norm_raster_space
    # Make pixels square by mul by aspect ratio
    ndc_space = screen_space * np.array([resolution[0]/resolution[1],1.0], dtype=config.floatX)
    # Ray Direction

    # Position on z-plane
    ndc_xyz = stack(ndc_space, width, height, 1.0)*0.5 # Change focal length

    # Put the origin farther along z-axis
    ro = np.array([0,0,1.5])

    # Rotate both by same rotation matrix
    ro_t = T.dot(ro, r)
    ndc_t = T.dot(ndc_xyz, r)

    # Increment by 0.5 since voxels are in [0, 1]
    ro_t = ro_t + 0.5
    ndc_t = ndc_t + 0.5
    # Find normalise ray dirs from origin to image plane
    unnorm_rd = ndc_t - ro_t
    rd = unnorm_rd / T.reshape(unnorm_rd.norm(2, axis=2), (width, height, 1))
    return rd, ro_t

def switch(cond, a, b):
    return cond*a + (1-cond)*b

def gen_img(shape_params, rotation_matrix, width, height, nsteps, res):
    raster_space = gen_fragcoords(width, height)
    rd, ro = symbolic_render(rotation_matrix, raster_space, width, height)
    a = 0 - ro # c = 0
    b = 1 - ro # c = 1
    tn = a/rd
    tf = b/rd
    tn_true = T.minimum(tn,tf)
    tf_true = T.maximum(tn,tf)
    # do X
    tn_x = tn_true[:,:,0]
    tf_x = tf_true[:,:,0]
    tmin = 0.0
    tmax = 10.0
    t0 = tmin
    t1 = tmax
    t02 = T.switch(tn_x > t0, tn_x, t0)
    t12 = T.switch(tf_x < t1, tf_x, t1)
    # y
    tn_x = tn_true[:,:,1]
    tf_x = tf_true[:,:,1]
    t03 = T.switch(tn_x > t02, tn_x, t02)
    t13 = T.switch(tf_x < t12, tf_x, t12)
    #z
    tn_x = tn_true[:,:,2]
    tf_x = tf_true[:,:,2]
    t04 = T.switch(tn_x > t03, tn_x, t03)
    t14 = T.switch(tf_x < t13, tf_x, t13)

    # Shit a little bit to avoid numerial inaccuracies
    t04 = t04*1.001
    t14 = t14*0.999

    # img = np.zeros(width * height)
    left_over = np.ones(width * height)
    step_size = (t14 - t04)/nsteps
    orig = ro + rd* T.reshape(t04,(width, height, 1))
    step_size = T.reshape(step_size, (width, height, 1))
    step_size_flat = step_size.flatten()
    xres = yres = zres = res
    nrays = width * height
    nvoxels = xres * yres * zres
    # A = csr_matrix((nrays, nvoxels))
    for i in range(nsteps):
        # print "step", i
        pos = orig + rd*step_size*i
        voxel_indices = T.floor(pos*res)
        pruned = T.clip(voxel_indices,0,res-1)
        p_int =  T.cast(pruned, 'int32')
        indices = T.reshape(p_int, (width*height,3))
        attenuation = shape_params[indices[:,0],indices[:,1],indices[:,2]]
        left_over = left_over*T.exp(-attenuation*step_size_flat)

    img = left_over
    pixels = T.reshape(img,(width, height))
    mask = t14>t04
    return T.switch(t14>t04, pixels, T.ones_like(pixels))

from matplotlib import pylab as plt

def load_voxels_binary(fname, width, height, depth, max_value=255.0):
    data = np.fromfile(fname, dtype='uint8')
    return np.reshape(data, (width, height, depth))/float(max_value)

def histo(x):
    # the histogram of the data
    n, bins, patches = plt.hist(x.flatten(), 500,range=(0.0001,1), normed=1, facecolor='green', alpha=0.75)
    plt.show()

plt.ion()

width = 100
height = 100
res = 256
nsteps = 100
## Example Data
# x, y, z = np.ogrid[-10:10:complex(res), -10:10:complex(res), -10:10:complex(res)]
# shape_params = np.sin(x*y*z)/(x*y*z)
# shape_params = np.clip(shape_params,0,1)
# shape_params = shape_params - np.min(shape_params) * (np.max(shape_params) - np.min(shape_params))


rotation_matrix = T.matrix()
shape_params = T.tensor3()
out = gen_img(shape_params, rotation_matrix, width, height, nsteps, res)
f = function([shape_params, rotation_matrix], out)

voxel_data = load_voxels_binary("person_0089.raw", res, res, res)*10.0
f(voxel_data)

# 1. TODO, update it to take in rotation tensor
#
one image in creates one voxel grid.  then we render that in 5 different waysself.

#
# def train():
#     views_per_img = 3
#     nobjects = 3
#     # Generate n images with n known different views of an m objects
#     nsteps = 100
#     rs = []
#     As = []
#     imgs = []
#     r = rand_rotation_matrix()
#     img, A, mask, img_lim = main(shape_params, r, width, height, nsteps, res)

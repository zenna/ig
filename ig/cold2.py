## Volume Raycasting
from theano import function, config, shared, printing
import numpy as np
# try:
#     from mayavi import mlab
# except:
#     print "couldnt import"
# from mayavi import mlab


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

# config.exception_verbosity='high'
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

# n random matrices
def random_rotation_matrices(n):
    return np.stack([rand_rotation_matrix() for i in range(n)])

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

def make_ro(r, raster_space, width, height):
    """Symbolically render rays starting with raster_space according to geometry
      e  defined by """
    nmatrices = r.shape[0]
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
    ro_t = T.dot(T.reshape(ro, (1,3)), r)
    ndc_t = T.dot(T.reshape(ndc_xyz, (1, width, height, 3)), r)
    ndc_t = T.reshape(ndc_t, (width, height, nmatrices, 3))
    ndc_t = T.transpose(ndc_t, (2,0,1,3))

    # Increment by 0.5 since voxels are in [0, 1]
    ro_t = ro_t + 0.5
    ndc_t = ndc_t + 0.5
    # Find normalise ray dirs from origin to image plane
    unnorm_rd = ndc_t - T.reshape(ro_t, (nmatrices,1,1,3))
    rd = unnorm_rd / T.reshape(unnorm_rd.norm(2, axis=3), (nmatrices, width, height, 1))
    return rd, ro_t

def switch(cond, a, b):
    return cond*a + (1-cond)*b

def gen_img(shape_params, rotation_matrix, width, height, nsteps, res):
    raster_space = gen_fragcoords(width, height)
    rd, ro = make_ro(rotation_matrix, raster_space, width, height)
    a = 0 - ro # c = 0
    b = 1 - ro # c = 1
    nmatrices = rotation_matrix.shape[0]
    tn = T.reshape(a, (nmatrices, 1, 1, 3))/rd
    tf = T.reshape(b, (nmatrices, 1, 1, 3))/rd
    tn_true = T.minimum(tn,tf)
    tf_true = T.maximum(tn,tf)
    # do X
    tn_x = tn_true[:,:,:,0]
    tf_x = tf_true[:,:,:,0]
    tmin = 0.0
    tmax = 10.0
    t0 = tmin
    t1 = tmax
    t02 = T.switch(tn_x > t0, tn_x, t0)
    t12 = T.switch(tf_x < t1, tf_x, t1)
    # y
    tn_x = tn_true[:,:,:,1]
    tf_x = tf_true[:,:,:,1]
    t03 = T.switch(tn_x > t02, tn_x, t02)
    t13 = T.switch(tf_x < t12, tf_x, t12)
    #z
    tn_x = tn_true[:,:,:,2]
    tf_x = tf_true[:,:,:,2]
    t04 = T.switch(tn_x > t03, tn_x, t03)
    t14 = T.switch(tf_x < t13, tf_x, t13)

    # Shift a little bit to avoid numerial inaccuracies
    t04 = t04*1.001
    t14 = t14*0.999

    # img = T.zeros(width * height)
    nvoxgrids = shape_params.shape[0]
    left_over = T.ones((nvoxgrids, nmatrices * width * height,))
    step_size = (t14 - t04)/nsteps
    orig = T.reshape(ro, (nmatrices, 1, 1, 3)) + rd * T.reshape(t04,(nmatrices, width, height, 1))
    # step_size = T.reshape(step_size, (width, height, 1))
    # step_size_flat = step_size.flatten()
    xres = yres = zres = res

    orig = T.reshape(orig, (nmatrices * width * height, 3))
    rd = T.reshape(rd, (nmatrices * width * height, 3))
    step_sz = T.reshape(step_size, (nmatrices * width * height,1))

    for i in range(nsteps):
        # print "step", i
        pos = orig + rd*step_sz*i
        voxel_indices = T.floor(pos*res)
        pruned = T.clip(voxel_indices,0,res-1)
        p_int =  T.cast(pruned, 'int32')
        indices = T.reshape(p_int, (nmatrices*width*height,3))
        attenuation = shape_params[:, indices[:,0],indices[:,1],indices[:,2]]
        left_over = left_over*T.exp(-attenuation*T.flatten(step_sz))

    img = left_over
    pixels = T.reshape(img, (nvoxgrids, nmatrices, width, height))
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


## Example Data
# x, y, z = np.ogrid[-10:10:complex(res), -10:10:complex(res), -10:10:complex(res)]
# shape_params = np.sin(x*y*z)/(x*y*z)
# shape_params = np.clip(shape_params,0,1)
# shape_params = shape_params - np.min(shape_params) * (np.max(shape_params) - np.min(shape_params))


#
def dist(a, b):
    eps = 1e-9
    return T.sum(T.maximum(eps, (a - b)**2))
#
# cost = dist(views, out)
# cost_f = function([shape_params, rotation_matrices, views], cost)
# result = cost_f(voxel_data, r, imgdata)
#
def second_order(rotation_matrices, imagebatch, width = 134, height = 134, nsteps = 100, res = 128, nvoxgrids = 4):
    """Creates a network which takes as input a image and returns a cost.
    Network extracts features of image to create shape params which are rendered.
    The similarity between the rendered image and the actual image is the cost
    """
    first_img = imagebatch[:,0,:,:]
    # nvoxgrids = imagebatch.shape[0]
    # height = imagebatch.shape[3]
    # width = imagebatch.shape[2]
    first_img = T.reshape(first_img, (nvoxgrids,1,width,height))
    layers_per_layer = 4


    # Put the different convnets into two channels
    net = {}
    net['input'] = InputLayer((nvoxgrids, 1, width, height), input_var = first_img)
    net['conv1'] = ConvLayer(net['input'], num_filters=layers_per_layer*128, filter_size=7, stride=1)
    net['norm1'] = NormLayer(net['conv1'], alpha=0.0001) # caffe has alpha = alpha * pool_size
    filters = lasagne.layers.get_output(net['norm1'])
    stacks = T.reshape(filters, (nvoxgrids, res, layers_per_layer, res, res))
    weights = shared(np.random.rand(1, res, layers_per_layer, res, res))
    accum = T.sum(stacks * weights, axis=2)
    # Relu
    voxels = lasagne.nonlinearities.rectify(accum)
    out = gen_img(voxels, rotation_matrices, width, height, nsteps, res)
    loss = dist(imagebatch, out)

    params = lasagne.layers.get_all_params(net['norm1'], trainable=True)
    params.append(weights)
    network_updates = lasagne.updates.nesterov_momentum(loss, params, learning_rate=0.01, momentum=0.9)
    return loss, network_updates

import os
def get_filepaths(directory):
    """
    This function will generate the file names in a directory
    tree by walking the tree either top-down or bottom-up. For each
    directory in the tree rooted at directory top (including top itself),
    it yields a 3-tuple (dirpath, dirnames, filenames).
    """
    file_paths = []  # List which will store all of the full filepaths.

    # Walk the tree.
    for root, directories, files in os.walk(directory):
        for filename in files:
            # Join the two strings in order to form the full filepath.
            filepath = os.path.join(root, filename)
            file_paths.append(filepath)  # Add it to the list.

    return file_paths  # Self-explanatory.

def get_rnd_voxels(n):
    files = filter(lambda x:x.endswith(".raw"), get_filepaths(os.getenv('HOME') + '/data/ModelNet40/chair/train'))
    return np.random.choice(files, n, replace=False)

def train(cost_f, render, nviews = 3, nvoxgrids=4, res = 128):
    print "Training"
    npochs = 1000
    for i in range(npochs):
        print "epoch: ", i
        filenames = get_rnd_voxels(nvoxgrids)
        voxel_data = [load_voxels_binary(v, res, res, res)*10.0 for v in filenames]
        r = random_rotation_matrices(nviews)
        print "Rendering Training Data"
        imgdata = render(voxel_data, r)
        cost = cost_f(imgdata, rotation_matrices)
        print "cost is ", cost

def main():
    width = 134
    height = 134
    res = 128
    nsteps = 100
    nvoxgrids = 4
    nviews = 5

    rotation_matrices = T.tensor3()
    shape_params = T.tensor4()
    out = gen_img(shape_params, rotation_matrices, width, height, nsteps, res)
    print "Compiling Render Function"
    render = function([shape_params, rotation_matrices], out)

    # voxel_data1 = load_voxels_binary("person_0089.raw", res, res, res)*10.0
    # voxel_data2 = load_voxels_binary("foot.raw", res, res, res)*10.0
    # voxel_data = np.stack([voxel_data1, voxel_data2])
    # r = random_rotation_matrices(3)
    # print "Rendering"
    # imgdata = f(voxel_data, r)

    views = T.tensor4() # nbatches * width * height
    cost, updates = second_order(rotation_matrices, views, width = width, height = height, nsteps = nsteps, res = res, nvoxgrids = nvoxgrids)
    print "Compiling ConvNet"
    cost_f = function([views, rotation_matrices], cost, updates = updates)
    train(cost_f, render, nviews = nviews, nvoxgrids = nvoxgrids, res = res)

main()

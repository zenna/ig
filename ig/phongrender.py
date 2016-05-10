# Build validation function
#
from __future__ import print_function

## Volume Raycasting
from theano import function, config, shared
import numpy as np
import time
import os

# Internal Imports
from ig.primitives import *
from ig.io import *
from ig.util import *
## Extract features from an image
import lasagne
import theano
import theano.sandbox.cuda.dnn
from lasagne.layers import InputLayer, DenseLayer, DropoutLayer
if theano.sandbox.cuda.dnn.dnn_available():
    from lasagne.layers.dnn import Conv2DDNNLayer as ConvLayer
    from lasagne.layers.dnn import Conv3DDNNLayer as Conv3DLayer
else:
    from lasagne.layers import Conv2DLayer as ConvLayer
    from conv3d import Conv2D3DLayer as Conv3DLayer

from lasagne.layers import MaxPool2DLayer as PoolLayer
from lasagne.layers import LocalResponseNormalization2DLayer as NormLayer
from lasagne.utils import floatX
from theano import tensor as T
from theano import function, config, shared
import pickle

from lasagne.nonlinearities import rectify

from lasagne.layers import batch_norm
# def batch_norm(x): return x
import numpy

from theano.compile.nanguardmode import NanGuardMode
curr_mode = None
# curr_mode = NanGuardMode(nan_is_error=True, inf_is_error=True, big_is_error=True)
# config.optimizer='fast_compile'
# config.optimizer='fast_compile'


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

def switch(cond, a, b):
    return cond*a + (1-cond)*b

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
    ro = np.array([0,0,1.5], dtype=config.floatX)

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

def cast(x,type, tnp = T):
    return x.astype(type)

def index_voxels(voxels, indices):
    """voxels : (nvoxgrids, res, res, res)
    indices : (npos, 3)
    returns (nvoxgrids, pos) - value at each of position in indices for each voxel grid
    """
    return voxels[:,indices[:,0],indices[:,1],indices[:,2]]

def get_indices(pos, res, tnp = T):
    """
    pos: (npos, 3)
    returns (npos, 3)
    """
    voxel_indices = tnp.floor(pos*res)
    clamped = tnp.clip(voxel_indices,0,res-1)
    p_int =  cast(clamped, 'int32', tnp = tnp)
    indices = tnp.reshape(p_int, (-1,3)) #
    return indices

def sample_volume(voxels, pos, res, tnp = T):
    """Samples voxels as pos positions
    voxels : (nvoxgrids, res, res, res)
    pos : (npos, 3)
    returns : (nvoxgrids, npos)"""
    indices = get_indices(pos, res, tnp = T)
    return voxels[:, indices[:,0],indices[:,1],indices[:,2]]

def attenuate(gdotl, indices, attenuation, ):
    grad_samples = voxels[:, indices[:,0],indices[:,1],indices[:,2]]

def normalise(x, axis, tnp = T):
    eps = 1e-9
    if tnp == T:
        return x/(x.norm(2, axis=axis)[:,:,np.newaxis] + eps) #IFIXME: generalise
    elif tnp == np:
        return x/((np.expand_dims(np.linalg.norm(x,axis=axis), axis)) + eps)
    else:
         raise ValueError

def gen_voxel_grid(res):
    "returns res * res * res * 3 grid"
    return np.transpose(np.mgrid[0:1:complex(res),0:1:complex(res),0:1:complex(res)],(1,2,3,0))

def compute_gradient(pos, voxels, res, n = 1, tnp = T):
    """Compute gradient using finite differences
    pos : (npos, 3)
    voxels : (num_voxels, res, res, res)
    returns : (num_voxels, npos, 3)"""
    x_delta_pos = floatX([n,0,0])
    x_delta_neg = floatX([-n,0,0])
    x1 = sample_volume(voxels, pos + x_delta_pos, res, tnp = tnp)
    x2 = sample_volume(voxels, pos + x_delta_neg, res, tnp = tnp)
    x_diff = x1 - x2   #nbatch * width * height * depth
    print("shp",x1.shape)
    print(x1)
    print("delta", x_delta_neg)

    y_delta_pos = floatX([0,n,0])
    y_delta_neg = floatX([0,-n,0])
    y1 = sample_volume(voxels, pos + y_delta_pos, res, tnp = tnp)
    y2 = sample_volume(voxels, pos + y_delta_neg, res, tnp = tnp)
    y_diff = y1 - y2   #nbatch * width * height * depth

    z_delta_pos = floatX([0,0,n])
    z_delta_neg = floatX([0,0,-n])
    z1 = sample_volume(voxels, pos + z_delta_pos, res, tnp = tnp)
    z2 = sample_volume(voxels, pos + z_delta_neg, res, tnp = tnp)
    z_diff = z1 - z2   #nbatch * width * height * depth
    gradients = tnp.stack([x_diff, y_diff, z_diff], axis=2)
    return normalise(gradients, axis=2, tnp = tnp)

def cartesian_product(arrays):
    broadcastable = numpy.ix_(*arrays)
    broadcasted = numpy.broadcast_arrays(*broadcastable)
    rows, cols = reduce(numpy.multiply, broadcasted[0].shape), len(broadcasted)
    out = numpy.empty(rows * cols, dtype=broadcasted[0].dtype)
    start, end = 0, rows
    for a in broadcasted:
        out[start:end] = a.reshape(-1)
        start, end = end, end + rows
    return out.reshape(cols, rows).T

def cube_filter(voxels, res, n = 1):
    """Take nvoxels,res,res,res voxels and return something of the same size"""
    indices_range = np.arange(res)
    indices = cartesian_product([indices_range, indices_range, indices_range])
    x_zero = indices[:,0]
    y_zero = indices[:,1]
    z_zero = indices[:,2]
    x_neg = np.clip(indices[:,0] - n, 0, res-1)
    x_add = np.clip(indices[:,0] + n, 0, res-1)
    y_neg = np.clip(indices[:,1] - n, 0, res-1)
    y_add = np.clip(indices[:,1] + n, 0, res-1)
    z_neg = np.clip(indices[:,2] - n, 0, res-1)
    z_add = np.clip(indices[:,2] + n, 0, res-1)
    v_x_add = voxels[:, x_add, y_zero, z_zero]
    v_x_neg = voxels[:, x_neg, y_zero, z_zero]
    v_y_add = voxels[:, x_zero, y_add, z_zero]
    v_y_neg = voxels[:, x_zero, y_neg, z_zero]
    v_z_add = voxels[:, x_zero, y_zero, z_add]
    v_z_neg = voxels[:, x_zero, y_zero, z_neg]

    voxels_flat = voxels.reshape((-1, res**3))
    voxels_mean = (v_x_add + v_x_neg + v_y_add + v_y_neg + v_z_add + v_z_neg + voxels_flat)/7.0
    return voxels_mean.reshape(voxels.shape)

    ## if i have a lsit of x values and y values and z values then
    ## When I have list of all the voxels, x,y,z
    ## Ill make 6 more, each perturbed by one of the axes.
    ## Then I'll take the average bitches.


def get_ts(voxels, nvoxgrids, nmatrices, rd, ro, width, height):
    a = 0 - ro # c = 0
    b = 1 - ro # c = 1
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
    return  t14, t04


def gen_img(voxels, rotation_matrix, width, height, nsteps, res, rgb, tnp = T):
    """Renders n voxel grids in m different views
    voxels : (n, res, res, res)
    rotation_matrix : (m, 4)
    returns (n, m, width, height))
    """
    nmatrices = rotation_matrix.shape[0]
    nvoxgrids = voxels.shape[0]
    raster_space = gen_fragcoords(width, height)
    rd, ro = make_ro(rotation_matrix, raster_space, width, height)
    t14, t04 = get_ts(voxels, nvoxgrids, nmatrices, rd, ro, width, height)

    dst = T.zeros((nvoxgrids, nmatrices * width * height, 4))
    step_size = (t14 - t04)/nsteps
    orig = T.reshape(ro, (nmatrices, 1, 1, 3)) + rd * T.reshape(t04,(nmatrices, width, height, 1))
    xres = yres = zres = res

    orig = T.reshape(orig, (nmatrices * width * height, 3))
    rd = T.reshape(rd, (nmatrices * width * height, 3))
    # Step size varies by ray because each ray intersects volum by different amount
    step_sz = T.reshape(step_size, (nmatrices * width * height,1))
    flat_step_sz = T.flatten(step_sz)

    # For each voxel grid, compute grid of dot product of light vector and voxel
    vox_grid = gen_voxel_grid(res)
    vox_grads = compute_gradient(vox_grid.reshape(-1,3), voxels, res, 1.0/nsteps, tnp = tnp)
    light_dir = floatX([[[0,1,1]]])
    gdotl = T.sum((light_dir * vox_grads), axis=2)
    gdotl_cube = gdotl.reshape((nvoxgrids, res, res, res))
    # Filter the gradients
    gdotl_cube = cube_filter(gdotl_cube, res, 1)
    gdotl_cube = cube_filter(gdotl_cube, res, 2)
    gdotl_cube = cube_filter(gdotl_cube, res, 3)
    gdotl_cube = cube_filter(gdotl_cube, res, 4)
    gdotl_cube = cube_filter(gdotl_cube, res, 5)
    gdotl_cube = T.maximum(0, gdotl_cube)

    for i in range(nsteps):
        # print "step", i
        pos = orig + rd*step_sz*i
        indices = get_indices(pos, res, tnp = T)
        attenuation = voxels[:, indices[:,0],indices[:,1],indices[:,2]]
        # attenuation = attenuation #* flat_step_sz # Scale by step size
        grad_samples = gdotl_cube[:, indices[:,0],indices[:,1],indices[:,2]]
        # rgb value at each position for each voxel
        rgb_scaled = 0.1 * rgb +  rgb * grad_samples[:,:,np.newaxis]
        rgba = T.concatenate([rgb_scaled, attenuation[:,:,np.newaxis]],axis=2)
        one_minus_a = (1 - dst[:,:,3])[:,:,np.newaxis]
        dst = dst + one_minus_a * rgba

    pixels = T.reshape(dst, (nvoxgrids, nmatrices, width, height, 4))
    mask = t14>t04
    return T.switch(mask[np.newaxis, :,:,:, np.newaxis], pixels, T.zeros_like(pixels))


def gen_vox(input_imgs, nmatrices, nvoxgrids, rotation_matrix, params, width, height, nsteps, res, rgb, tnp = T):
    """Renders n voxel grids in m different views
    input_imgs : (nvox, nrays, nchannels)
    rotation_matrix : (m, 4)
    params : (nsteps, 1, 1, 1)
    returns (n, m, width, height))
    """
    raster_space = gen_fragcoords(width, height)
    rd, ro = make_ro(rotation_matrix, raster_space, width, height)
    t14, t04 = get_ts(voxels, nvoxgrids, nmatrices, rd, ro, width, height)

    dst = T.zeros((nvoxgrids, nmatrices * width * height, 4))
    step_size = (t14 - t04)/nsteps
    orig = T.reshape(ro, (nmatrices, 1, 1, 3)) + rd * T.reshape(t04,(nmatrices, width, height, 1))
    xres = yres = zres = res

    orig = T.reshape(orig, (nmatrices * width * height, 3))
    rd = T.reshape(rd, (nmatrices * width * height, 3))
    # Step size varies by ray because each ray intersects volum by different amount
    step_sz = T.reshape(step_size, (nmatrices * width * height,1))
    flat_step_sz = T.flatten(step_sz)

    ## Output
    vox_out_means = T.zeros((nvoxgrids, res, res, res))
    # vox_out_var = T.zeros((nvoxgrids, res, res, res))
    input_imgs_flat = input_imgs.reshape((nvoxgrids, nmatrices * width * height))
    cn = input_imgs_flat
    params_flat = params.reshape((nsteps, nvoxgrids, 1))

    # step from n - 1 to 0
    for i in range(nsteps-2): # n-1 to 0
        print("Step", i)
        pos = orig + rd*step_sz*i
        indices = get_indices(pos, res, tnp = T)
        # Minus rgb from the img
        a = inv_plus_const(cn, rgb) # a = (1-a)c'
        theta = get_params(params_flat, i)
        b, c = inv_mul(a, theta)
        cn = b
        one_minus_c = 1 - c
        vox_out_means = update_voxel_means(vox_out_means, indices, one_minus_c)

    return vox_out_means/(nsteps-2)

def inv_plus_const(cn, rgb):
    """c' = ci (1-A)c'i-1. so the inverse is simply minusing rgb values
    cn: (nrays, )"""
    return cn - rgb

def get_params(params, i):
    """get parameters for ith iteration in inverse render"""
    return params[i]

def update_voxel_means(vox_out_means, indices, c):
    empty = T.zeros(vox_out_means.shape)
    filled = T.set_subtensor(empty[:, indices[:,0], indices[:,1], indices[:,2]], c)
    return vox_out_means + filled

# Mean square error
def mse(a, b):
    eps = 1e-9
    return (T.maximum(eps, (a - b)**2)).mean()

# Square error
def dist(a, b):
    eps = 1e-9
    return T.sum(T.maximum(eps, (a - b)**2))

# Mean distance to mean
def var(v, nvoxgrids, res):
    mean_voxels = T.mean(v, axis=0)
    return mse(mean_voxels, v)

def nparams(output_layer):
    return np.sum([q.flatten().shape[0] for q in lasagne.layers.get_all_param_values(output_layer)])

# def main(argv):
## Render with shading
global render, inv_render
width = height = 256
res = 128
nsteps = 100
rotation_matrices = T.tensor3()
voxels = T.tensor4()
# rgb = floatX([[[0.5,0.5,0.5]]])
rgb = floatX(0.5)

## Invere Render Function
print("Compiling Inverse Render Function")
input_imgs =  T.TensorType(dtype=T.config.floatX, broadcastable=(False,)*4)('input_imgs')
params = T.TensorType(dtype=T.config.floatX, broadcastable=(False,)*5)('params')
nvoxgrids = input_imgs.shape[0]
nmatrices = input_imgs.shape[1]
# nvoxgrids = 4
# nmatrices = 3
vox_means = gen_vox(input_imgs, nmatrices, nvoxgrids, rotation_matrices, params, width, height, nsteps, res, rgb)
inv_render = function([input_imgs, params, rotation_matrices], vox_means)

## Forward Render function
print("Compiling Render Function")
out = gen_img(voxels, rotation_matrices, width, height, nsteps, res, rgb)
render = function([voxels, rotation_matrices], out, mode=curr_mode)

## Test
voxel_data = floatX([load_voxels_binary(i, 128, 128, 128) for i in get_rnd_voxels(2)])
voxel_data = cube_filter(voxel_data, res)
voxel_data = cube_filter(voxel_data, res)
voxel_data = cube_filter(voxel_data, res)


views = rand_rotation_matrices(3)

print("Rendering Voxels")
imgs = render(floatX(voxel_data), views)




# def main(argv):
#     ## Args
#     global options
#     global render
#     global test_files, train_files
#     global net, output_layer, cost_f, cost_f_dict, val_f, call_f, call_f_dict
#     global views, voxels, outputs, net
#
#
#     options = handle_args(argv)
#     width = options['width'] = 64
#     height = options['height'] = 64
#     res = options['res'] = 64
#     nsteps = options['nsteps'] = 100
#     nvoxgrids = options['nvoxgrids'] = 8*8
#     nviews = options['nviews'] = 1
#     nepochs = options['nepochs'] = 10000
#
#     print(options)
#
#     rotation_matrices = T.tensor3()
#     voxels = T.tensor4()
#     out = gen_img(voxels, rotation_matrices, width, height, nsteps, res)
#     print("Compiling Render Function")
#     render = function([voxels, rotation_matrices], out, mode=curr_mode)
#
#     ## Training Function
#     views = T.tensor4() # nbatches * width * height
#     net, output_layer, outputs = second_order(rotation_matrices, views, voxels, width = width, height = height, nsteps = nsteps, res = res, nvoxgrids = nvoxgrids)
#     voxels = lasagne.layers.get_output(output_layer, deterministic = False)
#
#     losses = get_loss(net, voxels, voxels, nvoxgrids, res, output_layer)
#     loss = losses[0]
#     updates = get_updates(loss, output_layer, options)
#
#     # outputs.update({'loss1': loss1})
#     # outputs.update({'loss2': loss2})
#     outputs.update({'loss': loss})
#     outputs.update({'voxels': voxels})
#
#     selected_outputs = ['loss', 'voxels'] #+ net.keys()
#     print("Compiling Training Function")
#     cost_f = compile_conv_net(views, voxels, outputs, selected_outputs, updates)
#     cost_f_dict = named_outputs(cost_f, ['loss', 'voxels'])
#
#     ## Validation Function
#     val_voxels = lasagne.layers.get_output(output_layer, deterministic = True)
#     val_losses = get_loss(net, val_voxels, voxels, nvoxgrids, res, output_layer)
#     val_loss = val_losses[0]
#     outputs.update({'val_loss' : val_loss})
#     # outputs.update({'val_loss1' : val_loss1})
#     # outputs.update({'val_loss2' : val_loss2})
#     print("Compiling Validation Function")
#     val_f = compile_conv_net(views, voxels, outputs, ['loss'], None)
#     val_f_dict = named_outputs(cost_f, ['loss', 'voxels'])
#
#     to_print = ['loss']
#     to_save = ['loss', 'voxels', 'val_loss', 'filenames', 'param_values',
#                'width', 'height', 'res', 'nsteps', 'nvoxgrids', 'nviews',
#                'nepochs', 'learning_rate', 'momentum']
#
#     ## Call function
#     # print("Compiling Calling Function")
#     # call_f = compile_conv_net(views, voxels, outputs, net.keys(), None, on_unused_input='warn')
#     # call_f_dict = named_outputs(call_f, net.keys())
#
#     # Kinds of things I want to savea, (1) Output from function (2) Parameters (3) Constant values
#     test_files = filter(lambda x:x.endswith(".raw") and "test" in x, get_filepaths(os.getenv('HOME') + '/data/ModelNet40'))
#     train_files = filter(lambda x:x.endswith(".raw") and "train" in x, get_filepaths(os.getenv('HOME') + '/data/ModelNet40'))
#
#     train(cost_f_dict, val_f_dict, render, output_layer, options, test_files = test_files, train_files = train_files,
#           to_print = to_print, to_save = to_save, validate = True)
#
#if __name__ == "__main__":
#   main(sys.argv[1:])

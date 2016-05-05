# Build validation function
#
from __future__ import print_function

## Volume Raycasting
from theano import function, config, shared
import numpy as np
import time
import os

# Internal Imports
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

def get_indices(voxels, pos, res, tnp = T):
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
    indices = get_indices(voxels, pos, res, tnp = T)
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

def gen_img(voxels, rotation_matrix, width, height, nsteps, res, tnp = T):
    """Renders n voxel grids in m different views
    voxels : (n, res, res, res)
    rotation_matrix : (m, 4)
    returns (n, m, width, height))
    """
    nvoxgrids = voxels.shape[0]
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
    gdotl = T.maximum(0, gdotl)
    gdotl_cube = gdotl.reshape((nvoxgrids, res, res, res))
    rgb = floatX([[[0.9,0.2,0.3]]])  # We'd have one rgb value per voxel, andtherefore it would Have

    for i in range(nsteps):
        # print "step", i
        pos = orig + rd*step_sz*i
        indices = get_indices(voxels, pos, res, tnp = T)
        attenuation = voxels[:, indices[:,0],indices[:,1],indices[:,2]]
        # attenuation = attenuation #* flat_step_sz # Scale by step size
        grad_samples = gdotl_cube[:, indices[:,0],indices[:,1],indices[:,2]]
        # rgb value at each position for each voxel
        rgb_scaled = rgb * grad_samples[:,:,np.newaxis] + 0.1 * rgb
        one_minus_a = (1 - dst[:,:,3])[:,:,np.newaxis]
        ok = one_minus_a * rgb_scaled
        rgba = T.concatenate([ok, attenuation[:,:,np.newaxis]],axis=2)
        dst = dst + rgba

    pixels = T.reshape(dst, (nvoxgrids, nmatrices, width, height, 4))
    mask = t14>t04
    return T.switch(mask[np.newaxis, :,:,:, np.newaxis], pixels, T.zeros_like(pixels)), gdotl_cube

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
global render

width = height = 256
res = 128
nsteps = 100
rotation_matrices = T.tensor3()
voxels = T.tensor4()
out = gen_img(voxels, rotation_matrices, width, height, nsteps, res)
print("Compiling Render Function")
render = function([voxels, rotation_matrices], out, mode=curr_mode)

## Test
voxel_data = [load_voxels_binary(i, 128, 128, 128) for i in get_rnd_voxels(2)]
views = rand_rotation_matrices(3)

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
if __name__ == "__main__":
   main(sys.argv[1:])
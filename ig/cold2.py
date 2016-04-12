# Build validation function
#

## Volume Raycasting
from theano import function, config, shared, printing
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

from lasagne.layers import batch_norm
from lasagne.layers import MaxPool2DLayer as PoolLayer
from lasagne.layers import LocalResponseNormalization2DLayer as NormLayer
from lasagne.utils import floatX
from theano import tensor as T
from theano import function, config, shared
import pickle

from theano.compile.nanguardmode import NanGuardMode
curr_mode = None
# curr_mode = NanGuardMode(nan_is_error=True, inf_is_error=True, big_is_error=True)
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

    nvoxgrids = shape_params.shape[0]
    left_over = T.ones((nvoxgrids, nmatrices * width * height,))
    step_size = (t14 - t04)/nsteps
    orig = T.reshape(ro, (nmatrices, 1, 1, 3)) + rd * T.reshape(t04,(nmatrices, width, height, 1))
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
    return T.switch(t14>t04, pixels, T.ones_like(pixels)), rd, ro, tn_x, T.ones((nvoxgrids, nmatrices * width * height,)), orig, shape_params

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

def second_order(rotation_matrices, imagebatch, shape_params, width = 128, height = 128, nsteps = 100, res = 128, nvoxgrids = 4):
    """Creates a network which takes as input a image and returns a cost.
    Network extracts features of image to create shape params which are rendered.
    The similarity between the rendered image and the actual image is the cost
    """
    first_img = imagebatch[:,0,:,:]
    first_img = T.reshape(first_img, (nvoxgrids,1,width,height))

    net = {}
    net['input'] = InputLayer((None, 1, width, height), input_var = first_img)
    net['conv2d1'] = batch_norm(ConvLayer(net['input'], num_filters=res/4, filter_size=5, nonlinearity = lasagne.nonlinearities.rectify, W=lasagne.init.HeNormal(gain='relu'), pad='same' ))
    net['conv2d2'] = batch_norm(ConvLayer(net['conv2d1'], num_filters=res/2, filter_size=5, nonlinearity = lasagne.nonlinearities.rectify, W=lasagne.init.HeNormal(gain='relu'), pad='same' ))
    net['conv2d3'] = batch_norm(ConvLayer(net['conv2d2'], num_filters=res, filter_size=5, nonlinearity = lasagne.nonlinearities.rectify, W=lasagne.init.HeNormal(gain='relu'), pad='same' ))
    net['conv2d4'] = batch_norm(ConvLayer(net['conv2d3'], num_filters=res, filter_size=5, nonlinearity = lasagne.nonlinearities.rectify, W=lasagne.init.HeNormal(gain='relu'), pad='same' ))
    net['conv2d5'] = batch_norm(ConvLayer(net['conv2d4'], num_filters=res, filter_size=5, nonlinearity = lasagne.nonlinearities.rectify, W=lasagne.init.HeNormal(gain='relu'), pad='same'))
    last_conv2d_layer = net['conv2d5']
    net['reshape'] = lasagne.layers.ReshapeLayer(last_conv2d_layer, (nvoxgrids, 1, res, res, res))
    net['conv3d1'] = batch_norm(Conv3DLayer(net['reshape'], 4, (3,3,3), nonlinearity=lasagne.nonlinearities.rectify,W=lasagne.init.HeNormal(gain='relu'), pad='same', flip_filters=False))
    net['conv3d2'] = batch_norm(Conv3DLayer(net['conv3d1'], 4, (3,3,3), nonlinearity=lasagne.nonlinearities.rectify,W=lasagne.init.HeNormal(gain='relu'), pad='same'))
    net['conv3d3'] = batch_norm(Conv3DLayer(net['conv3d2'], 4, (3,3,3), nonlinearity=lasagne.nonlinearities.rectify,W=lasagne.init.HeNormal(gain='relu'),  pad='same'))
    net['conv3d4'] = batch_norm(Conv3DLayer(net['conv3d3'], 4, (3,3,3), nonlinearity=lasagne.nonlinearities.rectify,W=lasagne.init.HeNormal(gain='relu'), pad='same', flip_filters=False))
    net['conv3d5'] = batch_norm(Conv3DLayer(net['conv3d4'], 4, (3,3,3), nonlinearity=lasagne.nonlinearities.rectify,W=lasagne.init.HeNormal(gain='relu'), pad='same'))
    net['conv3d6'] = batch_norm(Conv3DLayer(net['conv3d5'], 4, (3,3,3), nonlinearity=lasagne.nonlinearities.rectify,W=lasagne.init.HeNormal(gain='relu'),  pad='same'))

    last_conv3d_layer = net['conv3d6']
    net['pooled'] = lasagne.layers.FeaturePoolLayer(last_conv3d_layer,4, pool_function=T.mean)
    net['voxels'] = lasagne.layers.ReshapeLayer(net['pooled'], (nvoxgrids, res, res, res))
    output_layer = net['voxels']
    outputs = {}
    voxels = lasagne.layers.get_output(output_layer)
    intermediate_outputs = {key : lasagne.layers.get_output(net[key]) for key in net.keys()}
    outputs.update(intermediate_outputs)
    return net, output_layer, outputs

def get_loss(net, voxels, shape_params, nvoxgrids, res, output_layer):
    # Voxel Variance loss
    loss1 = mse(voxels, shape_params)
    return (loss1,)
    # proposal_variance = var(voxels, nvoxgrids, res)
    # data_variance = var(shape_params, nvoxgrids, res)
    # loss2 = dist(proposal_variance, data_variance)

    # lambda1 = 1.0
    # lambda2 = 2.0
    # loss = lambda1 * loss1 + lambda2 * loss2
    # return loss, loss1, loss2

def get_updates(loss, output_layer):
    params = lasagne.layers.get_all_params(output_layer, trainable=True)
    lr = 0.1
    sh_lr = theano.shared(lasagne.utils.floatX(lr))

    # updates = lasagne.updates.nesterov_momentum(loss, params, learning_rate=0.01, momentum=0.9)
    # updates = lasagne.updates.adagrad(loss, params)
    updates = lasagne.updates.adamax(loss, params, learning_rate = lr)
    # updates = lasagne.updates.adam(loss, params, learning_rate=1e-4)
    # updates = lasagne.updates.momentum(loss, params, learning_rate=sh_lr, momentum=0.9)
    # updates = lasagne.updates.nesterov_momentum(loss, params, learning_rate=sh_lr, momentum=0.9)
    return updates, sh_lr

def build_conv_net(views, shape_params, outputs, selected_outputs, updates, mode = curr_mode):
    print "Building ConvNet with outputs", selected_outputs
    outputs_list = [outputs[so] for so in selected_outputs]
    return function([views, shape_params], outputs_list, updates = updates, mode=mode)

def load_parameters(output_layer, params_file):
    print "Loading Params", params_file
    param_values = np.load(params_file)['param_values']
    lasagne.layers.set_all_param_values(output_layer, param_values)

## Training
## ========

# def do_validation():
def iterate_minibatches(inputs, batchsize, shuffle=False):
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt]

def train(cost_f, val_f, render,  output_layer, nviews = 3, nvoxgrids=4, res = 128, save_data = True,
          nepochs = 100, save_every = 10, load_params = True, params_file = None,
          validate = True, validate_every = 50,
          fail_on_except = False, to_print = [], to_save = [], output_keys = [], lr = 0.1):
    """Learn Parameters for Neural Network"""
    print "Training"

    # Create directory with timestamp
    if save_data:
        datadir = os.environ['DATADIR']
        newdirname = str(time.time())
        full_dir_name = os.path.join(datadir, newdirname)
        print "Data will be saved to", full_dir_name
        os.mkdir(full_dir_name)
    if load_params:
        load_parameters(output_layer, params_file)

    # Validation
    test_files = filter(lambda x:x.endswith(".raw") and "test" in x, get_filepaths(os.getenv('HOME') + '/data/ModelNet40'))
    canonical_view = rand_rotation_matrices(1)
    val_loss = 0

    for i in range(nepochs):
        print "epoch: ", i, " of ", nepochs
        try:
            filenames = get_rnd_voxels(nvoxgrids)
            # print filenames
            voxel_data = [load_voxels_binary(v, 128, 128, 128, zoom = 0.5)*10.0 for v in filenames]
            voxel_dataX = [np.array(v,dtype=config.floatX) for v in voxel_data]
            r = rand_rotation_matrices(nviews)
            print "Rendering Training Data"
            imgdata = render(voxel_dataX, r)
            print "Computing Net"
            outputs = cost_f(imgdata[0], voxel_dataX)
            outputs_dict = dict(zip(output_keys, outputs))
            for key in to_print:
                print "%s: " % key, outputs_dict[key]
            print "learning_rate", lr.get_value()
            # Validation
            if validate and i % validate_every == 0:
                print "Assessing Validation Error"
                val_loss = 0
                val_minibatch_size = nvoxgrids
                j = 0
                for fnames in iterate_minibatches(test_files, val_minibatch_size):
                    voxel_data = [load_voxels_binary(v, 128, 128, 128, zoom = 0.5)*10.0 for v in fnames]
                    voxel_dataX = [np.array(v,dtype=config.floatX) for v in voxel_data]
                    imgdata = render(voxel_dataX, canonical_view)
                    [loss] = val_f(imgdata[0], voxel_dataX) # hack
                    val_loss = val_loss + loss
                    j = j + 1
                    print "validation error: ", loss
                val_loss = val_loss / j
                print val_loss

            if save_data and i % save_every == 0:
                fname = "epoch%s" % (i)
                full_fname = os.path.join(full_dir_name, fname)
                param_values = lasagne.layers.get_all_param_values(output_layer)
                to_save_dict = {key : outputs_dict[key] for key in to_save}
                to_save_dict['param_values'] = param_values
                to_save_dict['filenames'] = filenames
                to_save_dict['val_loss'] = val_loss
                np.savez_compressed(full_fname, **to_save_dict)
        except Exception as e:
            if fail_on_except:
                raise e
            else:
                print "Got error: ", e
                print "continuing"

def main(argv):
    width = 64
    height = 64
    res = 64
    nsteps = 100
    nvoxgrids = 8*8
    nviews = 1
    nepochs = 100000

    ## Args
    args = handle_args(argv)
    params_file = args['params_file']
    load_params = True
    if params_file == '':
        load_params = False

    rotation_matrices = T.tensor3()
    shape_params = T.tensor4()
    out = gen_img(shape_params, rotation_matrices, width, height, nsteps, res)
    print "Compiling Render Function"
    render = function([shape_params, rotation_matrices], out, mode=curr_mode)

    ## Training Function
    views = T.tensor4() # nbatches * width * height
    net, output_layer, outputs = second_order(rotation_matrices, views, shape_params, width = width, height = height, nsteps = nsteps, res = res, nvoxgrids = nvoxgrids)
    voxels = lasagne.layers.get_output(output_layer, deterministic = False)
    losses = get_loss(net, voxels, shape_params, nvoxgrids, res, output_layer)
    loss = losses[0]
    updates, lr = get_updates(loss, output_layer)

    # outputs.update({'loss1': loss1})
    # outputs.update({'loss2': loss2})
    outputs.update({'loss': loss})
    outputs.update({'voxels': voxels})

    selected_outputs = ['loss', 'voxels'] #+ net.keys()
    cost_f = build_conv_net(views, shape_params, outputs, selected_outputs, updates)

    ## Validation Function
    val_voxels = lasagne.layers.get_output(output_layer, deterministic = True)
    val_losses = get_loss(net, val_voxels, shape_params, nvoxgrids, res, output_layer)
    val_loss = val_losses[0]
    outputs.update({'val_loss' : val_loss})
    # outputs.update({'val_loss1' : val_loss1})
    # outputs.update({'val_loss2' : val_loss2})
    val_f = build_conv_net(views, shape_params, outputs, ['loss'], None)

    to_print = ['loss']
    to_save = ['loss', 'voxels']
    train(cost_f, val_f, render, output_layer, nviews = nviews, nvoxgrids = nvoxgrids, res = res,
          load_params=load_params, params_file=params_file, nepochs = nepochs,
          to_print = to_print, to_save = to_save, output_keys = selected_outputs, lr = lr)

if __name__ == "__main__":
   main(sys.argv[1:])

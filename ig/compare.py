## Learning to compare

# The idea is that the function will take as input a minibatch of images and a minibatch of scenes
# It will render those scenes to create a minibatch r(h)
# Then those images will be passed through some kind of convnet parameterised by theta, which will reduce it to a scalar
# We will find the gradient of the error with respect to the scenes.
# We will modify the scene by taking a step. Render the scene again.
# Then as output we will get images.
# We will take the euclidean distance.
# And get the derivatives of theta with respect to this distance.

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

import numpy as np
from ig.render import symbolic_render, make_render, gen_fragcoords
import pickle

# Config
config.scan.allow_output_prealloc = False
# config.optimizer = 'None'
config.exception_verbosity='high'

curr_mode = None

def go():
    return np.random.rand()*4 - 2

def gogo():
    return np.random.rand()*0.1

def genshapes(nprims):
    shapes = []
    for i in range(nprims):
        shapes.append([go(), go(), go(), gogo()])
    return np.array(shapes, dtype=config.floatX)

def genshapebatch(nprims, nbatch):
    shapes = np.random.rand(nprims, nbatch, 3)*2 - 2
    return np.array(shapes, dtype=config.floatX)

def learn_to_move(nprims = 200, nbatch = 50):
    """Creates a network which takes as input a image and returns a cost.
    Network extracts features of image to create shape params which are rendered.
    The similarity between the rendered image and the actual image is the cost
    """



    assert nbatch % 2 == 0      # Minibatch must be even in size
    width = 224
    height = 224
    params_per_prim = 3
    nshape_params = nprims * params_per_prim

    # Render the input shapes
    fragCoords = T.tensor3('fragCoords')
    shape_params = T.tensor3("scenes")
    res, scan_updates = symbolic_render(nprims, shape_params, fragCoords, width, height)
    res_reshape = res.dimshuffle([2,'x',0,1])

    # Split data set into two
    res_reshape_split = T.reshape(res_reshape, (nbatch/2, 2, 224, 224))

    # Put the different convnets into two channels
    net = {}
    net['input'] = InputLayer((nbatch/2, 2, 224, 224), input_var = res_reshape_split)
    net['conv1'] = ConvLayer(net['input'], num_filters=96, filter_size=7, stride=2)
    net['norm1'] = NormLayer(net['conv1'], alpha=0.0001) # caffe has alpha = alpha * pool_size
    net['pool1'] = PoolLayer(net['norm1'], pool_size=3, stride=3, ignore_border=False)
    net['conv2'] = ConvLayer(net['pool1'], num_filters=256, filter_size=5)
    net['pool2'] = PoolLayer(net['conv2'], pool_size=2, stride=2, ignore_border=False)
    net['conv3'] = ConvLayer(net['pool2'], num_filters=512, filter_size=3, pad=1)
    net['conv4'] = ConvLayer(net['conv3'], num_filters=512, filter_size=3, pad=1)
    net['conv5'] = ConvLayer(net['conv4'], num_filters=512, filter_size=3, pad=1)
    net['pool5'] = PoolLayer(net['conv5'], pool_size=3, stride=3, ignore_border=False)
    net['fc6'] = DenseLayer(net['pool5'], num_units=4096)
    net['drop6'] = DropoutLayer(net['fc6'], p=0.5)
    net['fc7'] = DenseLayer(net['drop6'], num_units=1, nonlinearity=lasagne.nonlinearities.rectify)
    # net['fc7'] = DenseLayer(net['pool5'], num_units=nshape_params, nonlinearity=lasagne.nonlinearities.tanh)
    output_layer = net['fc7']
    output = lasagne.layers.get_output(output_layer)

    #3 First half mvoe
    learning_rate = 0.1
    # first_half = res_reshape_split[:,0,:,:]
    # second_half = res_reshape_split[:,1,:,:]

    # Get partial derivatives of half of the.g parameters with respect to the cost and move them
    # Have to be careful about splitting to make sure that first half of params are those that render to
    # first channel of each image (and not that they render first half of all images in all channels)
    # shape_params_split = T.reshape(shape_params, (nprims, nbatch/2, 2, 4))
    delta_shape = T.grad(T.sum(output), shape_params)
    delta_shape_split = T.reshape(delta_shape, (nprims, nbatch/2, 2, 4))
    first_half_params = delta_shape_split[:,:,0,:]
    new_first_half = first_half_params + learning_rate * - delta_shape

    # Then render this half again to produce new images (width, height, nbatch/2)
    res2, scan_updates2 = symbolic_render(nprims, new_first_half, fragCoords, width, height)
    res_reshape2 = res2.dimshuffle([2,'x',0,1])

    # unchanged images
    unchanged_img = res_reshape_split[:,1,:,:]

    eps = 1e-9
    diff = T.maximum(eps, (unchanged_img - res_reshape2)**2)
    loss = T.sum(diff) / (nbatch/2*224*224)

    # Create update expressions for training, i.e., how to modify the
    # parameters at each training step. Here, we'll use Stochastic Gradient
    # Descent (SGD) with Nesterov momentum, but Lasagne offers plenty more.
    params = lasagne.layers.get_all_params(output_layer, trainable=True)
    network_updates = lasagne.updates.nesterov_momentum(loss, params, learning_rate=0.01, momentum=0.9)

    ## Merge Updates
    for k in network_updates.keys():
        assert not(scan_updates.has_key(k))
        scan_updates[k] = network_updates[k]

    for k in scan_updates2.keys():
        # assert not(scan_updates.has_key(k)) #FIXME
        scan_updates[k] = scan_updates2[k]

    print("Compiling Loss Function")
    netcost = function([fragCoords, shape_params], loss, updates=scan_updates, mode=curr_mode)
    return netcost, output_layer

# import ig.display
def train(network, costfunc, nprims = 200, nbatch = 50, num_epochs = 500):
    width = 224
    height = 224
    exfragcoords = gen_fragcoords(width, height)
    print("Starting Training")
    for epoch in range(num_epochs):
        rand_data = genshapebatch(nprims, nbatch)
        test_err = costfunc(exfragcoords, rand_data)
        print "loss1", test_err

    return lasagne.layers.get_all_param_values(network)

def network_mb(network):
    o = lasagne.layers.get_all_param_values(network)
    q = np.concatenate([pp.flatten() for pp in o])
    return (float(len(q))*32) / 1024.0**2

nprims = 100
nbatch = 4
costfunc, network = learn_to_move(nprims = nprims, nbatch = nbatch)
# print "Weights in MB"
# print network_mb(network)
train(network, costfunc, nprims = nprims, nbatch = nbatch)

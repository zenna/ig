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

def second_order(nprims = 200, nbatch = 50):
    """Creates a network which takes as input a image and returns a cost.
    Network extracts features of image to create shape params which are rendered.
    The similarity between the rendered image and the actual image is the cost
    """
    width = 224
    height = 224
    params_per_prim = 3
    nshape_params = nprims * params_per_prim

    img = T.tensor4("input image")
    net = {}
    net['input'] = InputLayer((nbatch, 1, 224, 224), input_var = img)
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
    # net['fc7'] = DenseLayer(net['drop6'], num_units=nshape_params, nonlinearity=lasagne.nonlinearities.sigmoid)
    net['fc7'] = DenseLayer(net['pool5'], num_units=nshape_params)
    output_layer = net['fc7']
    output = lasagne.layers.get_output(output_layer)
    scaled_output = output * 2 - 2

    ## Render these parameters
    shape_params = T.reshape(scaled_output, (nprims, nbatch, params_per_prim))
    fragCoords = T.tensor3('fragCoords')
    print "Symbolic Render"
    res, scan_updates = symbolic_render(nprims, shape_params, fragCoords, width, height)
    res_reshape = res.dimshuffle([2,'x',0,1])

    # Simply using pixel distance
    eps = 1e-9

    loss = T.sum(T.maximum(eps, (res_reshape - img)**2)) / (224*224)

    # Create update expressions for training, i.e., how to modify the
    # parameters at each training step. Here, we'll use Stochastic Gradient
    # Descent (SGD) with Nesterov momentum, but Lasagne offers plenty more.
    params = lasagne.layers.get_all_params(output_layer, trainable=True)
    network_updates = lasagne.updates.nesterov_momentum(loss, params, learning_rate=0.01, momentum=0.9)

    ## Merge Updates
    for k in network_updates.keys():
        assert not(scan_updates.has_key(k))
        scan_updates[k] = network_updates[k]

    print("Compiling Loss Function")
    grad = T.grad(loss, params[0])
    netcost = function([fragCoords, img], [loss, grad, res_reshape, shape_params], updates=scan_updates, mode=curr_mode)
    # netcost = function([fragCoords, img], loss, updates=scan_updates, mode=curr_mode)

    ## Generate Render Function to make data
    # Generate initial rays
    exfragcoords = gen_fragcoords(width, height)
    print("Compiling Renderer")
    render = make_render(nprims, width, height)
    return render, netcost, output_layer

def train(network, nprims = 200, nbatch = 50, num_epochs = 500):
    width = 224
    height = 224
    exfragcoords = gen_fragcoords(width, height)
    print("Starting Training")
    for epoch in range(num_epochs):
        rand_data = genshapebatch(nprims, nbatch)
        print("Rendering Test Data")
        test_data = render(exfragcoords, rand_data)
        print("Computing Loss")
        test_err = netcost(exfragcoords, np.reshape(test_data, (nbatch,1,width, height)))
        print(test_err[0])
        print(test_err[3])
        # print("  test loss:\t\t\t{:.6f}".format(test_err))

    return lasagne.layers.get_all_param_values(network)

def network_mb(network):
    o = lasagne.layers.get_all_param_values(network)
    q = np.concatenate([pp.flatten() for pp in o])
    return (float(len(q))*32) / 1024.0**2

nprims = 200
nbatch = 10
render, netcost, network = second_order(nprims = nprims, nbatch = nbatch)
print "Weights in MB"
print network_mb(network)
#train(network, nprims = nprims, nbatch = nbatch)

#!/usr/bin/env python

"""
Usage example employing Lasagne for digit recognition using the MNIST dataset.

This example is deliberately structured as a long flat file, focusing on how
to use Lasagne, instead of focusing on writing maximally modular and reusable
code. It is used as the foundation for the introductory Lasagne tutorial:
http://lasagne.readthedocs.org/en/latest/user/tutorial.html

More in-depth examples and reproductions of paper results are maintained in
a separate repository: https://github.com/Lasagne/Recipes
"""

from __future__ import print_function
from theano.tensor.nlinalg import matrix_inverse
import sys
import os
import time

import numpy as np
import theano
import theano.tensor as T

from invbatchnorm import *

import lasagne
from permute import PermuteLayer, Eye
from nonlinearities import *
from theano.compile.nanguardmode import NanGuardMode
from lasagne.regularization import regularize_layer_params, l2


curr_mode = None
# curr_mode = NanGuardMode(nan_is_error=True, inf_is_error=True, big_is_error=True)
# theano.config.optimizer = 'None'
# theano.config.optimizer = 'fast_compile'


# ################## Download and prepare the MNIST dataset ##################
# This is just some way of getting the MNIST dataset from an online location
# and loading it into numpy arrays. It doesn't involve Lasagne at all.

def load_dataset():
    # We first define a download function, supporting both Python 2 and 3.
    if sys.version_info[0] == 2:
        from urllib import urlretrieve
    else:
        from urllib.request import urlretrieve

    def download(filename, source='http://yann.lecun.com/exdb/mnist/'):
        print("Downloading %s" % filename)
        urlretrieve(source + filename, filename)

    # We then define functions for loading MNIST images and labels.
    # For convenience, they also download the requested files if needed.
    import gzip

    def load_mnist_images(filename):
        if not os.path.exists(filename):
            download(filename)
        # Read the inputs in Yann LeCun's binary format.
        with gzip.open(filename, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=16)
        # The inputs are vectors now, we reshape them to monochrome 2D images,
        # following the shape convention: (examples, channels, rows, columns)
        data = data.reshape(-1, 1, 28, 28)
        # The inputs come as bytes, we convert them to float32 in range [0,1].
        # (Actually to range [0, 255/256], for compatibility to the version
        # provided at http://deeplearning.net/data/mnist/mnist.pkl.gz.)
        return data / np.float32(256)

    def load_mnist_labels(filename):
        if not os.path.exists(filename):
            download(filename)
        # Read the labels in Yann LeCun's binary format.
        with gzip.open(filename, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=8)
        # The labels are vectors of integers now, that's exactly what we want.
        return data

    # We can now download and read the training and test set images and labels.
    X_train = load_mnist_images('train-images-idx3-ubyte.gz')
    y_train = load_mnist_labels('train-labels-idx1-ubyte.gz')
    X_test = load_mnist_images('t10k-images-idx3-ubyte.gz')
    y_test = load_mnist_labels('t10k-labels-idx1-ubyte.gz')

    # We reserve the last 10000 training examples for validation.
    X_train, X_val = X_train[:-10000], X_train[-10000:]
    y_train, y_val = y_train[:-10000], y_train[-10000:]

    # We just return all the arrays in order, as expected in main().
    # (It doesn't matter how we do this as long as we can read them again.)
    return X_train, y_train, X_val, y_val, X_test, y_test


# ##################### Build the neural network model #######################
# This script supports three types of models. For each one, we define a
# function that takes a Theano variable representing the input and returns
# the output layer of a neural network model built in Lasagne.

def build_mlp(input_var=None):
    # This creates an MLP of two hidden layers of 800 units each, followed by
    # a softmax output layer of 10 units. It applies 20% dropout to the input
    # data and 50% dropout to the hidden layers.

    # Input layer, specifying the expected input shape of the network
    # (unspecified batchsize, 1 channel, 28 rows and 28 columns) and
    # linking it to the given Theano variable `input_var`, if any:
    l_in = lasagne.layers.InputLayer(shape=(None, 1, 28, 28),
                                     input_var=input_var)

    # Apply 20% dropout to the input data:
    l_in_drop = lasagne.layers.DropoutLayer(l_in, p=0.2)

    # Add a fully-connected layer of 800 units, using the linear rectifier, and
    # initializing weights with Glorot's scheme (which is the default anyway):
    l_hid1 = lasagne.layers.DenseLayer(
            l_in_drop, num_units=28*28,
            nonlinearity=s_rectify,
            W=lasagne.init.GlorotUniform())

    # We'll now add dropout of 50%:
    l_hid1_drop = lasagne.layers.DropoutLayer(l_hid1, p=0.5)

    # Another 800-unit layer:
    l_hid2 = lasagne.layers.DenseLayer(
            l_hid1_drop, num_units=28*28,
            nonlinearity=s_rectify)

    # 50% dropout again:
    l_hid2_drop = lasagne.layers.DropoutLayer(l_hid2, p=0.5)

    l_slice = lasagne.layers.SliceLayer(l_hid2_drop, indices=slice(0,10),axis = 1)
    l_out = lasagne.layers.NonlinearityLayer(l_slice, nonlinearity = lasagne.nonlinearities.softmax)
    return l_out

def make_inv(y, p1, p2, nlayers, gammas, betas, weights, inv_stds, means):
    inv_network_layers = []
    mlow =  T.max(1.0/y, axis=1)
    e = np.array(np.exp(1.0),dtype=T.config.floatX)
    mhigh = T.min(e/y, axis=1)
    m = p1*(mhigh-mlow) + mlow
    unsoftmax = T.log(y*m.dimshuffle(0, 'x'))
    lastl = T.concatenate([unsoftmax,p2],axis=1)

    inv_network = lasagne.layers.InputLayer(shape=(None, 28*28),
                                     input_var=lastl)
    j = 1
    for i in range(nlayers):
        inv_network = lasagne.layers.NonlinearityLayer(inv_network, nonlinearity = inv_nonlinearity)
        inv_network_layers.append(inv_network)
        inv_network = InvBatchNormLayer(inv_network, gamma = gammas[-j], beta=betas[-j], inv_std=inv_stds[-j], mean=means[-j])
        inv_network_layers.append(inv_network)
        inv_network = lasagne.layers.DenseLayer(inv_network, 28*28, W = matrix_inverse(weights[-j]), b=None, nonlinearity=None)
        inv_network_layers.append(inv_network)
        j = j + 1

    return inv_network, inv_network_layers, p1, p2

def build_inv_mlp(input_var=None, param_var=None):
    ## OK so noninvertibility, i.e. using an s_rectify is incorrect for the obvious reasons
    ## What I want is a function which is invertible and defined on the unit interval.
    ## Why is it difficult to keep the transformations within specifeid bounds.
    ## It would
    network_layers = []
    forward_nonlinearity = leaky_rectify
    inv_nonlinearity = inv_leaky_rectify
    network = lasagne.layers.InputLayer(shape=(None, 1, 28, 28),
                                     input_var=input_var)
    # l0n = lasagne.layers.NonlinearityLayer(l0, nonlinearity=expand)
    # Apply 20% dropout to the input data:

    gammas = []
    betas = []
    weights= []
    means = []
    inv_stds = []

    nlayers = 2
    for i in range(nlayers):
        network = lasagne.layers.DropoutLayer(network, p=0.2)
        network = lasagne.layers.DenseLayer(
                network, num_units=28*28,
                nonlinearity=None,
                W=lasagne.init.GlorotUniform(),
                b=None)
        network_layers.append(network)
        weights.append(network.W)
        network = lasagne.layers.BatchNormLayer(network)
        network_layers.append(network)
        betas.append(network.beta)
        gammas.append(network.gamma)
        inv_stds.append(network.inv_std)
        means.append(network.mean)
        network = lasagne.layers.NonlinearityLayer(network, nonlinearity=forward_nonlinearity)
        network_layers.append(network)

    network = lasagne.layers.DropoutLayer(network, p=0.2)
    network = lasagne.layers.SliceLayer(network, indices=slice(0,10),axis = 1)
    network = lasagne.layers.NonlinearityLayer(network, nonlinearity = lasagne.nonlinearities.softmax)
    out_tensor = lasagne.layers.get_output(network,batch_norm_use_averages=False, batch_norm_update_averages=True)
    fwd_params = lasagne.layers.get_all_params(network, trainable=True)

    # y = T.matrix("y")
    y = out_tensor
    p1 = T.vector("p1")
    p2 = T.matrix("p2")
    inv_network, inv_network_layers = make_inv(y, p1, p2, nlayers, gammas, betas, weights, inv_stds, means)



    return network, inv_network, network_layers, inv_network_layers, y, p1, p2, [lasagne.layers.get_output(i,deterministic=True) for i in network_layers], [lasagne.layers.get_output(i) for i in inv_network_layers]

def build_mlp2(input_var=None):
    network_layers = []
    network = lasagne.layers.InputLayer(shape=(None, 1, 28, 28),
                                     input_var=input_var)
    network_layers.append(network)
    network = lasagne.layers.DropoutLayer(network, p=0.2)
    network_layers.append(network)

    x = lasagne.layers.ReshapeLayer(network, ([0],-1))
    nblocks = 5
    ninblock = 2
    # First layer with nonuniform init
    network = lasagne.layers.DenseLayer(
        network, num_units=28*28,
        nonlinearity=s_rectify,
        W=lasagne.init.GlorotUniform())
        # W = np.eye(28*28,28*28)
    network_layers.append(network)
    network = lasagne.layers.DropoutLayer(network, p=0.5)

    print("Nblocks, ninblock layers:", nblocks, ninblock)
    for j in range(nblocks):
        for i in range(ninblock):
            # Add a fully-connected layer of 800 units, using the linear rectifier, and
            # initializing weights with Glorot's scheme (which is the default anyway):
            network = lasagne.layers.DenseLayer(
                    network, num_units=28*28,
                    nonlinearity=s_rectify,
                    # W=lasagne.init.GlorotUniform())
                    # W = np.random.rand(28*28,28*28))
                    # W = np.eye(28*28,28*28)
                    W = Eye())
            network_layers.append(network)
            network = lasagne.layers.DropoutLayer(network, p=0.5)
            network_layers.append(network)
        # network = lasagne.layers.ElemwiseSumLayer([network, x])
        # network_layers.append(network)
        x = network

    # 50% dropout again:
    network = lasagne.layers.DropoutLayer(network, p=0.5)
    network_layers.append(network)

    network = lasagne.layers.SliceLayer(network, indices=slice(0,10),axis = 1)
    network_layers.append(network)
    network = lasagne.layers.NonlinearityLayer(network, nonlinearity = lasagne.nonlinearities.softmax)
    network_layers.append(network)

    # # Finally, we'll add the fully-connected output layer, of 10 softmax units:
    # network = lasagne.layers.DenseLayer(
    #         network, num_units=10,
    #         nonlinearity=lasagne.nonlinearities.softmax)
    # network_layers.append(network)
    return network


def build_custom_mlp(input_var=None, depth=2, width=800, drop_input=.2,
                     drop_hidden=.5):
    # By default, this creates the same network as `build_mlp`, but it can be
    # customized with respect to the number and size of hidden layers. This
    # mostly showcases how creating a network in Python code can be a lot more
    # flexible than a configuration file. Note that to make the code easier,
    # all the layers are just called `network` -- there is no need to give them
    # different names if all we return is the last one we created anyway; we
    # just used different names above for clarity.

    # Input layer and dropout (with shortcut `dropout` for `DropoutLayer`):
    network = lasagne.layers.InputLayer(shape=(None, 1, 28, 28),
                                        input_var=input_var)
    if drop_input:
        network = lasagne.layers.dropout(network, p=drop_input)
    # Hidden layers and dropout:
    nonlin = lasagne.nonlinearities.rectify
    for _ in range(depth):
        network = lasagne.layers.DenseLayer(
                network, width, nonlinearity=nonlin)
        if drop_hidden:
            network = lasagne.layers.dropout(network, p=drop_hidden)
    # Output layer:
    softmax = lasagne.nonlinearities.softmax
    network = lasagne.layers.DenseLayer(network, 10, nonlinearity=softmax)
    return network


def build_cnn(input_var=None):
    # As a third model, we'll create a CNN of two convolution + pooling stages
    # and a fully-connected hidden layer in front of the output layer.

    # Input layer, as usual:
    network = lasagne.layers.InputLayer(shape=(None, 1, 28, 28),
                                        input_var=input_var)
    # This time we do not apply input dropout, as it tends to work less well
    # for convolutional layers.

    # Convolutional layer with 32 kernels of size 5x5. Strided and padded
    # convolutions are supported as well; see the docstring.
    network = lasagne.layers.Conv2DLayer(
            network, num_filters=32, filter_size=(5, 5),
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform())
    # Expert note: Lasagne provides alternative convolutional layers that
    # override Theano's choice of which implementation to use; for details
    # please see http://lasagne.readthedocs.org/en/latest/user/tutorial.html.

    # Max-pooling layer of factor 2 in both dimensions:
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))

    # Another convolution with 32 5x5 kernels, and another 2x2 pooling:
    network = lasagne.layers.Conv2DLayer(
            network, num_filters=32, filter_size=(5, 5),
            nonlinearity=lasagne.nonlinearities.rectify)
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))

    # A fully-connected layer of 256 units with 50% dropout on its inputs:
    network = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(network, p=.5),
            num_units=256,
            nonlinearity=lasagne.nonlinearities.rectify)

    # And, finally, the 10-unit output layer with 50% dropout on its inputs:
    network = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(network, p=.5),
            num_units=10,
            nonlinearity=lasagne.nonlinearities.softmax)

    return network

def build_pcnn(input_var=None):
    # As a third model, we'll create a CNN of two convolution + pooling stages
    # and a fully-connected hidden layer in front of the output layer.

    # Input layer, as usual:
    network = lasagne.layers.InputLayer(shape=(None, 1, 28, 28),
                                        input_var=input_var)
    # This time we do not apply input dropout, as it tends to work less well
    # for convolutional layers.

    # Convolutional layer with 32 kernels of size 5x5. Strided and padded
    # convolutions are supported as well; see the docstring.
    network = PermuteLayer(network)
    network = lasagne.layers.Conv2DLayer(
            network, num_filters=32, filter_size=(5, 5),
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform())
    # Expert note: Lasagne provides alternative convolutional layers that
    # override Theano's choice of which implementation to use; for details
    # please see http://lasagne.readthedocs.org/en/latest/user/tutorial.html.

    # Max-pooling layer of factor 2 in both dimensions:
    network = PermuteLayer(network)
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))

    # Another convolution with 32 5x5 kernels, and another 2x2 pooling:
    network = PermuteLayer(network)
    network = lasagne.layers.Conv2DLayer(
            network, num_filters=32, filter_size=(5, 5),
            nonlinearity=lasagne.nonlinearities.rectify)
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))

    # A fully-connected layer of 256 units with 50% dropout on its inputs:
    network = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(network, p=.5),
            num_units=256,
            nonlinearity=lasagne.nonlinearities.rectify)

    # And, finally, the 10-unit output layer with 50% dropout on its inputs:
    network = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(network, p=.5),
            num_units=10,
            nonlinearity=lasagne.nonlinearities.softmax)

    return network

def build_pcnn2(input_var=None):
    # As a third model, we'll create a CNN of two convolution + pooling stages
    # and a fully-connected hidden layer in front of the output layer.

    # Input layer, as usual:
    network = lasagne.layers.InputLayer(shape=(None, 1, 28, 28),
                                        input_var=input_var)
    network_layers = []
    network_layers.append(network)
    network = lasagne.layers.Conv2DLayer(
            network, num_filters=32, filter_size=(5, 5),
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform(),pad='same')
    network_layers.append(network)
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))
    network_layers.append(network)

    # Another convolution with 32 5x5 kernels, and another 2x2 pooling:
    network = lasagne.layers.Conv2DLayer(
            network, num_filters=32, filter_size=(5, 5),
            nonlinearity=lasagne.nonlinearities.rectify)
    network_layers.append(network)

    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))
    network_layers.append(network)

    ## Permutation Layers
    x = network
    nblocks = 5
    ninblock = 2
    print("Nblocks, ninblock layers:", nblocks, ninblock)
    for j in range(nblocks):
        for i in range(ninblock):
            # A fully-concted layer of 256 units with 50% dropout on its inputs:
            network = PermuteLayer(network)
            network_layers.append(network)
            network = lasagne.layers.Conv2DLayer(
                    network, num_filters=32, filter_size=(5, 5),
                    nonlinearity=lasagne.nonlinearities.rectify,
                    W=lasagne.init.GlorotUniform(),pad='same')
            network_layers.append(network)
        network = lasagne.layers.ElemwiseSumLayer([network, x])
        network_layers.append(network)
        x = network

    network = lasagne.layers.FlattenLayer(network)
    network_layers.append(network)

    network = lasagne.layers.SliceLayer(network,indices=slice(0,10), axis=1)
    network = lasagne.layers.NonlinearityLayer(network, lasagne.nonlinearities.softmax)
    # network = lasagne.layers.DenseLayer(
    #     lasagne.layers.dropout(network, p=.5),
    #     num_units=10,
    #     nonlinearity=lasagne.nonlinearities.softmax)

    network_layers.append(network)

    return network


# ############################# Batch iterator ###############################
# This is just a simple helper function iterating over training data in
# mini-batches of a particular size, optionally in random order. It assumes
# data is available as numpy arrays. For big datasets, you could load numpy
# arrays as memory-mapped files (np.load(..., mmap_mode='r')), or write your
# own custom data iteration function. For small datasets, you can also copy
# them to GPU at once for slightly improved performance. This would involve
# several changes in the main program, though, and is not demonstrated here.

def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]


def bound_loss(x, tnp = np) :
  eps = 1e-9
  loss = tnp.maximum(tnp.maximum(eps,x-1), tnp.maximum(eps,-x)) + eps
  return tnp.maximum(loss, eps) + eps


# ############################## Main program ################################
# Everything else will be handled in our main program now. We could pull out
# more functions to better separate the code, but it wouldn't make it any
# easier to read.

def inv_batch_norm(y, mean, inv_sigma, beta, gamma):
    return (y - beta)/(gamma * inv_sigma) + mean

def main(model='mlp', num_epochs=500):
    # Load the dataset
    print("Loading data...")
    X_train, y_train, X_val, y_val, X_test, y_test = load_dataset()

    # Prepare Theano variables for inputs and targets
    input_var = T.tensor4('inputs')
    target_var = T.ivector('targets')

    # Create neural network model (depending on first command line parameter)
    print("Building model and compiling functions...")
    global params
    global network
    global inv_network
    global network_layers, inv_network_layers
    global call_fn

    if model == 'mlp':
        print("Building mlp")
        network = build_mlp(input_var)
    elif model.startswith('custom_mlp:'):
        depth, width, drop_in, drop_hid = model.split(':', 1)[1].split(',')
        network = build_custom_mlp(input_var, int(depth), int(width),
                                   float(drop_in), float(drop_hid))
    elif model == 'cnn':
        print("Building cnn")
        network = build_cnn(input_var)
    elif model == 'pcnn':
        print("Building pcnn")
        network = build_pcnn(input_var)
    elif model == 'pcnn2':
        print("Building pcnn2")
        network = build_pcnn2(input_var)
    elif model == 'mlp2':
        print("Building mlp2")
        network = build_mlp2(input_var)
    elif model == 'inv_mlp':
        print("Building inv_")
        network, inv_network, network_layers, inv_network_layers, y, p1, p2, outputs, inv_outputs = build_inv_mlp(input_var)
    else:
        print("Unrecognized model type %r." % model)
        return

    # Create a loss expression for training, i.e., a scalar objective we want
    # to minimize (for our multi-class problem, it is the cross-entropy loss):
    prediction = lasagne.layers.get_output(network, batch_norm_use_averages=False, batch_norm_update_averages=True)
    loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
    loss = loss.mean()
    # We could add some weight decay as well here, see lasagne.regularization.


    ## Inverse Loss
    inv_op = lasagne.layers.get_output(inv_network)
    # # Inversion should be within the training set bounds
    bl1 = bound_loss(inv_op, tnp = T).mean()/100

    # # Parameter should be within some reasonable bounds.
    # p_op = lasagne.layers.get_output(network_layers[2])
    # bl2 = bound_loss(p_op, tnp = T).mean()/10000
    l2_penalty_1 = regularize_layer_params(network_layers[1], l2)
    l2_penalty_2 = regularize_layer_params(network_layers[2], l2)

    total_loss = bl1 + loss# + l2_penalty_1 + l2_penalty_2

    losses = [loss, bl1, total_loss] #, l2_penalty_1, l2_penalty_2, total_loss]
    print(losses)
    # total_loss = loss


    # Create update expressions for training, i.e., how to modify the
    # parameters at each training step. Here, we'll use Stochastic Gradient
    # Descent (SGD) with Nesterov momentum, but Lasagne offers plenty more.
    params = lasagne.layers.get_all_params(network, trainable=True)
    # updates = lasagne.updates.nesterov_momentum(
    #         loss, params, learning_rate=0.01, momentum=0.9)
    updates = lasagne.updates.adam(
        total_loss, params, learning_rate=1e-3)
    # updates = lasagne.updates.momentum(
    #     total_loss, params, learning_rate=0.1)
    # updates = lasagne.updates.adagrad(total_loss, params)


    # Create a loss expression for validation/testing. The crucial difference
    # here is that we do a deterministic forward pass through the network,
    # disabling dropout layers.
    test_prediction = lasagne.layers.get_output(network, deterministic=True)
    test_loss = lasagne.objectives.categorical_crossentropy(test_prediction,
                                                            target_var)
    test_loss = test_loss.mean()
    # As a bonus, also create an expression for the classification accuracy:
    test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var),
                      dtype=theano.config.floatX)

    # Compile a function performing a training step on a mini-batch (by giving
    # the updates dictionary) and returning the corresponding training loss:
    train_fn = theano.function([input_var, target_var, p1, p2], losses, updates=updates, mode=curr_mode)

    # Compile a second function computing the validation loss and accuracy:
    val_fn = theano.function([input_var, target_var], [test_loss, test_acc], mode=curr_mode)

    # For calling the damn thing and getting some output
    call_fn = theano.function([input_var], outputs, mode=curr_mode)

    print("Loading Params")
    # with np.load('model.npz') as f:
    #     param_values = [f['arr_%d' % i] for i in range(len(f.files))]
    # lasagne.layers.set_all_param_values(network, param_values)


    # Finally, launch the training loop.
    print("Starting training...")
    # We iterate over epochs:
    for epoch in range(num_epochs):
        # In each epoch, we do a full pass over the training data:
        train_err = 0
        train_batches = 0
        start_time = time.time()
        j = 0
        atime = time.time()
        for batch in iterate_minibatches(X_train, y_train, 500, shuffle=True):
            inputs, targets = batch
            currbatchsize = inputs.shape[0]
            p1dat = np.array(np.random.rand(currbatchsize), dtype=T.config.floatX)
            p2dat = np.array(np.random.rand(currbatchsize, 28*28-10), dtype=T.config.floatX)
            output = train_fn(inputs, targets, p1dat, p2dat)
            param_values = lasagne.layers.get_all_param_values(network)
            # np.savez_compressed("/home/zenna/data/sandbox/%sepoch%s_%s" % (atime, epoch, j), *param_values)
            train_err += output[0]
            print(output)
            train_batches += 1
            if j == 0:
                print("maxmin", np.max(inputs), np.min(inputs))
            j = j + 1

        # And a full pass over the validation data:
        val_err = 0
        val_acc = 0
        val_batches = 0
        for batch in iterate_minibatches(X_val, y_val, 500, shuffle=False):
            inputs, targets = batch
            err, acc = val_fn(inputs, targets)
            val_err += err
            val_acc += acc
            val_batches += 1

        # Then we print the results for this epoch:
        print("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, num_epochs, time.time() - start_time))
        print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
        print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))
        print("  validation accuracy:\t\t{:.2f} %".format(
            val_acc / val_batches * 100))
        # print(lasagne.layers.get_all_params(network))
        # print(lasagne.layers.get_all_param_values(network)[0])
        # print(lasagne.layers.get_all_param_values(network)[3])
        # print(lasagne.layers.get_all_param_values(network)[4])

    # After training, we compute and print the test error:
    test_err = 0
    test_acc = 0
    test_batches = 0
    for batch in iterate_minibatches(X_test, y_test, 500, shuffle=False):
        inputs, targets = batch
        err, acc = val_fn(inputs, targets)
        test_err += err
        test_acc += acc
        test_batches += 1
    print("Final results:")
    print("  test loss:\t\t\t{:.6f}".format(test_err / test_batches))
    print("  test accuracy:\t\t{:.2f} %".format(
        test_acc / test_batches * 100))

    # Optionally, you could now dump the network weights to a file like this:
    np.savez('model.npz', *lasagne.layers.get_all_param_values(network))

    #
    # And load them again later on like this:



if __name__ == '__main__':
    if ('--help' in sys.argv) or ('-h' in sys.argv):
        print("Trains a neural network on MNIST using Lasagne.")
        print("Usage: %s [MODEL [EPOCHS]]" % sys.argv[0])
        print()
        print("MODEL: 'mlp' for a simple Multi-Layer Perceptron (MLP),")
        print("       'custom_mlp:DEPTH,WIDTH,DROP_IN,DROP_HID' for an MLP")
        print("       with DEPTH hidden layers of WIDTH units, DROP_IN")
        print("       input dropout and DROP_HID hidden dropout,")
        print("       'cnn' for a simple Convolutional Neural Network (CNN).")
        print("EPOCHS: number of training epochs to perform (default: 500)")
    else:
        kwargs = {}
        if len(sys.argv) > 1:
            kwargs['model'] = sys.argv[1]
        if len(sys.argv) > 2:
            kwargs['num_epochs'] = int(sys.argv[2])
        main(**kwargs)


def test():
    X_train, y_train, X_val, y_val, X_test, y_test = load_dataset()
    inp_img = X_train[3339].reshape(1,1,28,28)
    out = call_fn(inp_img)
    def softmax(x): return np.array(np.exp(x)/np.sum(np.exp(x)), dtype=T.config.floatX)
    ydat = softmax([[4.0,0.2,.3,0.2,0.2,0.2,.3,0.2,0.2,.1]])
    ydat = out[-1]
    p1dat = np.array([0.5], dtype=T.config.floatX)
    p2dat = np.array(np.random.rand(1,28*28-10),dtype=T.config.floatX)
    iout = inv_f(X_train[3339].reshape(1,1,28,28),p1dat,p2dat)
    fuzz = iout[-1].reshape(1,1,28,28)
    outout = call_fn(fuzz)
    reconstruction = outout[-1]
    inv_is_is_working = np.sum(reconstruction - ydat) # should be zero!

    # mlow = T.max(1.0/y, axis=1)
    # mhigh = T.min(np.exp(1)/y, axis=1)
    # m = p1*(mlow-mhigh) + mlow
    # unsoftmax = T.log(y*m)

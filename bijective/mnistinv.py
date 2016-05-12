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
from theano import printing
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
# theano.config.optimizer = 'None'
# curr_mode=NanGuardMode(nan_is_error=True, inf_is_error=True, big_is_error=False)
curr_mode = None

forward_nonlinearity = leaky_rectify
inv_nonlinearity = inv_leaky_rectify

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

def make_inv(y, p1, p2, nlayers, gammas, betas, weights, inv_stds, means):
    inv_network_layers = []
    mlow =  T.max(1.0/y, axis=1)
    e = np.array(np.exp(1.0),dtype=T.config.floatX)
    mhigh = T.min(e/y, axis=1)
    m = p1*(mhigh-mlow) + mlow
    unsoftmax = T.log(y*m.dimshuffle(0, 'x'))
    # unsoftmax = T.log(y*p1.dimshuffle(0, 'x'))
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

    return inv_network, inv_network_layers

def build_inv_mlp(input_var=None, param_var=None):
    ## OK so noninvertibility, i.e. using an s_rectify is incorrect for the obvious reasons
    ## What I want is a function which is invertible and defined on the unit interval.
    ## Why is it difficult to keep the transformations within specifeid bounds.
    ## It would
    network_layers = []
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
    network_layers.append(network)
    network = lasagne.layers.NonlinearityLayer(network, nonlinearity = lasagne.nonlinearities.softmax)
    network_layers.append(network)
    out_tensor = lasagne.layers.get_output(network,batch_norm_use_averages=False, batch_norm_update_averages=True)
    fwd_params = lasagne.layers.get_all_params(network, trainable=True)

    print("Compiling Forward Function")
    global call_fn
    out_tensor_vals = [lasagne.layers.get_output(layer,batch_norm_use_averages=True, batch_norm_update_averages=False, deterministic = True) for layer in network_layers]
    call_fn = theano.function([input_var], out_tensor_vals, mode=curr_mode)

    # y = T.matrix("y")
    # y = printing.Print('y')(out_tensor)
    y = out_tensor
    p1 = T.vector("p1")
    p2 = T.matrix("p2")
    inv_network, inv_network_layers = make_inv(y, p1, p2, nlayers, gammas, betas, weights, inv_stds, means)

    global inv_fn
    y_call = T.matrix()
    p1_call = T.vector("p1")
    p2_call = T.matrix("p2")
    inv_network_call, inv_network_layers_call = make_inv(y_call, p1_call, p2_call, nlayers, gammas, betas, weights, inv_stds, means)
    inv_network_out = [lasagne.layers.get_output(layer, deterministic=True) for layer in inv_network_layers_call]
    print("Compiling Inverse Function")
    inv_fn = theano.function([y_call,p1_call,p2_call], inv_network_out, mode=curr_mode)

    return network, inv_network, network_layers, inv_network_layers, y, p1, p2, [lasagne.layers.get_output(i,deterministic=True) for i in network_layers], [lasagne.layers.get_output(i) for i in inv_network_layers]


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

def get_update_state(updates, update_indices):
    return [updates.keys()[i].get_value() for i in update_indices]

def set_update_state(updates, update_indices, update_state):
    j = 0
    for i in update_indices:
        keys = updates.keys()
        keys[i].set_value(update_state[j])
        j = j + 1

def save_update_state(updates, fname, indices):
    update_state = get_update_state(updates, indices)
    np.savez_compressed(fname, *update_state)

def load_update_state(updates, fname, indices):
    loaded_update_state = np.load(fname)
    as_array = npz_to_array(loaded_update_state)
    print("sums", [np.sum(x) for x in as_array])
    set_update_state(updates, indices, as_array)

def npz_to_array(npzfile):
    nitems = len(npzfile.keys())
    return [npzfile['arr_%s' % i]  for i in range(nitems)]

def main(model='mlp', num_epochs=500):
    # Load the dataset
    print("Loading data...")
    X_train, y_train, X_val, y_val, X_test, y_test = load_dataset()
    X_train = X_train + np.array(np.random.rand(*X_train.shape)/256.0, dtype='float32')

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
    blx = bound_loss(inv_op, tnp = T)
    bl1 = blx.mean()
    # bl2 = blx.max()

    p_op = lasagne.layers.get_output(network_layers[-3])
    bl2 = bound_loss(p_op, tnp = T).mean() * 10

    ## I want (i) 774 values to be within 0, 1 and
    ## I want sum of weights to be between 0 and 1.
    ## but that's challenging because if the weight sum is between 0 and 1, then this implies something on the
    ## values which is not the same thing.
    ## Let's just say I want all of these values to be between 0 and 1.

    total_loss = loss + bl1 + bl2 # + bl2# + loss# + l2_penalty_1 + l2_penalty_2
    losses = [loss, bl1, bl2, total_loss]
    # losses = [loss, bl1, bl2, total_loss] #, l2_penalty_1, l2_penalty_2, total_loss]
    print(losses)
    # total_loss = loss
    params = lasagne.layers.get_all_params(network, trainable=True)
    all_params = lasagne.layers.get_all_params(network)
    print("params", params)
    print("all params", all_params)
    load_params = True
    if load_params:
        # data = np.load("/home/zenna/data/sandbox/1462218357.93epoch5.npz")
        # data = np.load("/home/zenna/data/sandbox/1462809191.22epoch211.npz")
        data = np.load("/home/zenna/data/sandbox/1462872665.36epoch51.npz")

        param_values = [data['arr_%s' % i]  for i in range(10)]
        lasagne.layers.set_all_param_values(network, param_values)
    # updates = lasagne.updates.nesterov_momentum(
    #         loss, params, learning_rate=0.01, momentum=0.9)
    global updates
    updates = lasagne.updates.adam(
        total_loss, params, learning_rate=1e-3)
    # updates = lasagne.updates.momentum(
    #     total_loss, params, learning_rate=1e-5)
    # updates = lasagne.updates.adagrad(total_loss, params)

    ## Reload optimisation Parameters
    update_indices = [0,1,3,4,6,7,9,10,12,13,15]

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
    global train_fn
    train_fn = theano.function([input_var, target_var, p1, p2], losses, updates=updates, mode=curr_mode, on_unused_input='warn')
    if load_params:
        # load_update_state(updates, "/home/zenna/data/sandbox/1462218357.93_updates4_100.npz", update_indices)
        # load_update_state(updates, "/home/zenna/data/sandbox/1462809191.22_updates210_100.npz", update_indices)
        load_update_state(updates, "/home/zenna/data/sandbox/1462872665.36_updates50_100.npz", update_indices)

    # Compile a second function computing the validation loss and accuracy:
    val_fn = theano.function([input_var, target_var], [test_loss, test_acc], mode=curr_mode)

    print("Loading Params")
    # with np.load('model.npz') as f:
    #     param_values = [f['arr_%d' % i] for i in range(len(f.files))]
    # lasagne.layers.set_all_param_values(network, param_values)


    # Finally, launch the training loop.
    print("Starting training...")
    # We iterate over epochs:
    atime = time.time()
    for epoch in range(num_epochs):
        # In each epoch, we do a full pass over the training data:
        train_err = 0
        train_batches = 0
        start_time = time.time()
        j = 0
        print(atime)
        param_values = lasagne.layers.get_all_param_values(network)
        np.savez_compressed("/home/zenna/data/sandbox/%sepoch%s" % (atime, epoch), *param_values)
        save_update_state(updates, "/home/zenna/data/sandbox/update_%sepoch%s" % (atime, epoch), update_indices)

        batch_size = 500

        for batch in iterate_minibatches(X_train, y_train, batch_size, shuffle=True):
            inputs, targets = batch
            currbatchsize = inputs.shape[0]
            p1dat = np.array(np.random.rand(currbatchsize), dtype=T.config.floatX)
            p2dat = np.array(np.random.rand(currbatchsize, 28*28-10), dtype=T.config.floatX)
            output = train_fn(inputs, targets, p1dat, p2dat)
            train_err += output[0]
            print(output)
            train_batches += 1
            if j == 0:
                print("maxmin", np.max(inputs), np.min(inputs))
            j = j + 1
            # if j % 10 == 0:
            #     save_update_state(updates, "/home/zenna/data/sandbox/%s_updates%s_%s.npz" % (atime, epoch, j), update_indices)

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

## The pre softmax values in the forward pass range from like -10 to 1
# However in the inverse pass they're about 0 to 10.  This is because the normalisation constant is very differennt

def test():
    X_train, y_train, X_val, y_val, X_test, y_test = load_dataset()
    def ok(i):
        inp_img = X_train[i].reshape(1,1,28,28)
        out = call_fn(inp_img)
        def softmax(x): return np.array(np.exp(x)/np.sum(np.exp(x)), dtype=T.config.floatX)
        ydat = out[-1]
        p1dat = np.array([0.5], dtype=T.config.floatX)
        p2dat = np.array(np.random.rand(1,28*28-10),dtype=T.config.floatX)
        iout = inv_fn(ydat,p1dat,p2dat)
        fuzz = iout[-1].reshape(1,1,28,28)
        outout = call_fn(fuzz)
        return out, outout, iout, fuzz
    out, outout, iout, fuzz = ok(6)
    np.min(fuzz), np.max(fuzz)
    reconstruction = outout[-1]
    inv_is_is_working = np.sum(np.abs((reconstruction - ydat))) # should be zero!

    # mlow = T.max(1.0/y, axis=1)
    # mhigh = T.min(np.exp(1)/y, axis=1)
    # m = p1*(mlow-mhigh) + mlow
    # unsoftmax = T.log(y*m)

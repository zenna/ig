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

def vgg_features(img):
    """Returns tuple of img features when passed through pre-trained vgg network

    Parameters
    ----------
    img : 3*224*224 theano tensor.
    """
    net = {}
    net['input'] = InputLayer((None, 3, 224, 224), input_var = img)
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
    net['fc7'] = DenseLayer(net['drop6'], num_units=4096)
    net['drop7'] = DropoutLayer(net['fc7'], p=0.5)
    net['fc8'] = DenseLayer(net['drop7'], num_units=1000, nonlinearity=lasagne.nonlinearities.softmax)
    output_layer = net['fc8']

    model = pickle.load(open('data/vgg_cnn_s.pkl'))
    CLASSES = model['synset words']
    MEAN_IMAGE = model['mean image']

    lasagne.layers.set_all_param_values(output_layer, model['values'])
    params = lasagne.layers.get_all_params(output_layer)
    inp_th = lasagne.layers.get_output(net['input'])
    conv1_th = lasagne.layers.get_output(net['conv1'])
    norm1_th = lasagne.layers.get_output(net['norm1'])
    pool1_th = lasagne.layers.get_output(net['pool1'])
    conv2_th = lasagne.layers.get_output(net['conv2'])
    pool2_th = lasagne.layers.get_output(net['pool2'])
    conv3_th = lasagne.layers.get_output(net['conv3'])
    conv4_th = lasagne.layers.get_output(net['conv4'])
    conv5_th = lasagne.layers.get_output(net['conv5'])
    pool5_th = lasagne.layers.get_output(net['pool5'])
    fc6_th = lasagne.layers.get_output(net['fc6'])
    drop6_th = lasagne.layers.get_output(net['drop6'])
    fc7_th = lasagne.layers.get_output(net['fc7'])
    output_layer_th = lasagne.layers.get_output(output_layer)
    #return (conv1_th, norm1_th, pool1_th, conv2_th, pool2_th, conv3_th, conv4_th,
    #        conv5_th, pool5_th, fc6_th, drop6_th, fc7_th, output_layer_th)
    return (inp_th, conv1_th, norm1_th, pool1_th, conv2_th, pool2_th, conv3_th, conv4_th,
            conv5_th, pool5_th)

def gen_vgg():
    """Generate an executable VGG Network"""
    inp_img = T.tensor4()
    output = vgg_features(inp_img)
    return function([inp_img], output)

def feature_compare(features, observed_features):
    # return T.sum((features[8] - observed_features[8])**2)
    eps = 1e-9
    nfeatures = 6
    dists = [T.maximum(eps, (features[i] - observed_features[i])**2) for i in range(nfeatures)]
    summed_dists = [T.sum(dists[i])/observed_features[i].size for i in range(nfeatures)]
    return summed_dists

from theano.tensor.nnet.conv3d2d import conv3d
from lasagne.layers.conv import BaseConvLayer
from lasagne import init
from lasagne import nonlinearities

class Conv2D3DLayer(BaseConvLayer):
    def __init__(self, incoming, num_filters, filter_size,
                 untie_biases=False, W=init.GlorotUniform(),
                 b=init.Constant(0.), nonlinearity=nonlinearities.rectify,
                 flip_filters=False, **kwargs):
        stride=(1, 1, 1)
        super(Conv2D3DLayer, self).__init__(incoming, num_filters,
                                             filter_size, stride, pad,
                                             untie_biases, W, b, nonlinearity,
                                             flip_filters, n=3, **kwargs)

    def convolve(self, input, **kwargs):
        # Conv3d expects input  [n_images, depth, channels, height, width]
        weights = self.W.dimshuffle(0, 2, 1, 3, 4)
        input_sh = input.dimshuffle(0, 2, 1, 3, 4)
        conved = conv3d(input_sh, weights, signals_shape=None, filters_shape=None, border_mode='valid')
        conved_sh = conved.dimshuffle(0, 2, 1, 3, 4)
        return conved_sh

## Test
# img_batch_shape = (2,1,10,10,10)
# img_inp = lasagne.layers.InputLayer(img_batch_shape)
# conv_layer = Conv2D3DLayer(img_inp, 4, (3,3,3))
# op = lasagne.layers.get_output(conv_layer)
# g = theano.function([img_inp.input_var], op)
#
# rnd_img = np.random.rand(*img_batch_shape)
# g(rnd_img)

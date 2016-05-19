import numy as np
from lasagne.layers import TransformerLayer
from lasagne.layers import DenseLayer
from lasagne.init import HeNormal, Constant
from lasagne.nonlinearities import softmax, rectify, sigmoid
import theano.sandbox.cuda.dnn
if theano.sandbox.cuda.dnn.dnn_available():
    from lasagne.layers.dnn import Conv2DDNNLayer as ConvLayer
    from lasagne.layers.dnn import Conv3DDNNLayer as Conv3DLayer
else:
    from lasagne.layers import Conv2DLayer as ConvLayers


def warp_layer(l_in, out_xy, sfx, params):
    inp_shape = l_in.output_shape
    x_in, y_in = inp_shape[2], inp_shape[3]
    sample_factor = (float(x_in)/float(out_xy[0]),
                     float(y_in)/float(out_xy[1]))
    W = lasagne.init.Constant(0.0)
    W = params['W_%s' % sfx, lasagne.init.Constant(0.0)]
    b = floatX(np.zeros((2, 3)))
    b[0, 0] = 1
    b[1, 1] = 1
    b = b.flatten()
    b = shared(b)
    params['b_%s' % sfx, b]
    l_loc = lasagne.layers.DenseLayer(l_in, num_units=6, W=W, b=b,
                                      nonlinearity=None)
    l_out = TransformerLayer(l_in, l_loc, downsample_factor=sample_factor)
    params.set("W_%s" % sfx, l_loc.W)
    assert l_out.output_shape[2:] == out_xy
    return l_out


def mchannels(shape):
    assert len(shape) == 4
    return shape[1]

def warp_conv_net(*inputs, **kwargs):
    """A residual convolutional network of n inputs and m outputs."""
    inp_shapes = kwargs['inp_shapes']
    out_shapes = kwargs['out_shapes']
    params = kwargs['params']
    width, height = kwargs['width'], kwargs['height']
    nblocks = kwargs['nblocks']
    block_size = kwargs['block_size']
    nfilters = kwargs['nfilters']
    npixels = width * height
    ninputs = len(inp_shapes)
    noutputs = len(out_shapes)
    output_args = kwargs['output_args']

    print("Res-convnet with: %s blocks of size %s" % (nblocks, block_size))

    # Each input is projected to a channel of the input image.
    channel_sizes = []
    input_channels = []
    for i in range(len(inputs)):
        inp_ndim = len(inp_shapes[i])
        if inp_ndim == 4 and inp_shapes[i][2] == width and inp_shapes[i][3] == height:
            print("input ", i, " does not need reshaping or projecting")
            i = InputLayer(inp_shapes[i], input_var=inputs[i])
            input_channels.append(i)
            channel_sizes.append(inp_shapes[i][1])
        elif inp_ndim == 4:
            # It is an image but the wrong size
            print("input", i, "is image but of wrong size, rescaling")
            i = InputLayer(inp_shapes[i], input_var=inputs[i])
            warped = warp_layer(i, (width, height), "in_warp_%s" % i, params)
            input_channels.append(warped)
            channel_sizes.append(inp_shapes[i][1])
        else:
            print("input", i, "is not image, reshaping to vector then warping")
            i = InputLayer(inp_shapes[i], input_var=inputs[i])
            r = ReshapeLayer(i, ([0], 1, 1, -1))
            warped = warp_layer(r, (width, height), "in_warp_%s" % i, params)
            input_channels.append(warped)
            channel_sizes.append(1)

    net = {}
    # input_Shape = (batch_size, ninputs, width, height)
    # concatenate over channels
    net['input_img'] = prev_layer = ConcatLayer(input_channels, axis=1)

    # Convolutional Layers
    wx_sfx = 'wx'
    wx = batch_norm_params(ConvLayer(prev_layer,
        num_filters=nfilters, filter_size=1, nonlinearity = rectify,
        W=params['W_%s' % wx_sfx, HeNormal(gain='relu')],
        b=params['b_%s' % wx_sfx, Constant(0)],
        pad='same'), wx_sfx, params)
    # net['resizeblock'] = prev_layer = x = lasagne.layers.ElemwiseSumLayer([wx, prev_layer])

    # 2d convolutional blocks
    for j in range(nblocks):
      for i in range(block_size):
          sfx = "%s_%s" % (j,i)
          net['conv2d%s_%s' % (j,i)] = prev_layer = batch_norm_params(ConvLayer(prev_layer,
            num_filters=nfilters, filter_size=3, nonlinearity = rectify,
            W=params['W_%s' % sfx, HeNormal(gain='relu')],
            b=params['b_%s' % sfx, Constant(0)],
            pad='same'), sfx, params)
      if nblocks > 1:
          net['block2d%s' % j] = wx = prev_layer = lasagne.layers.ElemwiseSumLayer([prev_layer, wx])

    out_nchannel = []
    out_xys = []
    for i in range(noutputs):
        out_shape == out_shapes[i]
        if len(out_shape) == 4:
            out_nchannel.append(out_shape[1])
            out_xy = out_shape[2:]
        else:
            out_nchannel.append(1)
            out_xy = (1, np.prod(out_shape[1:]))
        assert len(out_xy) == 2
        out_xys.append(out_xy)
    nout_channels = sum(out_nchannel)

    sfx = 'final_conv'
    net['final_conv'] = prev_layer = batch_norm_params(ConvLayer(prev_layer,
      num_filters=nout_channels, filter_size=3, nonlinearity = rectify,
      W=params['W_%s' % sfx, HeNormal(gain='relu')],
      b=params['b_%s' % sfx, Constant(0)],
      pad='same'), sfx, params)

    # Output Projection
    net['output'] = prev_layer
    output_product = lasagne.layers.get_output(net['output'], **output_args)
    pre_warp_params = params_from_layer(net['output'])
    outputs = []

    warp_params = []
    lb = 0
    for i in range(noutputs):
        out_shape == out_shapes[i]
        output = outputs[:, lb:out_nchannel[i]]
        o = InputLayer((None, out_nchannel[i], width, height),
                       input_var=output)
        w = warped_layer(o, out_xys[i], "out_warp_%s" % i, params)
        if nchannels(out_shape) != 4:
            w = ReshapeLayer(w, ([0],) + out_shape[1:])
        warp_params.append(params_from_layer(w))
        outputs.append(get_output(w))
        lb = lb + out_nchannel[i]

    assert lb == nout_channels

    all_params = pre_warp_params + warp_params
    print("all_params", all_params)
    print("pre_warp_params", params)
    print("warp_params", params)
    params.add_tagged_params(all_params)
    params.check(lasagne.layers.get_all_params(prev_layer))
    return outputs, params

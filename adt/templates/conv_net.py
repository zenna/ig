def conv_res_net(*inputs, **kwargs):
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

    input_width = np.sum([in_shape[1] for in_shape in inp_shapes])
    output_width = np.sum([out_shape[1] for out_shape in out_shapes])

    print("Res-convnet with: %s blocks of size %s" % (nblocks, block_size))

    # Each input is projected to a channel of the input image.
    # May need to project
    input_channels = []
    for i in range(len(inputs)):
        inp_ndim = len(inp_shapes[i])
        if inp_ndim == 4 and inp_shapes[i][2] == width and inp_shapes[i][3] == height:
            print("input ", i, " does not need reshaping or projecting")
            i = InputLayer(inp_shapes[i], input_var = inputs[i])
            input_channels.append(i)
        else:
            raise ValueError("Inputs must all be the right, same, size")
        #   print("input ", i, "of shape", inp_shapes[i][1:], " needs projecting to ", layer_shape)
        #   i = Input_layer(inp_shapes[i], input_var = inputs[i])
        #   r = ReshapeLayer(i, ([0], 1, 1, inp_size))
        #   c = ConvLayer(i, num_filters=npixels, filter_size=1, nonlinearity = rectify, W=lasagne.init.HeNormal(gain='relu'), pad='same' ))
        #   d = DenseLayer(i, npixels)
        #   r = ReshapeLayer(d, (1, width, height))
        #   input_channels.append(r)

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

    sfx = 'final_conv'
    net['final_conv'] = prev_layer = batch_norm_params(ConvLayer(prev_layer,
      num_filters=noutputs, filter_size=3, nonlinearity = rectify,
      W=params['W_%s' % sfx, HeNormal(gain='relu')],
      b=params['b_%s' % sfx, Constant(0)],
      pad='same'), sfx, params)

    ## Output Projection
    net['output'] = prev_layer
    output_product = lasagne.layers.get_output(net['output'], **output_args)
    outputs = []
    for i in range(noutputs):
        outputs.append(output_product[:, i:i+1])

    all_params = get_layer_params(lasagne.layers.get_all_layers(net['output']))
    print("ALL_PARAMS", all_params)
    print("PARAMS", params)
    params.add_tagged_params(all_params)
    params.check(lasagne.layers.get_all_params(prev_layer))
    return outputs, params

def get_layer_params(layers):
    params = OrderedDict()
    for layer in layers:
        params.update(layer.params)
    return params


def params_from_layer(alayer):
    layers = lasagne.layers.get_all_layers(alayer)
    return get_layer_params(layer)

def handle_batch_norm(params, suffix, bn_layer):
  params.set("W_%s" % suffix, bn_layer.input_layer.input_layer.W)
  params.set("b_%s" % suffix, bn_layer.input_layer.input_layer.b)
  params.set("beta_%s" % suffix, bn_layer.input_layer.beta)
  params.set("gamma_%s" % suffix, bn_layer.input_layer.gamma)
  params.set("inv_std_%s" % suffix, bn_layer.input_layer.inv_std)
  params.set("mean_%s" % suffix, bn_layer.input_layer.mean)

def batch_norm_params(input, sfx, params):
    l = batch_norm(input,
        beta=params['beta_%s' % sfx, Constant(0)], gamma=params['gamma_%s' % sfx, Constant(1)],
        mean=params['mean_%s' % sfx, Constant(0)], inv_std=params['inv_std_%s' % sfx, Constant(1)])
    handle_batch_norm(params, sfx, l)
    return l

# def batch_norm_params(input, sfx, params):
#     params.set("W_%s" % sfx, input.W)
#     params.set("b_%s" % sfx, input.b)
#     return input

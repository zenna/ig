import lasagne
from theano import function

class TrainParams():
    def __init__(self, adt, train_fn, call_fns, generators, gen_to_inputs,
                 train_outs, hyperparams):
        return self


def get_updates(loss, params, options):
    updates = {}
    print("Params", params)
    if options['update'] == 'momentum':
        updates = lasagne.updates.momentum(loss, params, learning_rate=options['learning_rate'], momentum=options['momentum'])
    elif options['update'] == 'adam':
        updates = lasagne.updates.adam(loss, params, learning_rate=options['learning_rate'])
    elif options['update'] == 'rmsprop':
        updates = lasagne.updates.rmsprop(loss, params, learning_rate=options['learning_rate'])
    return updates


def get_losses(axioms):
    losses = []
    for axiom in axioms:
        for loss in axiom.get_losses():
            losses.append(loss)
    return losses


def get_params(funcs, options, **tags):
    params = []
    for func in funcs:
        for param in func.get_params(**tags):
            params.append(param)

    return params


def compile_fns(funcs, consts, forallvars, axioms, train_outs, options):
    print("Compiling training fn...")
    losses = get_losses(axioms)
    func_params = get_params(funcs, options, trainable=True)
    constant_params = get_params(consts, options)
    params = func_params + constant_params
    loss = sum(losses)
    outputs = train_outs + losses
    updates = get_updates(loss, params, options)
    train_fn = function([forallvar.input_var for forallvar in forallvars],
                        outputs, updates=updates)
    # Compile the func for use
    if options['compile_fns']:
        print("Compiling func fns...")
        call_fns = [func.compile() for func in funcs]
    else:
        call_fns = []
    # FIXME Trainable=true, deterministic = true/false
    return train_fn, call_fns


def train(train_fn, generators, gen_to_inputs, ntrain_outs,
          num_epochs=1000, summary_gap=100):
    """One epoch is one pass through the data set"""
    print("Starting training...")
    for epoch in range(num_epochs):
        train_err = 0
        train_batches = 0
        start_time = time.time()
        train_outs = None
        [gen.next() for gen in generators]
        for i in range(summary_gap):
            gens = [gen.send(train_outs) for gen in generators]
            inputs = gen_to_inputs(gens)
            train_outs_losses = train_fn(*inputs)
            train_outs = train_outs_losses[0:ntrain_outs]
            losses = train_outs_losses[ntrain_outs:]
            print("epoch: ", epoch, "losses: ", losses)
            train_err += losses[0]
            train_batches += 1
            gens = [gen.next() for gen in generators]
        print("epoch: ", epoch, " Total loss per epoch: ", train_err)


def train_pbt(pbt, **kwargs):
    train(pbt.train_fn, pbt.generators, pbt.gen_to_inputs, len(pbt.train_outs),
          **kwargs)

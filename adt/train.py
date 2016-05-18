import lasagne
from theano import function
import time
from lasagne.updates import *
import os
import numpy as np


class TrainParams():
    def __init__(self, adt, train_fn, call_fns, generators,  gen_to_inputs,
                 train_outs, hyperparams):
        return self


def get_updates(loss, params, options):
    updates = {}
    print("Params", params)
    if options['update'] == 'momentum':
        updates = momentum(loss, params, learning_rate=options['learning_rate'],
                           momentum=options['momentum'])
    elif options['update'] == 'adam':
        updates = adam(loss, params, learning_rate=options['learning_rate'])
    elif options['update'] == 'rmsprop':
        updates = rmsprop(loss, params, learning_rate=options['learning_rate'])
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


def train(adt, pdt, num_epochs=1000, summary_gap=100, save_every=10, sfx='',
          save_dir="./"):
    """One epoch is one pass through the data set"""
    print("Starting training...")
    for epoch in range(num_epochs):
        train_err = 0
        train_batches = 0
        start_time = time.time()
        ntrain_outs = len(pdt.train_outs)
        train_outs = None
        [gen.next() for gen in pdt.generators]
        for i in range(summary_gap):
            gens = [gen.send(train_outs) for gen in pdt.generators]
            inputs = pdt.gen_to_inputs(gens)
            train_outs_losses = pdt.train_fn(*inputs)
            train_outs = train_outs_losses[0:ntrain_outs]
            losses = train_outs_losses[ntrain_outs:]
            print("epoch: ", epoch, "losses: ", losses)
            train_err += losses[0]
            train_batches += 1
            gens = [gen.next() for gen in pdt.generators]
            if i % save_every == 0:
                sfx2 = "epoch_%s_run_%sloss_%s" % (epoch, i, str(np.sum(losses)))
                path = os.path.join(save_dir, sfx2)
                adt.save_params(path)
        print("epoch: ", epoch, " Total loss per epoch: ", train_err)

    path = os.path.join(save_dir, "final" + sfx)
    adt.save_params(path)

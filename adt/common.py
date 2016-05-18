from ig.util import *
from train import *
import theano.tensor as T


def gen_sfx_key(keys, options):
    sfx_dict = {}
    for key in keys:
        sfx_dict[key] = options[key]
    sfx = stringy_dict(sfx_dict)
    print("sfx:", sfx)
    return sfx


def repeat_to_batch(x, batch_size, tnp=T):
    return tnp.repeat(x, batch_size, axis=0)


def load_train_save(options, adt, pbt, sfx, save_dir):
    if options['load_params'] is True:
        pbt.load_params(sfx)

    if options['save_params'] is True:
        adt.save_params(sfx)

    if options['train'] is True:
        train(adt, pbt, num_epochs=options['num_epochs'],
              sfx=sfx, save_dir=save_dir)

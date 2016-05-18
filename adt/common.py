from ig.util import *
from ig.io import *
from train import *
import theano.tensor as T
from templates import *

def gen_sfx_key(keys, options):
    sfx_dict = {}
    for key in keys:
        sfx_dict[key] = options[key]
    sfx = stringy_dict(sfx_dict)
    print("sfx:", sfx)
    return sfx


def repeat_to_batch(x, batch_size, tnp=T):
    return tnp.repeat(x, batch_size, axis=0)


def parse_template(template):
    if template == 'res_net':
        return res_net
    elif template == 'conv_net':
        return conv_res_net
    else:
        print("Invalid Template ", template)
        raise ValueError


def load_train_save(options, adt, pbt, sfx, save_dir):
    options_path = os.path.join(save_dir, "options")
    save_dict_csv(options, options_path)

    if options['load_params'] is True:
        adt.load_params(options['params_file'])

    if options['save_params'] is True:
        path = os.path.join(save_dir, "final" + sfx)
        adt.save_params(path)

    if options['train'] is True:
        train(adt, pbt, num_epochs=options['num_epochs'],
              sfx=sfx, save_dir=save_dir, save_every=options['save_every'])

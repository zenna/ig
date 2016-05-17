from ig.util import *

def gen_sfx_key(keys, options):
    sfx_dict = {}
    for key in keys:
        sfx_dict[key] = options[key]
    sfx = stringy_dict(sfx_dict)
    print("sfx:", sfx)
    return sfx


def load_train_save(options, funcs, train_pbt, sfx):
    if options['load_params'] is True:
        for i in range(len(funcs)):
            funcs[i].load_params_fname("%s_stack_interface_%s.npz" % (sfx, i))
        print("Loaded params")

    if options['train'] is True:
        train_pbt(pbt, num_epochs=options['num_epochs'])

    if options['save_params'] is True:
        for i in range(len(funcs)):
            funcs[i].save_params("%s_stack_interface_%s" % (sfx, i))
        print("saved params")

import theano
from adt import *
from mnist import *
from ig.util import *
from train import *
from common import *

# theano.config.optimizer = 'Non    e'
theano.config.optimizer = 'fast_compile'


def set_adt(train_data, options, set_shape=(1, 28, 28), push_args={},
            pop_args={}, item_shape=(1, 28, 28), batch_size=512, nitems=3):
    # Types
    Set = Type(set_shape)
    Item = Type(item_shape)

    # Interface
    store = Interface([Set, Item], [Set], 'store', **store_args)
    is_in = Interface([Set, Item], [Bool], 'is_in', **is_in_args)
    size = Interface([Set], [Integer], 'size', **size_args)
    # union = Interface([Set, Item], [Set], 'push', **push_args)
    # difference = Interface([Set], [Set, Item], 'pop', **pop_args)
    # subset = Interface([Set, Set], [Boolean], 'pop', **pop_args)

    funcs = [store, is_in]

    # train_outs
    train_outs = []
    gen_to_inputs = identity

    # Consts
    empty_set = Const(Set)
    consts = [empty_set]

    # Vars
    # set1 = ForAllVar(Set)
    items = [ForAllVar(Item) for i in range(nitems)]
    forallvars = items

    axiom1 = Axiom(is_empty(empty_set), (0,))
    axiom2 = Axiom(is_empty(store(set1, item1)), (1,))
    axiom3 = Axiom(size(empty_set), (0,))
    axiom4 = Axiom(is_in(empty_set, item1), (0,))
    item1_in_set1 = is_in(store(set1, item1))
    axiom5 = CondAxiom(i1, i2, item1_in_set1, (1,),
                               item1_in_set1, (is_in(set1, i1)))

    # union axioms
    axiom6 = Axiom(union(empty_set, set2), set2)
    axiom7 = Axiom(union(store(set1, item1), set2),
                   store(union(set1, set2), item1))

    # intersect axioms
    axiom8 = Axiom(intersect(empty_set, set2), empty_set)
    intersect_store = intersect(store(set1,), item1, set2)
    axiom9 = CondAxiom(is_in(T, item1), (1,),
                       intersect_store, store(intersect(set1, set2), item1),
                       intersect_store, interect(set1, set2))

    # Generators
    generators = [infinite_batches(train_data, batch_size, shuffle=True)
                  for i in range(nitems)]
    train_fn, call_fns = compile_fns(funcs, consts, forallvars, axioms,
                                     train_outs, options)
    set_adt = AbstractDataType(funcs, consts, forallvars, axioms, name='set')
    set_pdt = ProbDataType(set_adt, train_fn, call_fns, generators,
                           gen_to_inputs, train_outs)
    return set_adt, set_pdt

def main(argv):
    # Args
    global options
    global test_files, train_files
    global views, outputs, net
    global push, pop
    global X_train
    global adt, pdt
    global save_dir
    global sfx

    cust_options = {}
    cust_options['nitems'] = (int, 3)
    cust_options['width'] = (int, 28)
    cust_options['height'] = (int, 28)
    cust_options['num_epochs'] = (int, 100)
    cust_options['save_every'] = (int, 100)
    cust_options['compile_fns'] = (True,)
    cust_options['save_params'] = (True,)
    cust_options['train'] = (True,)
    cust_options['nblocks'] = (int, 1)
    cust_options['block_size'] = (int, 2)
    cust_options['batch_size'] = (int, 512)
    cust_options['nfilters'] = (int, 24)
    cust_options['layer_width'] = (int, 50)
    cust_options['adt'] = (str, 'set')
    cust_options['template'] = (str, 'res_net')
    options = handle_args(argv, cust_options)

    X_train, y_train, X_val, y_val, X_test, y_test = load_dataset()
    sfx = gen_sfx_key(('adt', 'nblocks', 'block_size', 'nfilters'), options)
    options['template'] = parse_template(options['template'])

    adt, pdt = set_adt(X_train, options, push_args=options,
                         nitems=options['nitems'], pop_args=options,
                         batch_size=options['batch_size'])

    save_dir = mk_dir(sfx)
    load_train_save(options, adt, pdt, sfx, save_dir)
    push, pop = pdt.call_fns
    loss, set, img, new_set, new_img = validate_set_img_rec(new_img, X_train, push, pop, 0, 1)


if __name__ == "__main__":
    main(sys.argv[1:])

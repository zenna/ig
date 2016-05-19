import theano
from adt import *
from mnist import *
from ig.util import *
from train import *
from common import *

# theano.config.optimizer = 'Non    e'
theano.config.optimizer = 'fast_compile'


def record_adt(train_data, options, record_shape=(1, 28, 28),
               field_shape=(1, 28, 28), record_shape=(1, 28, 28), find_args={},
               store_args={}, batch_size=512, nitems=3):
    # Types
    Record = Type(record_shape)
    Item = Type(item_shape)
    Field = Type(field_shape)

    # Interface
    store = Interface([Record, FieldItem, InfoItem], [Record], 'store',
                      **store_args)
    find = Interface([Record, FieldItem], [InfoItem], 'find', **find_args)
    funcs = [store, find]

    # Consts
    empty_record = Const(Record)
    consts = [empty_record]

    # train_outs
    train_outs = []
    gen_to_inputs = identity

    # Vars
    # record1 = ForAllVar(Record)
    item1 = ForAllVar(Item)
    field1 = ForAllVar(Field)
    field2 = ForAllVar(Field)
    record1 = ForAllVar(Record)
    forallvars = [item1, field1, field2, record1]

    find_it = find(store(record1, field2, item1), field1)
    axiom1 = CondAxiom(f1, f2, find_it, item1, find_it, find(record1, field1))

    # Generators
    generators = [infinite_batches(train_data, batch_size, shuffle=True)
                  for i in range(nitems)]
    train_fn, call_fns = compile_fns(funcs, consts, forallvars, axioms,
                                     train_outs, options)
    record_adt = AbstractDataType(funcs, consts, forallvars, axioms, name='record')
    record_pdt = ProbDataType(record_adt, train_fn, call_fns, generators,
                              gen_to_inputs, train_outs)
    return record_adt, record_pdt


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
    cust_options['adt'] = (str, 'record')
    cust_options['template'] = (str, 'res_net')
    options = handle_args(argv, cust_options)

    X_train, y_train, X_val, y_val, X_test, y_test = load_datarecord()
    sfx = gen_sfx_key(('adt', 'nblocks', 'block_size', 'nfilters'), options)
    options['template'] = parse_template(options['template'])

    adt, pdt = record_adt(X_train, options, push_args=options,
                         nitems=options['nitems'], pop_args=options,
                         batch_size=options['batch_size'])

    save_dir = mk_dir(sfx)
    load_train_save(options, adt, pdt, sfx, save_dir)
    push, pop = pdt.call_fns
    loss, record, img, new_record, new_img = validate_record_img_rec(new_img, X_train, push, pop, 0, 1)


if __name__ == "__main__":
    main(sys.argv[1:])

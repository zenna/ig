import theano
from adt import *
from mnist import *
from ig.util import *
from train import *
from common import *

# theano.config.optimizer = 'Non    e'
theano.config.optimizer = 'fast_compile'


def stack_adt(train_data, options, stack_shape=(1, 28, 28), push_args={},
              pop_args={}, item_shape=(1, 28, 28), batch_size=512, nitems=3):
    # Types
    Stack = Type(stack_shape)
    Item = Type(item_shape)

    # Interface
    push = Interface([Stack, Item], [Stack], **push_args)
    pop = Interface([Stack], [Stack, Item], conv_res_net, **pop_args)
    funcs = [push, pop]

    # train_outs
    train_outs = []
    gen_to_inputs = identity

    # Consts
    empty_stack = Const(Stack)
    consts = [empty_stack]

    # Vars
    # stack1 = ForAllVar(Stack)
    items = [ForAllVar(Item) for i in range(nitems)]
    forallvars = items

    # Axioms
    axioms = []
    batch_empty_stack = repeat_to_batch(empty_stack.input_var, batch_size)
    stack = batch_empty_stack
    for i in range(nitems):
        (stack,) = push(stack, items[i].input_var)
        pop_stack = stack
        for j in range(i, -1, -1):
            (pop_stack, pop_item) = pop(pop_stack)
            axiom = Axiom((pop_item,), (items[j].input_var,))
            axioms.append(axiom)

    # Generators
    generators = [infinite_batches(train_data, batch_size, shuffle=True)
                  for i in range(nitems)]
    train_fn, call_fns = compile_fns(funcs, consts, forallvars, axioms,
                                     train_outs, options)
    stack_adt = AbstractDataType(funcs, consts, forallvars, axioms,
                                 name='stack')
    stack_pdt = ProbDataType(stack_adt, train_fn, call_fns, generators,
                             gen_to_inputs, train_outs)
    return stack_adt, stack_pdt


# Validation
def validate_what(data, batch_size, nitems, es, push, pop):
    datalen = data.shape[0]
    es = np.repeat(es, batch_size, axis=0)
    data_indcs = np.random.randint(0, datalen-batch_size, nitems)
    items = [data[data_indcs[i]:data_indcs[i]+batch_size]
             for i in range(nitems)]
    losses = []
    stack = es
    for i in range(nitems):
        (stack,) = push(stack, items[i])
        pop_stack = stack
        for j in range(i, -1, -1):
            (pop_stack, pop_item) = pop(pop_stack)
            loss = mse(pop_item, items[j], tnp=np)
            losses.append(loss)
    print(losses)

def whitenoise_trick():
    new_img = floatX(np.array(np.random.rand(1,1,28,28)*2**8, dtype='int'))/256
    for i in range(1000):
        loss, stack, img, new_stack, new_img = validate_stack(new_img, X_train, push, pop, 0, 512)

def stack_unstack(n, stack, offset=0):
    lb = 0 + offset
    ub = 1 + offset
    imgs = []
    stacks = []
    stacks.append(stack)
    for i in range(n):
        new_img = floatX(X_train[lb+i:ub+i])
        imgs.append(new_img)
        (stack,) = push(stack,new_img)
        stacks.append(stack)

    for i in range(n):
        (stack, old_img) = pop(stack)
        stacks.append(stack)
        imgs.append(old_img)

    return stacks + imgs

def whitenoise(batch_size):
    return floatX(np.array(np.random.rand(batch_size,1,28,28)*2**8, dtype='int'))/256


def main(argv):
    # Args
    global options
    global test_files, train_files
    global views, outputs, net
    global push, pop
    global X_train
    global adt, pdt

    cust_options = {}
    cust_options['nitems'] = (int, 3)
    cust_options['width'] = (int, 28)
    cust_options['height'] = (int 28)
    cust_options['num_epochs'] = (int, 100)
    cust_options['compile_fns'] = (True,)
    cust_options['save_params'] = (True,)
    cust_options['train'] = (True,)
    cust_options['nblocks'] = (int, 1)
    cust_options['block_size'] = (int, 2)
    cust_options['batch_size'] = (int, 512)
    cust_options['nfilters'] = (int, 24)
    cust_options['layer_width'] = (int, 50)
    cust_options['adt'] = (str, 'stack')
    cust_options['template'] = (str, 'res_net')
    options = handle_args(argv, cust_options)

    X_train, y_train, X_val, y_val, X_test, y_test = load_dataset()
    sfx = gen_sfx_key(('adt', 'nblocks', 'block_size', 'nfilters'), options)
    print(options)
    adt, pdt = stack_adt(X_train, options, push_args=options,
                         nitems=options['nitems'], pop_args=options,
                         batch_size=options['batch_size'])

    save_dir = mk_dir(sfx)
    load_train_save(options, adt, pdt, sfx, save_dir)
    push, pop = pdt.call_fns
    loss, stack, img, new_stack, new_img = validate_stack_img_rec(new_img, X_train, push, pop, 0, 1)


if __name__ == "__main__":
    main(sys.argv[1:])

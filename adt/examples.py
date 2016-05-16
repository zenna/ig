## A stack
import theano
theano.config.optimizer = 'fast_compile'
theano.config.optimizer = 'None'
from adt import *
from mnist import *
from ig.util import *


def stack_example_conv(train_data, options, stack_shape = (1,28,28),
        push_args = {}, pop_args = {}, item_shape = (1,28,28), batch_size = 512):
    # Types
    Stack = Type(stack_shape)
    Item = Type(item_shape)

    # Interface
    push = Interface([Stack, Item],[Stack], conv_res_net, width=28, height = 28,
        **push_args)
    pop = Interface([Stack],[Stack, Item], conv_res_net, width=28, height = 28,
        **pop_args)
    interfaces = [push, pop]

    # Constants
    empty_stack = Constant(Stack)
    constants = [empty_stack]

    # Vars
    # stack1 = ForAllVar(Stack)
    item1 = ForAllVar(Item)

    # Generators
    generators = [infinite_minibatches(train_data, batch_size, True)]
    # generators = [infinite_samples(np.random.rand, batch_size, stack_shape),
    #               infinite_minibatches(train_data, batch_size, True)]
    forallvars = [item1]
    # forallvars = [stack1, item1]

    # Axioms
    es = T.repeat(empty_stack.input_var, batch_size, axis=0)
    (pushed_stack,) = push(es, item1.input_var)
    # (pushed_stack,) = push(stack1.input_var, item1.input_var)
    (popped_stack, popped_item) = pop(pushed_stack)

    axiom1 = Axiom((popped_stack, popped_item), (empty_stack.input_var, item1.input_var))
    # axiom1 = Axiom((popped_stack, popped_item), (stack1.input_var, item1.input_var))
    # axiom2 = BoundAxiom(pushed_stack)
    # axiom3 = BoundAxiom(popped_stack)
    axioms = [axiom1]
    # axioms = [axiom1, axiom2, axiom3]
    train_fn, call_fns = compile_fns(interfaces, constants, forallvars, axioms, options)
    return interfaces, constants, forallvars, axioms, generators, train_fn, call_fns

def stack_example_conv_rec_lord(train_data, options, stack_shape = (1,28,28),
        push_args = {}, pop_args = {}, item_shape = (1,28,28), batch_size = 512):
    # Types
    Stack = Type(stack_shape)
    Item = Type(item_shape)

    # Interface
    push = Interface([Stack, Item],[Stack], conv_res_net, width=28, height = 28,
        **push_args)
    pop = Interface([Stack],[Stack, Item], conv_res_net, width=28, height = 28,
        **pop_args)
    interfaces = [push, pop]

    # Constants
    empty_stack = Constant(Stack)
    constants = [empty_stack]

    # Vars
    # stack1 = ForAllVar(Stack)
    nitems = 3
    items = [ForAllVar(Item) for i in range(nitems)]
    axioms = []

    batch_empty_stack = T.repeat(empty_stack.input_var, batch_size, axis=0)
    stack = batch_empty_stack
    for i in range(nitems):
        (stack,) = push(stack, items[i].input_var)
        pop_stack = stack
        for j in range(i,-1,-1):
            (pop_stack, pop_item) = pop(pop_stack)
            axiom = Axiom((pop_item,), (items[j].input_var,))
            axioms.append(axiom)

    # Generators
    generators = [infinite_minibatches(train_data, batch_size, True) for i in range(nitems)]
    forallvars = items
    train_fn, call_fns = compile_fns(interfaces, constants, forallvars, axioms, options)
    return interfaces, constants, forallvars, axioms, generators, train_fn, call_fns


def stack_example_conv_rec(train_data, options, stack_shape = (1,28,28),
        push_args = {}, pop_args = {}, item_shape = (1,28,28), batch_size = 512):
    # Types
    Stack = Type(stack_shape)
    Item = Type(item_shape)

    # Interface
    push = Interface([Stack, Item],[Stack], conv_res_net, width=28, height = 28,
        **push_args)
    pop = Interface([Stack],[Stack, Item], conv_res_net, width=28, height = 28,
        **pop_args)
    interfaces = [push, pop]

    # Constants
    empty_stack = Constant(Stack)
    constants = [empty_stack]


    # Vars
    # stack1 = ForAllVar(Stack)
    item1 = ForAllVar(Item)
    item2 = ForAllVar(Item)
    item3 = ForAllVar(Item)

    # Generators
    generators = [infinite_minibatches(train_data, batch_size, True),
                  infinite_minibatches(train_data, batch_size, True),
                  infinite_minibatches(train_data, batch_size, True)]
    forallvars = [item1, item2, item3]

    # Axioms
    es = T.repeat(empty_stack.input_var, batch_size, axis=0)
    (pushed_stack,) = push(es, item1.input_var)
    (popped_stack, popped_item) = pop(pushed_stack)
    axiom1 = Axiom((popped_stack, popped_item), (es, item1.input_var))

    (p_pushed_stack,) = push(pushed_stack, item2.input_var)
    (p_popped_stack, p_popped_item) = pop(p_pushed_stack)
    axiom2 = Axiom((p_popped_stack, p_popped_item), (pushed_stack, item2.input_var))

    (p_p_pushed_stack,) = push(p_pushed_stack, item3.input_var)
    (p_p_popped_stack, p_p_popped_item) = pop(p_p_pushed_stack)
    axiom3 = Axiom((p_p_popped_stack, p_p_popped_item), (p_pushed_stack, item3.input_var))

    # baxiom1 = BoundAxiom(pushed_stack)
    # baxiom2 = BoundAxiom(popped_stack)
    # axioms = [axiom1]
    axioms = [axiom1, axiom2, axiom3]
    train_fn, call_fns = compile_fns(interfaces, constants, forallvars, axioms, options)
    return interfaces, constants, forallvars, axioms, generators, train_fn, call_fns


## Validation
## =========

def rand_int(n):
    return int(np.random.rand() * n)

def validate_what(data, batch_size, nitems, es, push, pop):
    datalen = data.shape[0]
    es = np.repeat(es,batch_size,axis=0)
    data_indices = [rand_int(datalen-batch_size) for i in range(nitems)]
    items = [data[data_indices[i]:data_indices[i]+batch_size] for i in range(nitems)]
    losses = []
    stack = es
    for i in range(nitems):
        (stack,) = push(stack, items[i])
        pop_stack = stack
        for j in range(i,-1,-1):
            (pop_stack, pop_item) = pop(pop_stack)
            loss = mse(pop_item, items[j], tnp = np)
            losses.append(loss)
    print(losses)

def validate_3stack_img(item1, item2, item3, es, push, pop, lb, ub):
    (pushed_stack,) = push(es, item1)
    (popped_stack, popped_item) = pop(pushed_stack)
    loss1 = mse(popped_stack, es, tnp = np)
    loss2 = mse(popped_item, item1, tnp = np)

    (p_pushed_stack,) = push(pushed_stack, item2)
    (p_popped_stack, p_popped_item) = pop(p_pushed_stack)
    loss3 = mse(p_popped_stack, pushed_stack, tnp = np)
    loss4 = mse(p_popped_item, item2, tnp = np)

    (p_p_pushed_stack,) = push(p_pushed_stack, item3)
    (p_p_popped_stack, p_p_popped_item) = pop(p_p_pushed_stack)
    loss5 = mse(p_p_popped_stack, p_pushed_stack, tnp = np)
    loss6 = mse(p_p_popped_item, item3, tnp = np)

    loss = [loss1, loss2, loss3, loss4, loss5, loss6]
    print(loss)
    stacks =  [pushed_stack, popped_stack, p_pushed_stack, p_popped_stack, p_p_pushed_stack, p_p_popped_stack]
    items = [item1, item2, item3, popped_item, p_popped_item, p_p_popped_item]
    return stacks, items


def validate_stack_img(imgbatch, stackbatch, push, pop, lb, ub):
    (new_stack,) = push(stackbatch,imgbatch)
    (old_new_stack, img) = pop(new_stack)
    loss1 = mse(old_new_stack, stackbatch, tnp = np)
    loss2 = mse(img, imgbatch, tnp = np)
    loss = [loss1, loss2]
    print(loss)
    return loss, stackbatch, imgbatch, old_new_stack, img

def validate_stack(data, push, pop, lb, ub):
    imgbatch = floatX(data[lb:ub])
    return validate_stack_img(imgbatch, push, pop, lb, ub)

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
    ## Args
    global options
    global test_files, train_files
    global net, output_layer, cost_f, cost_f_dict, val_f, call_f, call_f_dict
    global views, outputs, net
    global interfaces, constants, forallvars, axioms, generators, train_fn, call_fns
    global push, pop

    options = handle_args(argv)
    options['num_epochs'] = 50
    options['compile_fns'] = True
    options['save_params'] = True
    options['train'] = True
    options['nblocks'] = 1
    options['block_size'] = 2
    options['batch_size'] = 512
    options['nfilters'] = 24
    func_args = {'nblocks' : options['nblocks'], 'block_size' : options['block_size']}

    sfx_dict = {}
    for key in ('nblocks', 'block_size', 'nfilters'):
        sfx_dict[key] = options[key]
    sfx = stringy_dict(sfx_dict)
    print("sfx:", sfx)

    print(options)
    global X_train
    X_train, y_train, X_val, y_val, X_test, y_test = load_dataset()
    interfaces, constants, forallvars, axioms, generators, train_fn, call_fns = stack_example_conv_rec_lord(
        X_train, options, push_args = options,
        pop_args = options, batch_size = options['batch_size'])
    push, pop = call_fns

    if options['load_params'] == True:
        for i in range(len(interfaces)):
            interfaces[i].load_params_fname("%s_stack_interface_%s.npz" % (sfx, i))
        print("Loaded params")

    if options['train'] == True:
        train(train_fn, generators, num_epochs = options['num_epochs'])

    if options['save_params'] == True:
        for i in range(len(interfaces)):
            interfaces[i].save_params("%s_stack_interface_%s" % (sfx, i))
        print("saved params")

    loss, stack, img, new_stack, new_img = validate_stack_img_rec(new_img, X_train, push, pop, 0, 1)




if __name__ == "__main__":
   main(sys.argv[1:])

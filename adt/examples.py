## A stack
import theano
theano.config.optimizer = 'fast_compile'
theano.config.optimizer = 'None'
from adt import *
from mnist import *

def stack_example(train_data, options, stack_shape = (100,), item_shape = (28*28,), batch_sizes = (256, 256)):
    Stack = Type(stack_shape)
    Item = Type(item_shape)
    push = Interface([Stack, Item],[Stack], res_net, layer_width=100)
    pop = Interface([Stack],[Stack, Item], res_net, layer_width=884)
    stack1 = ForAllVar(Stack)
    item1 = ForAllVar(Item)
    global axiom1
    generators = [infinite_samples(np.random.rand, batch_sizes[0], stack_shape),
                  infinite_minibatches(train_data, batch_sizes[1], True)]

    # Axioms
    (pushed_stack,) = push(stack1.input_var, item1.input_var)
    (popped_stack, popped_item) = pop(pushed_stack)

    axiom1 = Axiom((popped_stack, popped_item), (stack1.input_var, item1.input_var))
    axiom2 = BoundAxiom(pushed_stack)
    axiom3 = BoundAxiom(popped_stack)
    axioms = [axiom1, axiom2, axiom3]
    train_fn, call_fns = compile_fns([push, pop], [stack1, item1], axioms, options)
    train(train_fn, generators)

def stack_example_conv(train_data, options, stack_shape = (1,28,28), item_shape = (1,28,28), batch_sizes = (512, 512)):
    # Types
    Stack = Type(stack_shape)
    Item = Type(item_shape)

    # Interface
    push = Interface([Stack, Item],[Stack], conv_res_net, width=28, height = 28, nblocks = 5, block_size = 2)
    pop = Interface([Stack],[Stack, Item], conv_res_net, width=28, height = 28, nblocks = 5, block_size = 2)
    interfaces = [push, pop]

    # Vars
    stack1 = ForAllVar(Stack)
    item1 = ForAllVar(Item)

    # Generators
    generators = [infinite_samples(np.random.rand, batch_sizes[0], stack_shape),
                  infinite_minibatches(train_data, batch_sizes[1], True)]
    forallvars = [stack1, item1]

    # Axioms
    (pushed_stack,) = push(stack1.input_var, item1.input_var)
    (popped_stack, popped_item) = pop(pushed_stack)

    axiom1 = Axiom((popped_stack, popped_item), (stack1.input_var, item1.input_var))
    axiom2 = BoundAxiom(pushed_stack)
    axiom3 = BoundAxiom(popped_stack)
    axioms = [axiom1, axiom2, axiom3]
    train_fn, call_fns = compile_fns(interfaces, forallvars, axioms, options)
    return interfaces, forallvars, axioms, generators, train_fn, call_fns
#
# def scalar_field_example(options, field_shape = (100,),batch_size=512):
#     Field = Type(field_shape)
#     Point = Type((3,))
#     Scalar = Type((1,))
#
#     s = Interface([Field, Point], [Scalar], res_net, layer_width = 100)
#     union = Interface([Field, Field], [Field], res_net, layer_width = 100)
#     intersection = Interface([Field, Field], [Field], res_net, layer_width = 100)
#     interfaces = [s, union, intersection]
#
#     field1 = ForAllVar(Field)
#     field2 = ForAllVar(Field)
#     point1 = ForAllVar(Point)
#     generators = [infinite_samples(np.random.rand, batch_size, field_shape),
#                   infinite_samples(np.random.rand, batch_size, field_shape),
#                   infinite_samples(np.random.rand, batch_size, (3,))]
#     global forallvars
#     forallvars = [field1, field2, point1]
#     # Boolean structure on scalar field
#     # ForAll p in R3, (f1 union f2)(p) = f1(p) f2(p)
#     axiom1 = Axiom(s(*(union(field1, field2) + [point1])), [s(field1, point1)[0] * s(field2, point1)[0]])
#     axiom2 = Axiom(s(*(intersection(field1, field2) + [point1])), [s(field1, point1)[0] + s(field2, point1)[0]])
#     axioms = [axiom1, axiom2]
#     train_fn, call_fns = compile_fns(interfaces, forallvars, axioms, options)
#     train(train_fn, generators)
# #
# def binary_tree(train_data, binary_tree_shape = (500,), item_shape = (28*28,),  batch_size = 256):
#     BinTree = Type(binary_tree_shape)
#     Item = Type(item_shape)
#     make = Interface([BinTree, Item, BinTree],[BinTree], res_net, layer_width=500)
#     left_tree = Interface([BinTree], [BinTree], res_net, layer_width=500)
#     right_tree = Interface([BinTree], [BinTree], res_net, layer_width=500)
#     get_item = Interface([BinTree], [Item], res_net, layer_width=500)
#     # is_empty = Interface([BinTree], [BoolType])
#
#     bintree1 = ForAllVar(BinTree)
#     bintree2 = ForAllVar(BinTree)
#     item1 = ForAllVar(Item)
#     # error = Constant(np.random.rand(item_shape))
#
#     # axiom1 = Axiom(left_tree(create), error)
#     make_stuff = make(bintree1.input_var, item1.input_var, bintree2.input_var)
#     axiom2 = Axiom(left_tree(*make_stuff), (bintree1.input_var,))
#     # axiom3 = Axiom(right_tree(create), error)
#     axiom4 = Axiom(right_tree(*make_stuff), (bintree2.input_var,))
#     # axiom5 = Axiom(item(create), error) # FIXME, how to handle True
#     axiom6 = Axiom(get_item(*make_stuff), (item1.input_var,))
#     # axiom7 = Axiom(is_empty(create), True)
#     # axiom8 = Axiom(is_empty(make(bintree1.input_var, item1, bintree2)), False)
#     interfaces = [make, left_tree, right_tree, get_item]
#     # axioms = [axiom1, axiom2, axiom3, axiom4, axiom5, axiom6, axiom6, axiom7. axiom8]
#     axioms = [axiom2, axiom4, axiom6]
#     forallvars = [bintree1, bintree2, item1]
#     generators = [infinite_samples(np.random.rand, batch_size, binary_tree_shape),
#                  infinite_samples(np.random.rand, batch_size, binary_tree_shape),
#                  infinite_minibatches(train_data, batch_size, True)]
#     train_fn, call_fns = compile_fns(interfaces, forallvars, axioms, options)
#     train(train_fn, generators)

def validate_stack(X_train, push, pop, lb, ub):
    imgbatch = floatX(X_train[lb:ub])
    stackbatch = floatX(np.random.rand(ub-lb,1,28,28))
    (new_stack,) = push(stackbatch,imgbatch)
    (old_new_stack, img) = pop(new_stack)
    print(mse(img, imgbatch, tnp = np))

def main(argv):
    ## Args
    global options
    global test_files, train_files
    global net, output_layer, cost_f, cost_f_dict, val_f, call_f, call_f_dict
    global views, outputs, net
    global interfaces, forallvars, axioms, train_fn, call_fns

    options = handle_args(argv)
    options['num_epochs'] = 10
    options['compile_fns'] = True
    options['load_params'] = True
    options['save_params'] = True
    options['train'] = True

    print(options)

    X_train, y_train, X_val, y_val, X_test, y_test = load_dataset()
    interfaces, forallvars, axioms, generators, train_fn, call_fns = stack_example_conv(X_train, options)
    push, pop = call_fns

    if options['load_params'] == True:
        for i in range(len(interfaces)):
            interfaces[i].load_params("stack_interface_%s" % i)

    if options['train'] == True:
        train(train_fn, generators, num_epochs = options['num_epochs'])

    if options['save_params'] == True:
        for i in range(len(interfaces)):
            interfaces[i].save_params("stack_interface_%s" % i)

    validate_stack(X_train, push, pop, 0, 512)


    # stack_example_conv(X_train, options)
    # binary_tree(X_train.reshape(50000,28*28))
    # scalar_field_example()


# def associative_array(keyed_table_shape = (100,)):
#     # Types
#     KeyedTable = Type(keyed_table_shape)
#     Key = Type(key_shape)
#     Value = Type(val_shape)
#     # Interface
#     update = Interface([KeyedTable, Key, Value], [KeyedTable])
#     delete = Interface([KeyedTable Key], [KeyedTable])
#     find = Interface([KeyedTable, Key], [Value])
#     # is_in = Interface([KeyedTable Key], [BoolType])
#     # is_empty = Interface([KeyedTable], [BoolType])
#     # interface = [store, delete, find, is_in, is_empty]
#     interface = [update, delete, find]
#
#     # Variables
#     item1 = ForAllVar(Value)
#     key1 = ForAllVar(Key)
#     key2 = ForAllVar(Key)
#     kt1 = ForAllVar(KeyedTable)
#
#     axiom1 = find(update(kt1, )))
#
#
#     Item = Type(item_shape)
#     make = Interface([BinTree, Item, BinTree],[BinTree], res_net, layer_width=500)
#     left_tree = Interface([BinTree], [BinTree], res_net, layer_width=500)
#     right_tree = Interface([BinTree], [BinTree], res_net, layer_width=500)
#     get_item = Interface([BinTree], [Item], res_net, layer_width=500)
# #
# # def hierarhical_concept():
# #     ...
#
# def hierarhical_concept():
#     a = 3
#
# def turing_machine(state_shape=(10,), Symbol(1,)):
#     State = Type(state_shape)
#     Symbol = Type(symbol)
#
#     Q_s = Constant(0)


if __name__ == "__main__":
   main(sys.argv[1:])

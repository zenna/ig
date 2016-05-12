## A stack
import theano
theano.config.optimizer = 'fast_compile'
theano.config.optimizer = 'None'
from adt import *

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
    Stack = Type(stack_shape)
    Item = Type(item_shape)
    push = Interface([Stack, Item],[Stack], conv_res_net, width=28, height = 28)
    pop = Interface([Stack],[Stack, Item], conv_res_net, width=28, height = 28)
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

def scalar_field_example(options, field_shape = (100,),batch_size=512):
    Field = Type(field_shape)
    Point = Type((3,))
    Scalar = Type((1,))

    s = Interface([Field, Point], [Scalar], res_net, layer_width = 100)
    union = Interface([Field, Field], [Field], res_net, layer_width = 100)
    intersection = Interface([Field, Field], [Field], res_net, layer_width = 100)
    interfaces = [s, union, intersection]

    field1 = ForAllVar(Field)
    field2 = ForAllVar(Field)
    point1 = ForAllVar(Point)
    generators = [infinite_samples(np.random.rand, batch_size, field_shape),
                  infinite_samples(np.random.rand, batch_size, field_shape),
                  infinite_samples(np.random.rand, batch_size, (3,))]
    global forallvars
    forallvars = [field1, field2, point1]
    # Boolean structure on scalar field
    # ForAll p in R3, (f1 union f2)(p) = f1(p) f2(p)
    axiom1 = Axiom(s(*(union(field1, field2) + [point1])), [s(field1, point1)[0] * s(field2, point1)[0]])
    axiom2 = Axiom(s(*(intersection(field1, field2) + [point1])), [s(field1, point1)[0] + s(field2, point1)[0]])
    axioms = [axiom1, axiom2]
    train_fn, call_fns = compile_fns(interfaces, forallvars, axioms, options)
    train(train_fn, generators)

def binary_tree(train_data, binary_tree_shape = (500,), item_shape = (28*28,),  batch_size = 256):
    BinTree = Type(binary_tree_shape)
    Item = Type(item_shape)
    make = Interface([BinTree, Item, BinTree],[BinTree], res_net, layer_width=500)
    left_tree = Interface([BinTree], [BinTree], res_net, layer_width=500)
    right_tree = Interface([BinTree], [BinTree], res_net, layer_width=500)
    get_item = Interface([BinTree], [Item], res_net, layer_width=500)
    # is_empty = Interface([BinTree], [BoolType])

    bintree1 = ForAllVar(BinTree)
    bintree2 = ForAllVar(BinTree)
    item1 = ForAllVar(Item)
    # error = Constant(np.random.rand(item_shape))

    # axiom1 = Axiom(left_tree(create), error)
    make_stuff = make(bintree1.input_var, item1.input_var, bintree2.input_var)
    axiom2 = Axiom(left_tree(*make_stuff), (bintree1.input_var,))
    # axiom3 = Axiom(right_tree(create), error)
    axiom4 = Axiom(right_tree(*make_stuff), (bintree2.input_var,))
    # axiom5 = Axiom(item(create), error) # FIXME, how to handle True
    axiom6 = Axiom(get_item(*make_stuff), (item1.input_var,))
    # axiom7 = Axiom(is_empty(create), True)
    # axiom8 = Axiom(is_empty(make(bintree1.input_var, item1, bintree2)), False)
    interfaces = [make, left_tree, right_tree, get_item]
    # axioms = [axiom1, axiom2, axiom3, axiom4, axiom5, axiom6, axiom6, axiom7. axiom8]
    axioms = [axiom2, axiom4, axiom6]
    forallvars = [bintree1, bintree2, item1]
    generators = [infinite_samples(np.random.rand, batch_size, binary_tree_shape),
                 infinite_samples(np.random.rand, batch_size, binary_tree_shape),
                 infinite_minibatches(train_data, batch_size, True)]
    train_fn, call_fns = compile_fns(interfaces, forallvars, axioms, options)
    train(train_fn, generators)

def main(argv):
    ## Args
    global options
    global test_files, train_files
    global net, output_layer, cost_f, cost_f_dict, val_f, call_f, call_f_dict
    global views, shape_params, outputs, net

    options = handle_args(argv)
    nepochs = options['nepochs'] = 10000
    options['compile_fns'] = False

    print(options)

    X_train, y_train, X_val, y_val, X_test, y_test = load_dataset()
    # stack_example_conv(X_train.reshape(50000,28*28), options)
    stack_example_conv(X_train, options)
    # binary_tree(X_train.reshape(50000,28*28))
    # scalar_field_example()


if __name__ == "__main__":
   main(sys.argv[1:])

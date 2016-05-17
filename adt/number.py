import theano
from ig.util import *
import ig.io
from adt import *
from train import *
from common import *

# theano.config.optimizer = 'None'
theano.config.optimizer = 'fast_compile'


def number_adt(options, niters=3, number_shape=(5,), batch_size=64,
               succ_args={}, add_args={}, mul_args={}):
    # Types
    Number = Type(number_shape)

    # Interface
    succ = Interface([Number], [Number], res_net, **succ_args)
    add = Interface([Number, Number], [Number], res_net, **add_args)
    mul = Interface([Number, Number], [Number], res_net, **mul_args)
    funcs = [succ, add, mul]

    # Vars
    a = ForAllVar(Number)
    b = ForAllVar(Number)
    forallvars = [a, b]

    # Constants
    zero = Constant(Number)
    consts = [zero]

    # axioms
    succ_b = succ(b)
    add_a_zero = add(a, zero)
    mul_a_succ_b = mul(a, *succ_b)
    mul_axiom2_rhs = mul(a, b) + [a.input_var]

    add_axiom1 = Axiom(add(a, zero), (a.input_var,))
    add_axiom2 = Axiom(add(a, *succ_b), succ(*add(a, b)))
    mul_axiom1 = Axiom(mul(a, zero), (zero.input_var,))
    mul_axiom2 = Axiom(mul(a, *succ_b), add(*mul_axiom2_rhs))
    axioms = [add_axiom1, add_axiom2, mul_axiom1, mul_axiom2]

    # generators
    generators = [infinite_samples(np.random.rand, batch_size, number_shape)
                  for i in range(2)]

    train_outs = []
    gen_to_inputs = identity

    train_fn, call_fns = compile_fns(funcs, consts, forallvars, axioms,
                                     train_outs, options)
    number_adt = AbstractDataType(funcs, consts, forallvars, axioms,
                                  name='natural number')
    number_pdt = ProbDataType(number_adt, train_fn, call_fns,
                              generators, gen_to_inputs, train_outs)
    return number_adt, number_pdt


def main(argv):
    # Args
    global options
    global test_files, train_files
    global net, output_layer, cost_f, cost_f_dict, val_f, call_f, call_f_dict
    global views, outputs, net
    global funcs, consts, forallvars, axioms, generators, train_fn, call_fns
    global push, pop
    global X_train
    global adt, pbt

    options = handle_args(argv)
    options['num_epochs'] = 100
    options['compile_fns'] = True
    options['save_params'] = True
    options['train'] = True
    options['nblocks'] = 1
    options['block_size'] = 1
    options['batch_size'] = 256
    options['nfilters'] = 24
    options['layer_width'] = 101
    options['adt'] = 'number'

    sfx = gen_sfx_key(('adt', 'nblocks', 'block_size', 'nfilters'), options)
    adt, pbt = number_adt(options,
                          number_shape=(102,),
                          succ_args=options, add_args=options,
                          mul_args=options, batch_size=options['batch_size'])

    load_train_save(options, funcs, train_pbt, sfx)

if __name__ == "__main__":
    main(sys.argv[1:])

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
    BinInteger = Type((1,))  # A python integer

    # Interface
    succ = Interface([Number], [Number], res_net, **succ_args)
    add = Interface([Number, Number], [Number], res_net, **add_args)
    mul = Interface([Number, Number], [Number], res_net, **mul_args)
    encode = Interface([BinInteger], [Number], res_net, **mul_args)
    decode = Interface([Number], [BinInteger], res_net, **mul_args)
    funcs = [succ, add, mul]

    # Vars
    # a = ForAllVar(Number)
    # b = ForAllVar(Number)
    bi = ForAllVar(BinInteger)
    bj = ForAllVar(BinInteger)

    forallvars = [bi, bj]

    # Constants
    zero = Constant(Number)
    zero_batch = repeat_to_batch(zero.input_var, batch_size)
    consts = [zero]

    # axioms
    (encoded1,) = encode(bi)
    (encoded2,) = encode(bj)

    # axiom_zero = Axiom(decode(zero_batch), (0,))

    axiom_ed = Axiom(decode(encoded1), (bi.input_var,))
    (succ_encoded,) = succ(encoded1)
    axiom_succ_ed = Axiom(decode(succ_encoded), (bi.input_var + 1,))

    encode_axioms = [axiom_ed, axiom_succ_ed]

    a = encoded1
    b = encoded2

    (succ_b,) = succ(b)
    mul_a_succ_b = mul(a, succ_b)
    mul_axiom2_rhs = mul(a, b) + [a]

    add_axiom1 = Axiom(add(a, zero_batch), (a,))
    add_axiom2 = Axiom(add(a, succ_b), succ(*add(a, b)))
    mul_axiom1 = Axiom(mul(a, zero_batch), (zero_batch,))
    mul_axiom2 = Axiom(mul(a, succ_b), add(*mul_axiom2_rhs))
    arith_axioms = [add_axiom1, add_axiom2, mul_axiom1, mul_axiom2]
    axioms = encode_axioms + arith_axioms

    # generators
    def realistic_nums(*shape):
        return np.random.randint(0, 10, shape)
        # return floatX(np.random.zipf(1.7, shape) +
        #               np.random.randint(-1, 10, shape))

    generators = [infinite_samples(realistic_nums, batch_size, (1,))
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
    global adt, pdt

    options = handle_args(argv)
    options['num_epochs'] = 100
    options['compile_fns'] = True
    options['save_params'] = True
    options['train'] = True
    options['nblocks'] = 5
    options['block_size'] = 2
    options['batch_size'] = 1024
    options['nfilters'] = 24
    options['layer_width'] = 50
    options['adt'] = 'number'

    sfx = gen_sfx_key(('adt', 'nblocks', 'block_size', 'nfilters'), options)
    adt, pdt = number_adt(options,
                          number_shape=(10,),
                          succ_args=options, add_args=options,
                          mul_args=options, batch_size=options['batch_size'])

    load_train_save(options, adt.funcs, pdt, sfx)

if __name__ == "__main__":
    main(sys.argv[1:])

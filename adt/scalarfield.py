import theano
theano.config.optimizer = 'fast_compile'
# theano.config.optimizer = 'None'

from ig.util import *
import ig.io

from adt import *

def repeat_to_batch(x, batch_size, tnp = T):
    return tnp.repeat(x, batch_size, axis = 0)

def indices_sat_predicate(v, pred, res):
    return floatX(np.array(np.where(pred(v))).transpose())/res

def dadum(v, res, pred, npoints):
    indices = indices_sat_predicate(v, pred, res)
    nvoxels = indices.shape[0]
    # print(nvoxels)
    if nvoxels == npoints:
        return indices
    elif nvoxels > npoints:
        return indices[0:npoints]
    elif nvoxels < npoints:
        # Repeat
        repeatnum = int(np.ceil(float(npoints) / float(nvoxels)))
        i = np.arange(nvoxels)
        iis = np.tile(i,repeatnum)
        iiis = iis[0:npoints]
        return indices[iiis]

def create_pos_neg_batch(voxbatch, npoints, res, zoom = 0.25):
    point_batches = []
    anti_point_batches = []
    for v in voxbatch:
        non_zero_pred = lambda x : x > 0
        zero_pred = lambda x : x == 0
        point_batches.append(dadum(v, res, non_zero_pred, npoints))
        anti_point_batches.append(dadum(v, res, zero_pred, npoints))

    return np.array(point_batches), np.array(anti_point_batches)

def ratio_scaling(inputs, batch_sizes, inp_sizes = None, tnp = T):
    if inp_sizes is None:
        inp_sizes = [inp.size[0] for inp in inputs]

    batches = []
    for i in range(len(inputs)):
        inp = inputs[i]
        input_size = inp_sizes[0]
        if input_size == batch_sizes[i]:
            batches.append(inp)
        elif input_size > batch_sizes[i]:
            indices = np.random.choice(np.arange(input_size), batch_sizes[i],
                                       replace = False)
            batches.append(inp[indices])
        elif batch_sizes[i] > input_size:
            indices = circular_indices(0,batch_sizes[i],input_size)
            batches.append(inp[indices])
    return T.concatenate(batches, axis=0)

def get_batch_sizes(ratios, batch_size):
    ratios = floatX(ratios)
    ratio_counts = (ratios[0:-1] * batch_size).astype('int')
    left_over = batch_size - np.sum(ratio_counts)
    ratio_counts = np.concatenate([ratio_counts, [left_over]])
    assert batch_size = np.sum(ratio_counts)
    assert (ratio_counts() > 0).all()
    return ratio_counts

def handle_field(batch, intermediates, indices, batch_size, ratios):
    ninoutput = len(indices) + 1
    valid_intermediates = []
    for i in indices:
        intermediate = intermediates[i]
        valid_intermediates.append(intermdiate)

    # convert ratios into integer counts for each intermediate and batch
    ratios = floatX(ratios)
    assert len(ratios) == len(valid_intermediates)+1
    ratio_counts = (ratios[0:-1] * batch_size).astype('int')
    left_over = batch_size - np.sum(ratio_counts)
    ratio_counts = np.concatenate([ratio_counts, [left_over]])

    batches = []
    for i in




def destructure(x):
    assert False, "fixme"
    outputs = []
    for i in range(len(x) - 1):
        outputs.append(x[i][0])
        outputs.append(x[i][1])
    outputs.append(x[-1]) # poss diff
    return outputs

def f(zero_field, field):
    # Zero field will be of 1 batc
    # and field

def scalar_field_example(voxel_grids, res, options, niters = 3, field_shape = (100,), npoints = 100,
                         batch_size=64, s_args = {}, add_args = {}):
    ## Types
    points_shape = (npoints,3)
    Field = Type(field_shape)
    Points = Type(points_shape)
    Scalar = Type((npoints,))

    ## Interface
    s = Interface([Field, Points], [Scalar], res_net, **s_args)
    add = Interface([Field, Points], [Field], res_net, **add_args)
    interfaces = [s, add]

    ## Constants
    zero_field = Constant(Field)
    constants = [zero_field]

    ## Variables
    field = ForallVar(Field)
    pos_perturb = ForAllVar(Points)
    poses = [ForAllVar(Points) for i in range(niters*2)]
    forallvars = poses + [pos_perturb, field]

    ## Generators
    zero_field_batch = repeat_to_batch(zero_field.input_var, batch_size)
    field = f(zero_field.input_var, field.input_Var)

    pos1_pos2_gen = [send_infinite_minibatches(voxel_grids, batch_size,
                    f = lambda x : create_pos_neg_batch(x, npoints, res), shuffle=True) for i in range(niters)]
    pos_perturb_gen = infinite_samples(lambda *x: np.random.rand(*x) * 1.0/res, batch_size, points_shape)
    generators = pos1_pos2_gen + [pos_perturb_gen]

    # Since pos1 and pos2 batches are dependent (use the same generator),
    # we need to define a a gen_to_inputs to generate inputs from
    gen_to_inputs = destructure
    # gen_to_inputs = identity

    ## Axioms
    # Zero field is zero everywhere
    axioms = []
    axiom1 = Axiom(s(zero_field_batch, poses[0]), (0,))
    axiom2 = Axiom(s(zero_field_batch, poses[1]), (0,))
    axioms.append(axiom1)
    axioms.append(axiom2)

    intermediates = []

    # If you add a point to a field, the scalar value of that point becomes one
    # and everything else is unchanged
    for i in range(0,niters*2,2intermediates):
        (newfield,) = add(field, poses[i].input_var)
        intermediates.appends(newfield)
        axiom2 = Axiom(s(newfield, poses[i].input_var + pos_perturb.input_var), (1,))
        axiom3 = Axiom(s(newfield, poses[i+1].input_var + pos_perturb.input_var),
                       s(zero_field_batch, poses[i+1].input_var + pos_perturb.input_var))
        field = newfield
        axioms.append(axiom2)
        axioms.append(axiom3)

    ## ratios
    get_batch_sizes

    train_fn, call_fns = compile_fns(interfaces, constants, forallvars, axioms, intermediates, options)
    scalar_field_adt = AbstractDataType(interfaces, constants, forallvars, axioms, name = 'scalar field')
    scalar_field_pbt = ProbDataType(scalar_field_adt, train_fn, call_fns, generators, gen_to_inputs, intermediates)
    return scalar_field_adt, scalar_field_pbt

def main(argv):
    ## Args
    global options
    global test_files, train_files
    global net, output_layer, cost_f, cost_f_dict, val_f, call_f, call_f_dict
    global views, outputs, net
    global interfaces, constants, forallvars, axioms, generators, train_fn, call_fns
    global push, pop
    global X_train
    global adt, pbt

    options = handle_args(argv)
    options['num_epochs'] = 50
    options['compile_fns'] = True
    options['save_params'] = True
    options['train'] = True
    options['nblocks'] = 4
    options['block_size'] = 2
    options['batch_size'] = 512
    options['nfilters'] = 24
    options['layer_width'] = 101
    options['adt'] = 'scalarfield'
    res = options['res'] = 32


    sfx_dict = {}
    for key in ('adt', 'nblocks', 'block_size', 'nfilters'):
        sfx_dict[key] = options[key]
    sfx = stringy_dict(sfx_dict)
    print("sfx:", sfx)
    print(options)

    # X_train = ['test', 'test']
    # X_train = filter(lambda x:x.endswith(".raw") and "train" in x, get_filepaths(os.getenv('HOME') + '/data/ModelNet40'))
    voxel_grids = np.load("/home/zenna/data/ModelNet40/alltrain32.npy")
    adt, pbt = scalar_field_example(voxel_grids, res, options, s_args = options,
        npoints = 500, field_shape = (102,),
        add_args = options, batch_size = options['batch_size'])

    if options['load_params'] == True:
        for i in range(len(interfaces)):
            interfaces[i].load_params_fname("%s_stack_interface_%s.npz" % (sfx, i))
        print("Loaded params")

    if options['train'] == True:
        train_pbt(pbt, num_epochs = options['num_epochs'])

    if options['save_params'] == True:
        for i in range(len(interfaces)):
            interfaces[i].save_params("%s_stack_interface_%s" % (sfx, i))
        print("saved params")

# def circular_indices(lb, ub, thresh):
#     diff = ub - lb
#     if ub > thresh:
#         adada = diff - (thresh - lb)
#         return np.concatenate([np.arange(lb, thresh), np.arange(adada)])
#     else:
#         return np.arange(lb, ub)

def circular_indices(lb, ub, thresh):
    indices = []
    while True:
        stop = min(ub, thresh)
        ix = np.arange(lb, stop)
        indices.append(ix)
        if stop != ub:
            diff = ub - stop
            lb = 0
            ub = diff
        else:
            break

    return np.concatenate(indices)

def get_all_indices(res):
    v = np.zeros((res, res, res))
    return indices_sat_predicate(v, lambda x : x == 0, res)

def val(esf, res, v, npoints):
    """Return a scalar field in voxel form"""
    all_i = np.array(np.where(v>0)).transpose()
    field = add(esf, indices.reshape(1,npoints,3))

    vals = s(field[0], indices.reshape(1,100,3))
    v = np.zeros((res, res, res))
    out_v = np.zeros((res, res, res))
    all_all_i = np.array(np.where(out_v == 0)).transpose()
    nvoxels = all_all_i.shape[0]
    repeatnum = int(np.ceil(float(nvoxels) / float(npoints)))
    for i in range(min(repeatnum, max_n)):
        indices = circular_indices(i*npoints, i*npoints + npoints, nvoxels)
        blank_pos = all_all_i[indices]
        pos_pos = floatX(blank_pos)/res
        vals = s(field[0],pos_pos.reshape(1,npoints,3))[0]
        out_v[blank_pos[:,0], blank_pos[:,1], blank_pos[:,2]] = vals

    if nvoxels == npoints:
        return indices
    elif nvoxels > npoints:
        return indices[0:npoints]
    elif nvoxels < npoints:
        # Repeat
        repeatnum = int(np.ceil(float(npoints) / float(nvoxels)))
        i = np.arange(nvoxels)
        iis = np.tile(i,repeatnum)
        iiis = iis[0:npoints]
        return indices[iiis]


def test():
    voxel_fname = np.random.choice(X_train)
    v = ig.io.load_voxels_binary(voxel_fname, 128, 128, 128, zoom=0.25)
    indices = dadum(v, 32, lambda x : x > 0, 100)
    field = add(esf, indices.reshape(1,100,3))
    s(field[0], indices.reshape(1,100,3))

if __name__ == "__main__":
   main(sys.argv[1:])

import numpy as np
import nlopt

optim_iter = 0
def mk_cost_func(theano_cost_func, exfragcoords, params_shape):
    def cost_func(x, grad):
        global optim_iter
        reshaped_shapes = np.reshape(x, params_shape)
        reshaped_shapes = np.array(reshaped_shapes, dtype='float32')
        obj_cost = theano_cost_func(exfragcoords, reshaped_shapes)
        print obj_cost[1]
        np.save('data/proposal' + str(optim_iter), obj_cost[0], allow_pickle=True, fix_imports=True)
        grad[:] = obj_cost[2].flatten()
        optim_iter = optim_iter + 1
        return float(obj_cost[1])
    return cost_func

def optimize(init_state, cost_func):
    # reset counter
    global optim_iter
    optim_iter = 0
    init_shapes = init_state.flatten()
    nparams = init_state.size
    opt = nlopt.opt(nlopt.LD_MMA, nparams)
    opt.set_min_objective(cost_func)
    opt.set_xtol_rel(1e-4)
    x = opt.optimize(init_shapes)
    minf = opt.last_optimum_value()
    # print "optimum at ", x[0],x[1]
    print "minimum value = ", minf
    # print "result code = ", opt.last_optimize_result()

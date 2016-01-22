from theano import tensor as T
from theano import function, config, shared, printing
import numpy as np
import theano
import numpy
import pickle

from features import *
from theano.compile.nanguardmode import NanGuardMode
curr_mode = None
#curr_mode = NanGuardMode(nan_is_error=True, inf_is_error=True, big_is_error=True)

def dotty(x,y,axis):
    return T.sum(x * y,axis=axis)

## THe function will take as input
# theano.config.optimizer = 'None'
def mindist(translate, radii, min_so_far, ro, rd):
    # ro: 3
    # transalate: nbatch * 3
    # min_so_far: nbatch * width * height
    # rd: width * height * 3
    ro = ro + translate
    # d_o = T.dot(rd, ro)   # 640, 480
    # d_o = dotty(rd, ro, axis=1)
    d_o = T.tensordot(rd, ro, axes=[2,1])
    o_o =  T.sum(ro**2,axis=1)
    b = 2*d_o
    c = o_o - 0.001 #FIXME, remove this squaring
    inner = b **2 - 4 * c   # 640 480
    does_not_intersect = inner < 0.0
    minus_b = -b
    # sqrt_inner = T.sqrt(T.maximum(0.0001, inner))
    eps = 1e-9
    background_dist = 10.0
    sqrt_inner = T.sqrt(T.maximum(eps, inner))
    root1 = (minus_b - sqrt_inner)/2.0
    root2 = (minus_b + sqrt_inner)/2.0
    depth = T.switch(does_not_intersect, background_dist,
                        T.switch(root1 > 0, root1,
                        T.switch(root2 > 0, root2, background_dist)))
    return T.min([min_so_far, depth], axis=0)

def mapedit(ro, rd, params, nprims, width, height):
    # Translate ray origin by the necessary parameters
    nbatch = 20
    translate_params = params[:,:, 0:3]
    sphere_radii = params[:,:, 3]

    # background = np.full((width, height, params.shape[0]), background_dist, dtype=config.floatX)(width, height, params.shape[0])
    background_dist = np.array(10,dtype=config.floatX)
    init_depth = shared(np.full((width, height, nbatch), background_dist, dtype=config.floatX))
    # init_depth = T.alloc(background_dist, width, height, params.shape[1])
    results, updates = theano.scan(mindist, outputs_info=init_depth, sequences=[translate_params, sphere_radii], non_sequences = [ro, rd])
    return results[-1], updates

def castray(ro, rd, shape_params, nprims, width, height):
    return mapedit(ro, rd, shape_params, nprims, width, height)

## Render with ray at ray origin ro and direction rd
def renderrays(ro, rd, shape_params, nprims, width, height):
    # col = np.array([0.7, 0.9, 1.0]) + T.reshape(rd[:,:,1], (width, height, 1)) * 0.8
    return castray(ro, rd, shape_params, nprims, width, height)

# Normalise a vector
def normalize(v):
    return v / np.linalg.norm(v)

def set_camera(ro, ta, cr):
    cw = normalize(ta - ro)
    cp = np.array([np.sin(cr), np.cos(cr),0.0], dtype=config.floatX)
    cu = normalize(np.cross(cw,cp))
    cv = normalize(np.cross(cu,cw))
    return (cu, cv, cw)

# Append an image filled with scalars to the back of an image.
def stack(intensor, width, height, scalar):
    scalars = np.ones([width, height, 1], dtype=config.floatX) * scalar
    return T.concatenate([intensor, scalars], axis=2)

def symbolic_render(nprims, shape_params, fragCoords, width, height):
    """Symbolically render rays starting with fragCoords according to geometry
        defined by shape_params"""
    iResolution = np.array([width, height], dtype=config.floatX)
    q = fragCoords / iResolution
    p = -1.0 + 2.0 * q
    p2 = p * np.array([iResolution[0]/iResolution[1],1.0], dtype=config.floatX)
    # Ray Direction
    op = stack(p2, width, height, 2.0)
    outop = op / T.reshape(op.norm(2, axis=2), (width, height, 1))
    ro = np.array([-0.5+3.5*np.cos(3.0), 2.0, 0.5 + 3.5*np.sin(3.0)], dtype=config.floatX)
    ta = np.array([-0.5, -0.4, 0.5], dtype=config.floatX)
    (cu, cv, cw) = set_camera(ro, ta, 0.0)
    # setup Camera
    a = T.sum(cu * outop, axis=2)
    b = T.sum(cv * outop, axis=2)
    c = T.sum(cw * outop, axis=2)
    # Get ray direction
    rd = T.stack([a,b,c], axis=2)
    ro_ = np.tile(ro, [width, height, 1])
    return renderrays(ro, rd, shape_params, nprims, width, height)

def make_render(nprims, width, height):
    shape_params = T.tensor3('shape')
    fragCoords = T.tensor3('fragCoords')
    res, updates = symbolic_render(nprims, shape_params, fragCoords, width, height)
    render = function([fragCoords, shape_params], res, updates=updates, mode=curr_mode)
    return render

# This generates a d
def similarity_cost(observed_features, nprims, width, height):
    shared_observed_features = [shared(feature) for feature in observed_features]
    shape_params = T.matrix('shape')
    fragCoords = T.tensor3()
    res, updates = symbolic_render(nprims, shape_params, fragCoords, width, height)
    res_tiled = T.tile(res,(1,3,1,1))
    proposal_features = vgg_features(res_tiled)
    summed_dists = feature_compare(proposal_features,shared_observed_features)
    cost = sum(summed_dists)
    cost_grad = T.grad(cost, shape_params)
    cost_compiled = function([fragCoords, shape_params], [res, cost, cost_grad] + summed_dists, updates=updates, mode=curr_mode)
    return cost_compiled

# This generates a d
def similarity_cost2(observed_img, nprims, width, height):
    shared_img = shared(observed_img)
    shape_params = T.matrix('shape')
    fragCoords = T.tensor3()
    res, updates = symbolic_render(nprims, shape_params, fragCoords, width, height)
    eps = 1e-9
    cost = T.sum(T.maximum(eps, (res - observed_img)**2))
    cost_grad = T.grad(cost, shape_params)
    cost_compiled = function([fragCoords, shape_params], [res, cost, cost_grad], updates=updates, mode=NanGuardMode(nan_is_error=True, inf_is_error=True, big_is_error=True))
    return cost_compiled

def gen_fragcoords(width, height):
    """Create a (width * height * 2) matrix, where element i,j is [i,j]
       This is used to generate ray directions based on an increment"""
    fragCoords = np.zeros([width, height, 2], dtype=config.floatX)
    for i in range(width):
        for j in range(height):
            fragCoords[i,j] = np.array([i,j], dtype=config.floatX) + 0.5
    return fragCoords

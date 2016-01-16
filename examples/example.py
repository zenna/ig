#!/usr/bin/env python

"""
Usage example employing ig
"""
import ig
from ig.render import make_render, gen_fragcoords, similarity_cost, similarity_cost2
# from ig.display import draw
import numpy as np
from theano import config


def go():
    return np.random.rand()*4 - 2

def gogo():
    return np.random.rand()*0.1

def genshapes(nprims):
    shapes = []
    for i in range(nprims):
        shapes.append([go(), go(), go(), gogo()])
    return np.array(shapes, dtype=config.floatX)

width = 224
height = 224
# Generate initial rays
exfragcoords = gen_fragcoords(width, height)
nprims = 500
print("Compiling Renderer")
render = make_render(nprims, width, height)

shapes = genshapes(nprims)
print("Rendering")
img = render(exfragcoords, shapes)

np.save('data/observed_img', img, allow_pickle=True, fix_imports=True)

# print("Drawing Img")
# draw(img)

print("Doing Pixel Comparison")
cost = similarity_cost2(img, nprims, width, height)
sim = cost(exfragcoords, shapes)

print(sim)

img_tiled = np.tile(img,(1,3,1,1)) # Tile because vgg expects RGB but img is depth
## Render the image to create an observation
print("Compiling VGG")
vgg_network = ig.features.gen_vgg()
print("Generating Features")
observed_features = vgg_network(img_tiled)

# similarity_cost
print("Compiling Cost Function")
cost = similarity_cost(observed_features, nprims, width, height)
print("Evaluating Cost and Gradient")

# Generate new shapes
shapes = genshapes(nprims)
sim2 = cost(exfragcoords, shapes)
print(sim2)

import nlopt
print("Doing Pyton Optimisation")

i = 0
def cost_func(x, grad):
    global i
    reshaped_shapes = np.reshape(x, shapes.shape)
    reshaped_shapes = np.array(reshaped_shapes, dtype='float32')
    obj_cost = cost(exfragcoords, reshaped_shapes)
    print obj_cost[1]
    np.save('data/proposal' + str(i), obj_cost[0], allow_pickle=True, fix_imports=True)
    # grad[:] = obj_cost[2].flatten()
    i = i + 1
    return float(obj_cost[1])

init_shapes = shapes.flatten()
nparams = shapes.size
opt = nlopt.opt(nlopt.LD_MMA, nparams)
opt.set_min_objective(cost_func)
opt.set_xtol_rel(1e-4)
x = opt.optimize(init_shapes)
minf = opt.last_optimum_value()
# print "optimum at ", x[0],x[1]
print "minimum value = ", minf
# print "result code = ", opt.last_optimize_result()

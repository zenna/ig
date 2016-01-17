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
nprims = 10
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
theano_cost_func = similarity_cost(observed_features, nprims, width, height)
print("Evaluating Cost and Gradient")

# Generate new shapes
from ig.optimize import mk_cost_func, optimize
init_shapes = genshapes(nprims)
theano_cost = theano_cost_func(exfragcoords, shapes)

print("Doing Pyton Optimisation")
cost_func = mk_cost_func(theano_cost_func, exfragcoords, init_shapes.shape)
optimize(init_shapes, cost_func)

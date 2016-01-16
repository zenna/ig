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

width = 224
height = 224
# Generate initial rays
exfragcoords = gen_fragcoords(width, height)
nprims = 500
print("Compiling Renderer")
render = make_render(nprims, width, height)
shapes = []
for i in range(nprims):
    shapes.append([go(), go(), go(), gogo()])

shapes = np.array(shapes, dtype=config.floatX)
print("Rendering")
img = render(exfragcoords, shapes)

# print("Drawing Img")
# draw(img)

print("Doing Pixel Comparison")
cost = similarity_cost2(img, nprims, width, height)
print(cost(exfragcoords, shapes))

img_tiled = np.tile(img,(1,3,1,1)) # Tile because vgg expects RGB but img is depth
## Render the image to create an observation
print("Compiling VGG")
vgg_network = ig.features.gen_vgg()
print("Generating Features")
observed_features = vgg_network(img_tiled)

# similarity_cost
print("Compiling Cost Function")
cost = similarity_cost(observed_features, nprims, width, height)
print(cost(exfragcoords, shapes))

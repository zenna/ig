#!/usr/bin/env python

"""
Usage example employing ig
"""
import ig
from ig.render import make_render, gen_fragcoords, similarity_cost, similarity_cost2
# from ig.display import draw
import numpy as np
from theano import config
config.optimizer='fast_compile'

def go():
    return np.random.rand()*4 - 2

def gogo():
    return np.random.rand()*0.1

def genshapes(nprims):
    shapes = []
    for i in range(nprims):
        shapes.append([go(), go(), go()])
    return np.array(shapes, dtype=config.floatX)

def genshapebatch(nprims, nbatch):
    shapes = np.random.rand(nprims, nbatch, 3)*2 - 2
    return np.array(shapes, dtype=config.floatX)

width = 224
height = 224
# Generate initial rays
exfragcoords = gen_fragcoords(width, height)
nprims = 500
nbatch = 1
print("Compiling Renderer")
render = make_render(nprims, width, height)
shapes = genshapebatch(nprims, nbatch)
print("Rendering")
img = render(exfragcoords, shapes)

print("Drawing")
from ig import display
display.draw(img[:,:,0])

np.save('data/observed_img', img, allow_pickle=True, fix_imports=True)
img_tiled = np.tile(img,(1,3,1,1)) # Tile because vgg expects RGB but img is depth

img_tiled = np.reshape(np.tile(img,3),(1,3,width,height))
## Render the image to create an observation
print("Compiling VGG")
vgg_network = ig.features.gen_vgg()
print("Generating Features")
observed_features = vgg_network(img_tiled)

import theano
rand_shape_params = theano.shared(genshapebatch(nprims, nbatch))

# similarity_cost
print("Compiling Cost Function")
theano_cost_func = ig.render.similarity_cost3(observed_features, rand_shape_params, nprims, width, height)
print("Evaluating Cost and Gradient")

# import ig.display
def train(costfunc,  exfragcoords,  nprims = 200, nbatch = 1, num_epochs = 5000, width = 224, height = 224, save_data = True):
    full_dir_name = ""
    if save_data:
        datadir = os.environ['DATADIR']
        newdirname = str(time.time())
        full_dir_name = os.path.join(datadir, newdirname)
        print "Making Directory", full_dir_name
        os.mkdir(full_dir_name)

    print("Starting Training")
    for epoch in range(num_epochs):
        test_err = costfunc(exfragcoords)
        print "epoch", epoch
        print "loss", test_err
        print "\n"
        if save_data:
            fname = "epoch%s" % (epoch)
            full_fname = os.path.join(full_dir_name, fname)
            np.savez_compressed(full_fname, *test_err)

    return lasagne.layers.get_all_param_values(network)

train(theano_cost_func, exfragcoords, nprims = nprims)

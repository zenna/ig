## Volume Raycasting
from theano import function, config, shared, printing
import numpy as np


def gen_fragcoords(width, height):
    """Create a (width * height * 2) matrix, where element i,j is [i,j]
       This is used to generate ray directions based on an increment"""
    fragCoords = np.zeros([width, height, 2], dtype=config.floatX)
    for i in range(width):
        for j in range(height):
            fragCoords[i,j] = np.array([i,j], dtype=config.floatX) + 0.5
    return fragCoords

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
    return np.concatenate([intensor, scalars], axis=2)

def symbolic_render(ro, ta, shape_params, fragCoords, width, height):
    """Symbolically render rays starting with fragCoords according to geometry
        defined by shape_params"""
    iResolution = np.array([width, height], dtype=config.floatX)
    q = fragCoords / iResolution
    p = -1.0 + 2.0 * q
    p2 = p * np.array([iResolution[0]/iResolution[1],1.0], dtype=config.floatX)
    # Ray Direction
    op = stack(p2, width, height, 2.0)
    outop = op / np.reshape(np.linalg.norm(op,2,axis=2), (width, height, 1))
    ro = np.array(ro, dtype=config.floatX)
    ta = np.array(ta, dtype=config.floatX)
    (cu, cv, cw) = set_camera(ro, ta, 0.0)
    # setup Camera
    a = np.sum(cu * outop, axis=2)
    b = np.sum(cv * outop, axis=2)
    c = np.sum(cw * outop, axis=2)
    # Get ray direction
    rd = np.stack([a,b,c], axis=2)
    return rd,ro

def make_render(shape_params, width, height):
    fragCoords = gen_fragcoords(width, height)
    return symbolic_render(shape_params, fragCoords, width, height)

def switch(cond, a, b):
    return cond*a + (1-cond)*b

def main(shape_params, width, height, nsteps, res, ro = [3.5, 2.8, 3.0], ta = [-0.5, -0.4, 0.5]):
    fragCoords = gen_fragcoords(width, height)
    rd, ro = symbolic_render(ro, ta, shape_params, fragCoords, width, height)
    a = 0 - ro # c = 0
    b = 1 - ro # c = 1
    tn = a/rd
    tf = b/rd
    tn_true = np.minimum(tn,tf)
    tf_true = np.maximum(tn,tf)
    # do X
    tn_x = tn_true[:,:,0]
    tf_x = tf_true[:,:,0]
    tmin = np.full(tn_x.shape, 0)
    tmax = np.full(tn_x.shape, 10)
    t0 = tmin
    t1 = tmax
    t02 = switch(tn_x > t0, tn_x, t0)
    t12 = switch(tf_x < t1, tf_x, t1)
    # y
    tn_x = tn_true[:,:,1]
    tf_x = tf_true[:,:,1]
    t03 = switch(tn_x > t02, tn_x, t02)
    t13 = switch(tf_x < t12, tf_x, t12)
    #z
    tn_x = tn_true[:,:,2]
    tf_x = tf_true[:,:,2]
    t04 = switch(tn_x > t03, tn_x, t03)
    t14 = switch(tf_x < t13, tf_x, t13)

    # Shit a little bit to avoid numerial inaccuracies
    t04 = t04*1.001
    t14 = t14*0.999

    # img = np.zeros(width * height)
    left_over = np.ones(width * height)
    step_size = (t14 - t04)/nsteps
    orig = ro + rd* np.reshape(t04,(width, height, 1))
    step_size = np.reshape(step_size, (width, height, 1))
    for i in range(nsteps):
        # print "step", i
        pos = orig + rd*step_size*i
        voxel_indices = np.floor(pos*res)
        pruned = np.clip(voxel_indices,0,res-1)
        p_int = pruned.astype('int')
        indices = np.reshape(p_int, (width*height,3))
        attenuation = shape_params[indices[:,0],indices[:,1],indices[:,2]]
        left_over = left_over*np.exp(-attenuation*np.reshape(step_size, (width * height)))
        # print "value", np.sum(value)
        # print "indices", np.sum(indices)
        # print "pos", np.sum(pos)
        # img = img + value * left_over
        # left_over = (1-value)*left_over
    img = 1.0 - left_over
    pixels = np.reshape(img,(width, height))
    # return t14>t04
    return switch(t14>t04, pixels, np.zeros(pixels.shape))

from matplotlib import pylab as plt

def load_voxels_binary(fname, width, height, depth, max_value=255.0):
    data = np.fromfile(fname, dtype='uint8')
    return np.reshape(data, (width, height, depth))/float(max_value)

def histo(x):
    # the histogram of the data
    n, bins, patches = plt.hist(x.flatten(), 500,range=(0.0001,1), normed=1, facecolor='green', alpha=0.75)
    plt.show()

plt.ion()

width = 200
height = 200
res = 256
## Example Data
# x, y, z = np.ogrid[-10:10:complex(res), -10:10:complex(res), -10:10:complex(res)]
# shape_params = np.sin(x*y*z)/(x*y*z)
# shape_params = np.clip(shape_params,0,1)
# shape_params = shape_params - np.min(shape_params) * (np.max(shape_params) - np.min(shape_params))
shape_params = load_voxels_binary("aneurism.raw", 256, 256, 256)*10.0
#
# shape_params = np.zeros((res,res,res))
# q = np.transpose(np.mgrid[0:1:complex(res),0:1:complex(res),0:1:complex(res)],(1,2,3,0))
# distances = np.sum((q-[0.5,0.5,0.5])**2,axis=3)
# # shape_params = 1 - distances
# shape_params =  (distances < 0.3) * 10.0

nsteps = 100
ro = [1.5, 1.4, 1.5]
ta = [0.7, 0.1, 0.5]
print "Drawing"
img = main(shape_params, width, height, nsteps, res, ro=ro, ta=ta)
plt.figure()
plt.imshow(img)
plt.draw()

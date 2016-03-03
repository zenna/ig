## Volume Raycasting
from theano import function, config, shared, printing
import numpy as np
from scipy.sparse import csr_matrix
try:
    from mayavi import mlab
except:
    print "couldnt import"
from mayavi import mlab

def rand_rotation_matrix(deflection=1.0, randnums=None):
    """
    Creates a random rotation matrix.

    deflection: the magnitude of the rotation. For 0, no rotation; for 1, competely random
    rotation. Small deflection => small perturbation.
    randnums: 3 random numbers in the range [0, 1]. If `None`, they will be auto-generated.
    """
    # from http://www.realtimerendering.com/resources/GraphicsGems/gemsiii/rand_rotation.c

    if randnums is None:
        randnums = np.random.uniform(size=(3,))

    theta, phi, z = randnums

    theta = theta * 2.0*deflection*np.pi  # Rotation about the pole (Z).
    phi = phi * 2.0*np.pi  # For direction of pole deflection.
    z = z * 2.0*deflection  # For magnitude of pole deflection.

    # Compute a vector V used for distributing points over the sphere
    # via the reflection I - V Transpose(V).  This formulation of V
    # will guarantee that if x[1] and x[2] are uniformly distributed,
    # the reflected points will be uniform on the sphere.  Note that V
    # has length sqrt(2) to eliminate the 2 in the Householder matrix.

    r = np.sqrt(z)
    Vx, Vy, Vz = V = (
        np.sin(phi) * r,
        np.cos(phi) * r,
        np.sqrt(2.0 - z)
        )

    st = np.sin(theta)
    ct = np.cos(theta)

    R = np.array(((ct, st, 0), (-st, ct, 0), (0, 0, 1)))

    # Construct the rotation matrix  ( V Transpose(V) - I ) R.

    M = (np.outer(V, V) - np.eye(3)).dot(R)
    return M


# Genereate values in raster space, x[i,j] = [i,j]
def gen_fragcoords(width, height):
    """Create a (width * height * 2) matrix, where element i,j is [i,j]
       This is used to generate ray directions based on an increment"""
    raster_space = np.zeros([width, height, 2], dtype=config.floatX)
    for i in range(width):
        for j in range(height):
            raster_space[i,j] = np.array([i,j], dtype=config.floatX) + 0.5
    return raster_space

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

def symbolic_render(r, shape_params, raster_space, width, height):
    """Symbolically render rays starting with raster_space according to geometry
        defined by shape_params"""
    resolution = np.array([width, height], dtype=config.floatX)
    # Normalise it to be bound between 0 1
    norm_raster_space = raster_space / resolution
    # Put it in NDC space, -1, 1
    screen_space = -1.0 + 2.0 * norm_raster_space
    # Make pixels square by mul by aspect ratio
    ndc_space = screen_space * np.array([resolution[0]/resolution[1],1.0], dtype=config.floatX)
    # Ray Direction

    # Position on z-plane
    ndc_xyz = stack(ndc_space, width, height, 1.0)*0.5 # Change focal length

    # Put the origin farther along z-axis
    ro = np.array([0,0,1.5])

    # Rotate both by same rotation matrix
    ro_t = np.dot(ro, r)
    ndc_t = np.dot(ndc_xyz, r)

    # Increment by 0.5 since voxels are in [0, 1]
    ro_t = ro_t + 0.5
    ndc_t = ndc_t + 0.5
    # Find normalise ray dirs from origin to image plane
    unnorm_rd = ndc_t - ro_t
    rd = unnorm_rd / np.reshape(np.linalg.norm(unnorm_rd, 2, axis=2), (width, height, 1))
    return rd, ro_t

def make_render(shape_params, width, height):
    raster_space = gen_fragcoords(width, height)
    return symbolic_render(shape_params, raster_space, width, height)

def switch(cond, a, b):
    return cond*a + (1-cond)*b

def main(shape_params, rotation_matrix, width, height, nsteps, res):
    raster_space = gen_fragcoords(width, height)
    rd, ro = symbolic_render(rotation_matrix, shape_params, raster_space, width, height)
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
    step_size_flat = step_size.flatten()
    xres = yres = zres = res
    nrays = width * height
    nvoxels = xres * yres * zres
    A = csr_matrix((nrays, nvoxels))
    for i in range(nsteps):
        # print "step", i
        pos = orig + rd*step_size*i
        voxel_indices = np.floor(pos*res)
        pruned = np.clip(voxel_indices,0,res-1)
        p_int = pruned.astype('int')
        indices = np.reshape(p_int, (width*height,3))
        unique_indices = indices[:,0]*yres*zres + indices[:,1]*zres + indices[:,2]
        # print unique_indices[50]
        A_inc = csr_matrix((-step_size_flat, (np.arange(nrays), unique_indices)), shape=(nrays, nvoxels))
        A = A + A_inc
        attenuation = shape_params[indices[:,0],indices[:,1],indices[:,2]]
        left_over = left_over*np.exp(-attenuation*np.reshape(step_size, (width * height)))
        # print "value", np.sum(value)
        # print "indices", np.sum(indices)
        # print "pos", np.sum(pos)
        # img = img + value * left_over
        # left_over = (1-value)*left_over
    img = left_over
    pixels = np.reshape(img,(width, height))
    # return t14>t04
    mask = t14>t04
    valid_ray_indices = np.nonzero(np.reshape(mask, width*height))[0]
    return switch(t14>t04, pixels, np.ones(pixels.shape)), A[valid_ray_indices, :], t14>t04, np.reshape(pixels, width*height)[valid_ray_indices]

from matplotlib import pylab as plt

def load_voxels_binary(fname, width, height, depth, max_value=255.0):
    data = np.fromfile(fname, dtype='uint8')
    return np.reshape(data, (width, height, depth))/float(max_value)

def histo(x):
    # the histogram of the data
    n, bins, patches = plt.hist(x.flatten(), 500,range=(0.0001,1), normed=1, facecolor='green', alpha=0.75)
    plt.show()

def drawdata(fname):
  data = np.load(fname)
  data2 = data.items()
  voxels = data2[3][1]
  r = random_rotation_matrices(3)
  img = render(voxels, r)[0]
  drawimgbatch(img)

plt.ion()

width = 100
height = 100
res = 256
## Example Data
# x, y, z = np.ogrid[-10:10:complex(res), -10:10:complex(res), -10:10:complex(res)]
# shape_params = np.sin(x*y*z)/(x*y*z)
# shape_params = np.clip(shape_params,0,1)
# shape_params = shape_params - np.min(shape_params) * (np.max(shape_params) - np.min(shape_params))
shape_params = load_voxels_binary("person_0089.raw", res, res, res)*10.0
#
# shape_params = np.zeros((res,res,res))
# q = np.transpose(np.mgrid[0:1:complex(res),0:1:complex(res),0:1:complex(res)],(1,2,3,0))
# distances = np.sum((q-[0.5,0.5,0.5])**2,axis=3)
# # shape_params = 1 - distances
# shape_params =  (distances < 0.3) * 10.0

nsteps = 500
rs = []
As = []
imgs = []
for i in range(50):
    r = rand_rotation_matrix()
    rs.append(r)
    print "Drawing", i
    img, A, mask, img_lim = main(shape_params, r, width, height, nsteps, res)
    As.append(A)
    imgs.append(img_lim)
    plt.figure()
    plt.imshow(img)
    plt.draw()


from scipy import sparse
As_t = sparse.vstack(As)
imgs_t = np.concatenate(imgs)
log_imgs = np.log(imgs_t)

sol = sparse.linalg.lsqr(As_t, log_imgs)
voxels = np.reshape(sol[0], (res, res, res))
img_hyp, A, mask, img_hyp_lim = main(voxels, rand_rotation_matrix(), width, height, nsteps, res)
plt.figure()
plt.imshow(img_hyp)
plt.draw()

mlab.pipeline.volume(mlab.pipeline.scalar_field(voxels/np.max(voxels*10)))

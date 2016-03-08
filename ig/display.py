import pylab
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import theano.printing

try:
    from mayavi import mlab
except:
    print "couldnt import"
from mayavi import mlab

plt.ion()

def drawdata(fname, nviews = 3, floatX = 'float32'):
  data = np.load(fname)
  voxels = data['voxels']
  r = rand_rotation_matrices(nviews, floatX = floatX)
  img = render(voxels, r)[0]
  drawimgbatch(img)

def drawimgbatch(imbatch):
    from matplotlib import pylab as plt
    plt.ion()
    nvoxgrids = imbatch.shape[0]
    nviews = imbatch.shape[1]
    plt.figure()

    for i in range(nvoxgrids):
        for j in range(nviews):
            plt.subplot(nvoxgrids, nviews, i * nviews + j + 1)
            plt.imshow(imbatch[i,j])

    plt.draw()

def draw(img):
    plt.figure()
    plt.imshow(img)
    plt.draw()

def topng(fname):
    img = np.load(fname + '.npy')
    plt.figure()
    plt.imshow(img)
    plt.savefig(fname + '.png')

def draw_graph(fname):
    theano.printing.pydotprint(res, outfile=fname, var_with_name_simple=True)

def visoptim(narrays):
    img = np.load('/home/zenna/data/observed_img.npy')
    plt.figure()
    plt.imshow(img)
    plt.savefig('/home/zenna/data/observed_img.png')

    for i in range(narrays):
        plt.figure()
        plt.imshow(np.load('/home/zenna/data/proposal' + str(i) + '.npy'))
        plt.savefig('/home/zenna/data/proposal' + str(i) + '.png')

def histo(x):
    # the histogram of the data
    n, bins, patches = plt.hist(x.flatten(), 500,range=(0.0001,1), normed=1, facecolor='green', alpha=0.75)
    plt.show()

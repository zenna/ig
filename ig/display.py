import pylab
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

def draw(img):
    plt.imshow(img)
    plt.show()

def topng(fname):
    img = np.load(fname + '.npy')
    plt.figure()
    plt.imshow(img)
    plt.savefig(fname + '.png')

def visoptim(narrays):
    img = np.load('/home/zenna/data/observed_img.npy')
    plt.figure()
    plt.imshow(img)
    plt.savefig('/home/zenna/data/observed_img.png')

    for i in range(narrays):
        plt.figure()
        plt.imshow(np.load('/home/zenna/data/proposal' + str(i) + '.npy'))
        plt.savefig('/home/zenna/data/proposal' + str(i) + '.png')

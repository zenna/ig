from ig import display
import numpy as np

def compare(epoch, batch=0,prefix=""):
  x0  = np.load("%s/epoch%s.npz" % (prefix,epoch))
  print "loss", x0.items()[0][1]
  unchanged_img = x0.items()[-3][1]
  changed_img = x0.items()[-2][1]
  res_reshape2 = x0.items()[-1][1]
  display.draw(unchanged_img[batch])
  display.draw(changed_img[batch])
  display.draw(res_reshape2[batch])

def show_losses(nfiles,elem=0):
    losses = []
    for epoch in range(nfiles):
      x0  = np.load("epoch%s.npz" % epoch)
      losses.append(x0.items()[elem][1])

    return losses


compare(292,0, "/home/zenna/awsdata/1453896892.63")



def show_imgs():
    imgs = np.load("/home/zenna/omhome/repos/ig/data/whitenoiseimgs.npy")
    nimgs = imgs.shape[0]
    nstacks = int(np.floor(float(nimgs)/2))
    for i in range(nstacks):
        plt.subplot(nstacks,1,i+1)
        plt.imshow(imgs[i].reshape(28,28))

    plt.figure()
    for j in range(nimgs-nstacks-1):
        i = j + nstacks + 1
        plt.subplot(nimgs-nstacks,1,j+1)
        plt.imshow(imgs[i].reshape(28,28))

    plt.show()

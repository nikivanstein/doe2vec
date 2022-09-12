import os
import numpy as np
import bbobbenchmarks as bbob
from doe2vec import doe_model

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import matplotlib.pyplot as plt
from matplotlib import cm

def plotReconstruction(sample, originals, decoded_does):
    n = len(decoded_does)
    fig = plt.figure(figsize=(8*4, 6*4))
    for i in range(24):
        # display original
        pos = 1 + (i%6 * 8)
        if (i >= 18):
            pos += 6
        elif (i >= 12):
            pos += 4
        elif (i >= 6):
            pos += 2
        ax = fig.add_subplot(6, 8, pos, projection="3d")
        ax.plot_trisurf(
            sample[:, 0],
            sample[:, 1],
            originals[i],
            cmap=cm.jet,
            antialiased=True,
        )
        ax.xaxis.set_ticklabels([])
        ax.yaxis.set_ticklabels([])
        ax.zaxis.set_ticklabels([])

        for line in ax.xaxis.get_ticklines():
            line.set_visible(False)
        for line in ax.yaxis.get_ticklines():
            line.set_visible(False)
        for line in ax.zaxis.get_ticklines():
            line.set_visible(False)
        plt.title(f"BBOB $f_{{{i+1}}}$", fontdict={'fontsize':20})

        pos = 2 + (i%6 * 8)
        if (i >= 18):
            pos += 6
        elif (i >= 12):
            pos += 4
        elif (i >= 6):
            pos += 2
        # display reconstruction
        ax = fig.add_subplot(6, 8, pos, projection="3d")
        ax.plot_trisurf(
            sample[:, 0],
            sample[:, 1],
            decoded_does[i],
            cmap=cm.terrain,
            antialiased=True,
        )
        ax.xaxis.set_ticklabels([])
        ax.yaxis.set_ticklabels([])
        ax.zaxis.set_ticklabels([])

        for line in ax.xaxis.get_ticklines():
            line.set_visible(False)
        for line in ax.yaxis.get_ticklines():
            line.set_visible(False)
        for line in ax.zaxis.get_ticklines():
            line.set_visible(False)
        plt.title(f"Reconstructed $f_{{{i+1}}}$", fontdict={'fontsize':20})
    #plt.tight_layout()
    plt.savefig("reconstructions.png", bbox_inches='tight')



obj = doe_model(
    2, 8, n=1000000, latent_dim=24, model_type="VAE", kl_weight=0.001, use_mlflow=False
)
if not obj.loadModel("../models"):
    raise Exception("Model not trained yet")
sample = obj.sample * 10 - 5

i = 0
Y = []
for f in range(1, 25):
    fun, opt = bbob.instantiate(f, i)
    bbob_y = np.asarray(list(map(fun, sample)))
    array_y = (bbob_y.flatten() - np.min(bbob_y)) / (
        np.max(bbob_y) - np.min(bbob_y)
    )
    Y.append(array_y)

encoded_does = obj.encode(np.array(Y))
decoded_does = obj.autoencoder.decoder(encoded_does).numpy()
    
plotReconstruction(sample, Y, decoded_does)

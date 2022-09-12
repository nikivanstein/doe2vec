import os
import numpy as np
import bbobbenchmarks as bbob
from doe2vec import doe_model

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import matplotlib.pyplot as plt
from matplotlib import cm

def plotReconstruction(sample, originals, decoded_does, filename, titles):
    n = len(decoded_does)
    fig = plt.figure(figsize=(8*4, 6*4))
    for i in range(len(decoded_does)):
        # display original
        pos = i+1
        ax = fig.add_subplot(1, len(decoded_does)+1, pos, projection="3d")

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
        plt.title(titles[i], fontdict={'fontsize':20})
        """
        pos = i*2 + 1
        # display reconstruction
        ax = fig.add_subplot(2, len(decoded_does), pos, projection="3d")
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
        """
    #plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight')



obj = doe_model(
    2, 10, n=250000, latent_dim=24, model_type="VAE", kl_weight=0.001, use_mlflow=False
)
if not obj.loadModel("../models"):
    obj.generateData()
    obj.compile()
    obj.fit(100)
    obj.saveModel("../models/")

sample = obj.sample * 10 - 5

#i = {17:7, 5:2}


for first in range(1,25):
    Y = []
    fs = [first,5]
    i = {fs[0]:7, fs[1]:2}
    for f in fs:
        fun, opt = bbob.instantiate(f, i[f])
        bbob_y = np.asarray(list(map(fun, sample)))
        array_y = (bbob_y.flatten() - np.min(bbob_y)) / (
            np.max(bbob_y) - np.min(bbob_y)
        )
        Y.append(array_y)

    encoded_does = obj.encode(np.array(Y))
    decoded_does = obj.autoencoder.decoder(encoded_does).numpy()
    min_doe =  encoded_does[0]/2 + encoded_does[1]/2
    min_doe2 = decoded_does[0]/2 + decoded_does[1]/2
    print(encoded_does)
    print(min_doe)
    print(decoded_does.shape)
    enc_min = obj.autoencoder.decoder(np.array([min_doe])).numpy()
    print("enc",enc_min[0])
    decoded_does = list(decoded_does)
    decoded_does.append(enc_min[0])
    decoded_does.append(min_doe2)
    decoded_does = np.array(decoded_does)
    print("all", decoded_does.shape)
    plotReconstruction(sample, Y,decoded_does, f"average/{fs[0]}plus{fs[1]}.png", [f"$d(e(f_{{{fs[0]}}}))$", f"$d(e(f_{{{fs[1]}}}$))", f"$d(e(f_{{{fs[0]}}})/2$ + $e(f_{{{fs[1]}}})/2)$", f"$d(e(f_{{{fs[0]}}}))/2$ + $d(e(f_{{{fs[1]}}}))/2$"])

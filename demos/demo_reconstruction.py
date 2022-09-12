"""Reconstruction demo
@author: Bas van Stein
For this demo you need gradio `pip install gradio`

Shows the reconstruction of 2d landscapes by sliding the latent space variables.
"""


import os
import numpy as np
import bbobbenchmarks as bbob
from doe2vec import doe_model

#remove line if you only use a device with 1 GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import matplotlib.pyplot as plt
from matplotlib import cm


obj = doe_model(
    2, 8, n=250000, latent_dim=8, model_type="VAE", kl_weight=0.001, use_mlflow=False
)
if not obj.loadModel("../models"):
    obj.generateData()
    obj.compile()
    obj.fit(100)
    obj.save()

def plotReconstruction(sample, decoded):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1, projection="3d")
    ax.plot_trisurf(
        sample[:, 0],
        sample[:, 1],
        decoded[0],
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
    plt.title("Reconstruction of the latent space", fontdict={'fontsize':14})
    plt.savefig("reco.png", bbox_inches='tight', transparent=True)
    plt.close()

def predict(x1,x2,x3,x4,x5,x6,x7,x8):
    latentspace = np.atleast_2d([x1,x2,x3,x4,x5,x6,x7,x8])
    doe = obj.autoencoder.decoder(latentspace).numpy()
    plotReconstruction(obj.sample, doe)
    return 'reco.png'


import gradio as gr

gr.Interface(
    predict,
    inputs=[
        gr.Slider(-2.0, 2.0, label='x1', step=0.01, value=0.0),
        gr.Slider(-2.0, 2.0, label='x2', step=0.01, value=0.0),
        gr.Slider(-2.0, 2.0, label='x3', step=0.01, value=0.0),
        gr.Slider(-2.0, 2.0, label='x4', step=0.01, value=0.0),
        gr.Slider(-2.0, 2.0, label='x5', step=0.01, value=0.0),
        gr.Slider(-2.0, 2.0, label='x6', step=0.01, value=0.0),
        gr.Slider(-2.0, 2.0, label='x7', step=0.01, value=0.0),
        gr.Slider(-2.0, 2.0, label='x8', step=0.01, value=0.0),
    ],
    outputs="image",
    live=True,
    allow_flagging=False,
).launch()




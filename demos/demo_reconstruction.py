"""Reconstruction demo
@author: Bas van Stein
For this demo you need gradio `pip install gradio plotly`

Shows the reconstruction of 2d landscapes by sliding the latent space variables.
"""


import os
import numpy as np
import bbobbenchmarks as bbob
from doe2vec import doe_model
import gradio as gr

#remove line if you want to use a GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import matplotlib.pyplot as plt
from matplotlib import cm
import plotly.graph_objects as go

obj = doe_model(
    2, 8, n=250000, latent_dim=8, model_type="VAE", kl_weight=0.001, use_mlflow=False
)
if not obj.loadModel("../models"):
    obj.generateData()
    obj.compile()
    obj.fit(100)
    obj.save()


def plotReconstructionInteractive(sample, decoded):
    layout = go.Layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        scene={"aspectmode":"cube"}
    )
    fig = go.Figure(data=[
        go.Mesh3d(
            x=sample[:,0],
            y=sample[:,1],
            z=decoded[0],
            opacity=0.8,
            intensity=decoded[0],
            showlegend=False,
            showscale=False,
            colorscale="Viridis",
        ),], layout=layout)
    fig.update_layout(
        width=600,
        height=600, 
        margin=dict(l=10, r=10, b=10, t=10))
    return fig


def plotReconstruction(sample, decoded):
    plt.rcParams.update({
        "figure.facecolor":  (0.0, 0.0, 0.0, 0.0), 
        "axes.facecolor":    (0.0, 0.0, 0.0, 0.0),  
    })
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
    return fig


def predict_ls(x1,x2,x3,x4,x5,x6,x7,x8):
    latentspace = np.atleast_2d([x1,x2,x3,x4,x5,x6,x7,x8])
    doe = obj.autoencoder.decoder(latentspace).numpy()
    return plotReconstructionInteractive(obj.sample, doe)


gr.Interface(
    predict_ls,
    title="DoE2Vec - reconstruction demo",
    description="Change the sliders on the left to generate different landscapes with the decoder part of the VAE.",
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
    outputs="plot",
    live=True,
    allow_flagging='never',
).launch()
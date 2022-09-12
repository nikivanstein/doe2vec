"""Reconstruction demo
@author: Bas van Stein
For this demo you need gradio and plotly installed `pip install gradio plotly`

Shows the reconstruction of 2d landscapes by sliding the latent space variables.
"""


import os
import numpy as np
import bbobbenchmarks as bbob
from doe2vec import doe_model
import copy
import plotly.graph_objects as go
from plotly.subplots import make_subplots
#remove line if you want to use GPUs
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import matplotlib.pyplot as plt
from matplotlib import cm

latent_size = 4


obj = doe_model(
    2, 10, n=250000, latent_dim=latent_size, model_type="VAE", kl_weight=0.001, use_mlflow=False
)
if not obj.loadModel("../models"):
    obj.generateData()
    obj.compile()
    obj.fit(100)
    obj.saveModel("../models")

def calcVariance(encoding, feature_nr):
    encodings = []
    for x in np.linspace(encoding[feature_nr]-1.0, encoding[feature_nr] + 1.0, num=200):
        new_enc = copy.deepcopy(encoding)
        new_enc[feature_nr] = x
        encodings.append(new_enc)
    encodings = np.array(encodings)
    decoded_does = obj.autoencoder.decoder(encodings).numpy()
    decoded_var = np.var(decoded_does, axis=0)
    return decoded_var


def plotReconstructionInteractive(sample, decoded, ori, encoded):
    fig = make_subplots(
        rows=2, cols=4,
        specs=[[None,{'type': 'scene'}, {'type': 'scene'},None],
            [{'type': 'scene'}, {'type': 'scene'}, {'type': 'scene'}, {'type': 'scene'}]],
        subplot_titles=('','Original DOE','Reconstruction','','Var 1', 'Var 2', 'Var 3', 'Var 4'))
    
    fig.add_trace(
        go.Scatter3d(x=sample[:, 0], y=sample[:, 1], z=ori, mode="markers", marker=dict(
                size=8,
                color=ori,  # set color to an array/list of desired values
                colorscale='Viridis',   # choose a colorscale
                opacity=0.8
            )),
            row=1, col=2)
    fig.add_trace(
        go.Scatter3d(x=sample[:, 0], y=sample[:, 1], z=decoded[0], mode="markers", marker=dict(
                size=8,
                color=decoded[0],  # set color to an array/list of desired values
                colorscale='Viridis',   # choose a colorscale
                opacity=0.8
            )),
            row=1, col=3)

    for i in range(latent_size):
        var = calcVariance(encoded[0], i)
        fig.add_trace(
            go.Scatter3d(x=sample[:, 0], y=sample[:, 1], z=decoded[0], mode="markers", marker=dict(
                size=8,
                color=var,  # set color to an array/list of desired values
                colorscale='Viridis',   # choose a colorscale
                opacity=0.8
            )),
            row=2, col=i+1)
    fig.update_layout(height=800, width=1400, paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        scene={"aspectmode":"cube"})
    return fig

def plotReconstruction(sample, decoded, ori, encoded):
    plt.rcParams.update({
        "figure.facecolor":  (0.0, 0.0, 0.0, 0.0), 
        "axes.facecolor":    (0.0, 0.0, 0.0, 0.0),  
    })
    fig = plt.figure(figsize=(16, 8))
    ax = fig.add_subplot(2, latent_size, int(latent_size/2), projection="3d")
    ax.scatter(
        sample[:, 0],
        sample[:, 1],
        ori,
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
    plt.title("Original DOE", fontdict={'fontsize':14, "color":"#ffffff"})

    ax = fig.add_subplot(2, latent_size, int(latent_size/2)+1, projection="3d")
    ax.scatter(
        sample[:, 0],
        sample[:, 1],
        decoded[0],
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
    plt.title("Reconstruction", fontdict={'fontsize':14, "color":"#ffffff"})

    for i in range(latent_size):
        var = calcVariance(encoded[0], i)
        ax = fig.add_subplot(2, latent_size, latent_size+i+1, projection="3d")
        ax.scatter(
            sample[:, 0],
            sample[:, 1],
            decoded[0],
            #cmap=cm.jet,
            c=var,
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
        plt.title(f"Variance from latent variable {i+1}", fontdict={'fontsize':14, "color":"#ffffff"})
    #plt.savefig("bbob.png", bbox_inches='tight', transparent=True)
    return fig

def predict_ls(f,i):
    fun, opt = bbob.instantiate(f, i)
    bbob_y = np.asarray(list(map(fun, obj.sample)))
    ori = (bbob_y.flatten() - np.min(bbob_y)) / (
        np.max(bbob_y) - np.min(bbob_y)
    )
    encoded = obj.encode([ori])
    decoded = obj.autoencoder.decoder(encoded).numpy()
    return plotReconstructionInteractive(obj.sample, decoded, ori, encoded)


import gradio as gr



demo = gr.Blocks()

with demo:
    gr.Markdown(
        """# DoE2Vec
        Choose a BBOB function and instance, and see the highlighted effect of each latent variable.
        """
    )

    with gr.Row():
        f = gr.Slider(1, 24, label='BBOB function id', step=1, value=1)
        i = gr.Slider(1, 40, label='BBOB instance id', step=1, value=1)

    output = gr.Plot()
    btn = gr.Button(value="Visualise")
    btn.click(predict_ls, [f, i], output)

if __name__ == "__main__":
    demo.launch()

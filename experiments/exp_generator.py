import bbobbenchmarks as bbob
from doe2vec import doe_model
import numpy as np
import os
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import cross_val_score
from sklearn.metrics import multilabel_confusion_matrix
import matplotlib.pyplot as plt
from matplotlib import cm
import copy

from sklearn.metrics import confusion_matrix, classification_report
from matplotlib import pyplot as plt
import seaborn as sns
import matplotlib.animation as animation




"""Generation experiment
"""



def generateMovie(f, i, model_type, latent_size, frn  = 50):
    obj = doe_model(2, 8, n=20000, latent_dim=latent_size, use_mlflow=False, model_type=model_type, kl_weight=0.001)
    if not obj.load("../models/"):
        obj.generateData()
        obj.compile()
        obj.fit(100)
        obj.save("../models/")

    sample = obj.sample * 10 - 5

    fun, opt = bbob.instantiate(f,i)
    bbob_y =  np.asarray(list(map(fun, sample)))
    array_x = (bbob_y.flatten() - np.min(bbob_y)) / (
                np.max(bbob_y) - np.min(bbob_y)
            )
    encoded = obj.encode([array_x])[0]
    print(encoded)

    #[ 0.06200744 -1.4341657   3.165122    0.8527643 ]
    frames = []
    fig = plt.figure(figsize=(4, 4))

    # display reconstruction
    ax = fig.add_subplot(1, 1, 1, projection="3d")

    def update_plot(frame_number, frames, titles, plot, sample ):
        plot[0].remove()
        plot[0] = ax.plot_trisurf(
            sample[:, 0],
            sample[:, 1],
            frames[frame_number],
            cmap=cm.jet,
            antialiased=True,
        )
        ax.set_title(titles[frame_number])
        ax.xaxis.set_ticklabels([])
        ax.yaxis.set_ticklabels([])
        ax.zaxis.set_ticklabels([])

        for line in ax.xaxis.get_ticklines():
            line.set_visible(False)
        for line in ax.yaxis.get_ticklines():
            line.set_visible(False)
        for line in ax.zaxis.get_ticklines():
            line.set_visible(False)
        #if frame_number%120 == 0:
        #    plt.savefig("AE/frame"+str(frame_number)+".png")

    titles = []
    for index in range(latent_size):
        print(index)
        for x in np.linspace(encoded[index], encoded[index]+1.0, num=frn):
            encoded_x = copy.deepcopy(encoded)
            encoded_x[index] = x
            decoded = obj.autoencoder.decoder(np.array([encoded_x])).numpy()[0]
            titles.append(f"Varying latent variable {(index+1)} of {latent_size}")
            frames.append(decoded)
        for x in np.linspace(encoded[index]+1.0, encoded[index], num=frn):
            encoded_x = copy.deepcopy(encoded)
            encoded_x[index] = x
            decoded = obj.autoencoder.decoder(np.array([encoded_x])).numpy()[0]
            titles.append(f"Varying latent variable {(index+1)} of {latent_size}")
            frames.append(decoded)
        for x in np.linspace(encoded[index], encoded[index]-1.0, num=frn):
            encoded_x = copy.deepcopy(encoded)
            encoded_x[index] = x
            decoded = obj.autoencoder.decoder(np.array([encoded_x])).numpy()[0]
            titles.append(f"Varying latent variable {(index+1)} of {latent_size}")
            frames.append(decoded)
        for x in np.linspace(encoded[index]-1.0, encoded[index], num=frn):
            encoded_x = copy.deepcopy(encoded)
            encoded_x[index] = x
            decoded = obj.autoencoder.decoder(np.array([encoded_x])).numpy()[0]
            titles.append(f"Varying latent variable {(index+1)} of {latent_size}")
            frames.append(decoded)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    plot = [ax.plot_trisurf(
            sample[:, 0],
            sample[:, 1],
            frames[0],
            cmap=cm.jet,
            antialiased=True,
        )]
    ax.xaxis.set_ticklabels([])
    ax.yaxis.set_ticklabels([])
    ax.zaxis.set_ticklabels([])

    for line in ax.xaxis.get_ticklines():
        line.set_visible(False)
    for line in ax.yaxis.get_ticklines():
        line.set_visible(False)
    for line in ax.zaxis.get_ticklines():
        line.set_visible(False)
    #ax.set_zlim(0,1.1)
    fps = 40
    ani = animation.FuncAnimation(fig, update_plot, len(frames), fargs=(frames, titles, plot, sample), interval=1000/fps)


    # ani.save('movie.mp4')
    #plt.show()

    fn = f'gifs/{model_type}_2d_reconstruction_{latent_size}_f{f}_i{i}'
    ani.save(fn+'.mp4',writer='ffmpeg',fps=fps)
    #ani.save(fn+'.gif',writer='imagemagick',fps=fps)

i = 0
for latent_size in [4]: #8
    for f in [22]:#11,12,13,14,15,24
        for model_type in ["AE"]:#, "AE"
            generateMovie(f,i,model_type, latent_size,frn=120)

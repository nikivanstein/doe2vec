from modulesRandFunc import generate_tree as genTree
from modulesRandFunc import generate_tree2exp as genTree2exp
from modulesRandFunc import generate_exp2fun as genExp2fun
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import cm
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, losses
from tensorflow.keras.models import Model
from scipy.stats import qmc
import numpy as np

d = 2
m = 8 #power of 2 for sample size
seed = 0

class Autoencoder(Model):
  def __init__(self, latent_dim, sample_size):
    super(Autoencoder, self).__init__()
    self.latent_dim = latent_dim   
    self.encoder = tf.keras.Sequential([
      layers.Dense(sample_size / 2, activation='relu'),
      layers.Dense(sample_size / 4, activation='relu'),
      layers.Dense(latent_dim, activation='relu'),
    ])
    self.decoder = tf.keras.Sequential([
      layers.Dense(sample_size / 4, activation='relu'),
      layers.Dense(sample_size / 2, activation='relu'),
      layers.Dense(sample_size, activation='sigmoid')
    ])

  def call(self, x):
    encoded = self.encoder(x)
    decoded = self.decoder(encoded)
    return decoded

class Doe2Vec():
    def __init__(self, dim, m, n=1000, latent_dim=32, seed=0):
        self.dim = dim
        self.m = m
        self.n = n
        self.latent_dim = latent_dim
        self.seed = seed
        #generate the DOE using Sobol
        self.sampler = qmc.Sobol(d=self.dim, scramble=False, seed = self.seed)
        self.sample = self.sampler.random_base2(m=self.m)


    def generateData(self):
        array_x = self.sample #it is required to be named array_x for the eval
        self.Y = []
        tries = 0
        while len(self.Y) < self.n:
            tries += 1
            # create an artificial function
            tree = genTree.generate_tree(6, 12)
            exp = genTree2exp.generate_tree2exp(tree)
            fun = genExp2fun.generate_exp2fun(exp, len(self.sample), self.sample.shape[1])
            try:
                array_y = eval(fun)
                if (np.isnan(array_y).any() or np.isinf(array_y).any() or np.any(abs(array_y)<1e-8) or np.any(abs(array_y)>1e8) 
                    or np.var(array_y)<1.0 or array_y.ndim!=1):
                    continue
                #normalize the train data (this should be done per row (not per column!))
                array_y = array_y.flatten()
                array_y = (array_y - np.min(array_y)) / (np.max(array_y) - np.min(array_y))
                self.Y.append(array_y)
            except:
                continue
        self.Y = np.array(self.Y)
        self.train_data = tf.cast(self.Y[:-50], tf.float32)
        self.test_data = tf.cast(self.Y[-50:], tf.float32)
        return self.Y

    def compileAutoEncoder(self):
        self.autoencoder = Autoencoder(self.latent_dim, self.Y.shape[1])
        self.autoencoder.compile(optimizer='adam', loss=losses.MeanSquaredError())

    def train(self, epochs=50):
        self.autoencoder.fit(self.train_data, self.train_data,
                epochs=epochs,
                shuffle=True,
                validation_data=(self.test_data, self.test_data))

    def visualizeTestData(self, n=10):
        encoded_does = self.autoencoder.encoder(self.test_data).numpy()
        decoded_does = self.autoencoder.decoder(encoded_does).numpy()
        fig = plt.figure(figsize=(n*3, 8))
        for i in range(n):
            # display original
            ax = fig.add_subplot(2, n, i+1, projection='3d')
            ax.plot_trisurf(self.sample[:,0], self.sample[:,1], self.test_data[i],cmap=cm.jet, antialiased = True)
            ax.xaxis.set_ticklabels([])
            ax.yaxis.set_ticklabels([])
            ax.zaxis.set_ticklabels([])

            for line in ax.xaxis.get_ticklines():
                line.set_visible(False)
            for line in ax.yaxis.get_ticklines():
                line.set_visible(False)
            for line in ax.zaxis.get_ticklines():
                line.set_visible(False)
            plt.title("original")
            plt.gray()

            # display reconstruction
            ax = fig.add_subplot(2, n, i+1+n, projection='3d')
            ax.plot_trisurf(self.sample[:,0], self.sample[:,1],decoded_does[i],cmap=cm.jet,antialiased = True)
            ax.xaxis.set_ticklabels([])
            ax.yaxis.set_ticklabels([])
            ax.zaxis.set_ticklabels([])

            for line in ax.xaxis.get_ticklines():
                line.set_visible(False)
            for line in ax.yaxis.get_ticklines():
                line.set_visible(False)
            for line in ax.zaxis.get_ticklines():
                line.set_visible(False)
            plt.title("reconstructed")
            plt.gray()
        plt.tight_layout()
        plt.show()

obj = Doe2Vec(d, m, n=10000)
y = np.array(obj.generateData())
obj.compileAutoEncoder()
obj.train(50)
obj.visualizeTestData()
"""
TODO:
- optimize parameters of autoencoder
- check bbob functions, to find corresponding random function
- display reconstruction errors.
- perform for 2,5,10,15,20 dimensions
- fix seed!
"""
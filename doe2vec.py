import matplotlib.pyplot as plt
import numpy as np
import os.path
import pandas as pd
import tensorflow as tf
from matplotlib import cm
from mpl_toolkits import mplot3d
from scipy.stats import qmc
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, losses
from tensorflow.keras.models import Model
import mlflow
from numpy.random import seed
import mlflow.tensorflow
from sklearn.neighbors import NearestNeighbors

from modulesRandFunc import generate_exp2fun as genExp2fun
from modulesRandFunc import generate_tree as genTree
from modulesRandFunc import generate_tree2exp as genTree2exp


class Autoencoder(Model):
    def __init__(self, latent_dim, sample_size):
        super(Autoencoder, self).__init__()
        self.latent_dim = latent_dim
        self.encoder = tf.keras.Sequential(
            [
                layers.Dense(sample_size / 2, activation="relu"),
                layers.Dense(sample_size / 4, activation="relu"),
                layers.Dense(latent_dim, activation="relu"),
            ]
        )
        self.decoder = tf.keras.Sequential(
            [
                layers.Dense(sample_size / 4, activation="relu"),
                layers.Dense(sample_size / 2, activation="relu"),
                layers.Dense(sample_size, activation="sigmoid"),
            ]
        )

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


class Doe2Vec:
    def __init__(self, dim, m, n=1000, latent_dim=32, seed_nr=0, use_mlflow=True, mlflow_name="Doc2Vec"):
        self.dim = dim
        self.m = m
        self.n = n
        self.latent_dim = latent_dim
        self.seed = seed_nr
        seed(self.seed)
        # generate the DOE using Sobol
        self.sampler = qmc.Sobol(d=self.dim, scramble=False, seed=self.seed)
        self.sample = self.sampler.random_base2(m=self.m)
        self.use_mlflow = use_mlflow
        if use_mlflow:
            mlflow.set_tracking_uri("mlflow/")
            mlflow.set_experiment(mlflow_name)
            mlflow.start_run(run_name=f"run {self.dim}-{self.m}-{self.latent_dim}-{self.seed}")
            mlflow.log_param("dim", self.dim)
            mlflow.log_param("m", self.m)
            mlflow.log_param("latent_dim", self.latent_dim)
            mlflow.log_param("seed", self.seed)

    def load(self, dir="models"):
        if (os.path.exists(f"{dir}/sample_{self.dim}-{self.m}-{self.latent_dim}-{self.seed}.npy")):
            self.autoencoder = tf.keras.models.load_model(
                f"{dir}/model_{self.dim}-{self.m}-{self.latent_dim}-{self.seed}"
            )
            self.sample = np.load(
                f"{dir}/sample_{self.dim}-{self.m}-{self.latent_dim}-{self.seed}.npy"
            )
            self.Y = np.load(
                f"{dir}/data_{self.dim}-{self.m}-{self.latent_dim}-{self.seed}.npy"
            )
            self.functions = np.load(
                f"{dir}/functions_{self.dim}-{self.m}-{self.latent_dim}-{self.seed}.npy"
            )
            self.train_data = tf.cast(self.Y[:-50], tf.float32)
            self.test_data = tf.cast(self.Y[-50:], tf.float32)
            print("Loaded pre-existng model and data")
            self.summary()
            self.fitNN()
            return True
        else:
            return False

    def getSample(self):
        return self.sample

    def generateData(self):
        array_x = self.sample  # it is required to be named array_x for the eval
        self.Y = []
        self.functions = []
        tries = 0
        while len(self.Y) < self.n:
            tries += 1
            # create an artificial function
            tree = genTree.generate_tree(6, 12)
            exp = genTree2exp.generate_tree2exp(tree)
            fun = genExp2fun.generate_exp2fun(
                exp, len(self.sample), self.sample.shape[1]
            )
            try:
                array_y = eval(fun)
                if (
                    np.isnan(array_y).any()
                    or np.isinf(array_y).any()
                    or np.any(abs(array_y) < 1e-8)
                    or np.any(abs(array_y) > 1e8)
                    or np.var(array_y) < 1.0
                    or array_y.ndim != 1
                ):
                    continue
                # normalize the train data (this should be done per row (not per column!))
                array_y = array_y.flatten()
                array_y = (array_y - np.min(array_y)) / (
                    np.max(array_y) - np.min(array_y)
                )
                self.functions.append(fun)
                self.Y.append(array_y)
            except:
                continue
        self.Y = np.array(self.Y)
        self.functions = np.array(self.functions)
        self.train_data = tf.cast(self.Y[:-50], tf.float32)
        self.test_data = tf.cast(self.Y[-50:], tf.float32)
        return self.Y

    def setData(self,Y):
        self.Y = Y
        self.train_data = tf.cast(self.Y[:-50], tf.float32)
        self.test_data = tf.cast(self.Y[-50:], tf.float32)

    def compile(self):
        self.autoencoder = Autoencoder(self.latent_dim, self.Y.shape[1])
        self.autoencoder.compile(optimizer="adam", loss=losses.MeanSquaredError())

    def fit(self, epochs=100):
        if self.use_mlflow:
            mlflow.tensorflow.autolog(every_n_iter=1)
        self.autoencoder.fit(
            self.train_data,
            self.train_data,
            epochs=epochs,
            batch_size=128,
            shuffle=True,
            validation_data=(self.test_data, self.test_data),
        )
        self.fitNN()
        if self.use_mlflow:
            self.visualizeTestData()
            mlflow.end_run()

    def fitNN(self):
        self.encoded_Y = self.encode(self.Y)
        self.nn = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(self.encoded_Y)

    def getNeighbourFunction(self, features):
        distances, indices = self.nn.kneighbors(features)
        return self.functions[indices[0]][0], distances[0]

    def save(self, dir="models"):
        # Save the model, sample and data set
        self.autoencoder.save(
            f"{dir}/model_{self.dim}-{self.m}-{self.latent_dim}-{self.seed}"
        )
        np.save(
            f"{dir}/sample_{self.dim}-{self.m}-{self.latent_dim}-{self.seed}.npy",
            self.sample,
        )
        np.save(
            f"{dir}/data_{self.dim}-{self.m}-{self.latent_dim}-{self.seed}.npy", self.Y
        )
        np.save(
            f"{dir}/functions_{self.dim}-{self.m}-{self.latent_dim}-{self.seed}.npy", self.functions
        )

    def encode(self, X, return_error=False):
        X = tf.cast(X, tf.float32)
        encoded_doe = self.autoencoder.encoder(X).numpy()
        if return_error:
            enc = tf.cast(np.array(encoded_doe), tf.float32)
            decoded_doe = self.autoencoder.decoder(enc).numpy()
            # todo: return reconstruction error
        return encoded_doe

    def summary(self):
        self.autoencoder.summary()

    def visualizeTestData(self, n=5):
        encoded_does = self.autoencoder.encoder(self.test_data).numpy()
        decoded_does = self.autoencoder.decoder(encoded_does).numpy()
        fig = plt.figure(figsize=(n * 4, 8))
        for i in range(n):
            # display original
            ax = fig.add_subplot(2, n, i + 1, projection="3d")
            ax.plot_trisurf(
                self.sample[:, 0],
                self.sample[:, 1],
                self.test_data[i],
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
            plt.title("original")
            plt.gray()

            # display reconstruction
            ax = fig.add_subplot(2, n, i + 1 + n, projection="3d")
            ax.plot_trisurf(
                self.sample[:, 0],
                self.sample[:, 1],
                decoded_does[i],
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
            plt.title("reconstructed")
            plt.gray()
        if self.use_mlflow:
            plt.savefig("test.png")
            mlflow.log_artifact("test.png", "img")
        else:
            plt.show()


if __name__ == "__main__":
    for d in [2,5,10]:
        for m in [8,9,10]:
            for latent_dim in [8,16,24]:
                obj = Doe2Vec(d, m, n=d*50000, latent_dim=latent_dim)
                #if not obj.load():
                obj.generateData()
                obj.compile()
                obj.fit(100)
                obj.save()
"""
TODO:
- optimize parameters of autoencoder
- check bbob functions, to find corresponding random function
- display reconstruction errors.
- perform for 2-20 dimensions
- fix seed!
"""

import os.path

import matplotlib.pyplot as plt
import mlflow.tensorflow
import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib import cm
from mpl_toolkits import mplot3d
from numpy.random import seed
from scipy.stats import qmc
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from tensorflow.keras import layers, losses
from tensorflow.keras.models import Model

import mlflow
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


class doe_model:
    def __init__(
        self,
        dim,
        m,
        n=1000,
        latent_dim=16,
        seed_nr=0,
        custom_sample=None,
        use_mlflow=False,
        mlflow_name="Doc2Vec",
    ):
        """Doe2Vec model to transform Design of Experiments to feature vectors.

        Args:
            dim (int): Number of dimensions of the DOE
            m (int): Power for number of samples used in the Sobol sampler (not used for custom_sample)
            n (int, optional): Number of generated functions to use a training data. Defaults to 1000.
            latent_dim (int, optional): Number of dimensions in the latent space (vector size). Defaults to 16.
            seed_nr (int, optional): Random seed. Defaults to 0.
            custom_sample (array, optional): dim-d Array with a custom sample or None to use Sobol sequences. Defaults to None.
            use_mlflow (bool, optional): To use the mlflow backend to log experiments. Defaults to False.
            mlflow_name (str, optional): The name to log the mlflow experiment. Defaults to "Doc2Vec".
        """
        self.dim = dim
        self.m = m
        self.n = n
        self.latent_dim = latent_dim
        self.seed = seed_nr
        seed(self.seed)
        # generate the DOE using Sobol
        if custom_sample is None:
            self.sampler = qmc.Sobol(d=self.dim, scramble=False, seed=self.seed)
            self.sample = self.sampler.random_base2(m=self.m)
        else:
            self.sample = custom_sample
        self.use_mlflow = use_mlflow
        if use_mlflow:
            mlflow.set_tracking_uri("mlflow/")
            mlflow.set_experiment(mlflow_name)
            mlflow.start_run(
                run_name=f"run {self.dim}-{self.m}-{self.latent_dim}-{self.seed}"
            )
            mlflow.log_param("dim", self.dim)
            mlflow.log_param("m", self.m)
            mlflow.log_param("latent_dim", self.latent_dim)
            mlflow.log_param("seed", self.seed)

    def load(self, dir="models"):
        """Load a pre-trained Doe2vec model and data.

        Args:
            dir (str, optional): The directory where the model and data are stored. Defaults to "models".

        Returns:
            bool: True if loaded, else False.
        """
        if os.path.exists(
            f"{dir}/sample_{self.dim}-{self.m}-{self.latent_dim}-{self.seed}.npy"
        ):
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
        """Get the sample DOE used.

        Returns:
            array: Sample
        """
        return self.sample

    def generateData(self):
        """Generate the random functions for training the autoencoder.

        Returns:
            array: array with evaluated random functions on sample.
        """
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

    def setData(self, Y):
        """Helper function to load the data and split in train validation sets.

        Args:
            Y (nd array): the data set to use.
        """
        self.Y = Y
        self.train_data = tf.cast(self.Y[:-50], tf.float32)
        self.test_data = tf.cast(self.Y[-50:], tf.float32)

    def compile(self):
        """Compile the autoencoder architecture."""
        self.autoencoder = Autoencoder(self.latent_dim, self.Y.shape[1])
        self.autoencoder.compile(optimizer="adam", loss=losses.MeanSquaredError())

    def fit(self, epochs=100):
        """Fit the autoencoder model.

        Args:
            epochs (int, optional): Number of epochs to train. Defaults to 100.
        """
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
        """Fit the neirest neighbour tree."""
        self.encoded_Y = self.encode(self.Y)
        self.nn = NearestNeighbors(n_neighbors=1, algorithm="ball_tree").fit(
            self.encoded_Y
        )

    def getNeighbourFunction(self, features):
        """Get the closest random generated function depending on a set of features (from another function).

        Args:
            features (array): Feature vector (given by the encode() function)

        Returns:
            tuple: random function string, distance
        """
        distances, indices = self.nn.kneighbors(features)
        return self.functions[indices[0]][0], distances[0]

    def save(self, dir="models"):
        """Save the model, sample and data set

        Args:
            dir (str, optional): Directory to store the data. Defaults to "models".
        """
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
            f"{dir}/functions_{self.dim}-{self.m}-{self.latent_dim}-{self.seed}.npy",
            self.functions,
        )

    def encode(self, X):
        """Encode a Design of Experiments.

        Args:
            X (array): The DOE to encode.

        Returns:
            array: encoded feature vector.
        """
        X = tf.cast(X, tf.float32)
        encoded_doe = self.autoencoder.encoder(X).numpy()
        return encoded_doe

    def summary(self):
        """Get a summary of the autoencoder model"""
        self.autoencoder.encoder.summary()

    def visualizeTestData(self, n=5):
        """Get a visualisation of the validation data.

        Args:
            n (int, optional): The number of validation DOEs to show. Defaults to 5.
        """
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
    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    for d in [2, 5, 10]:
        for m in [8, 9, 10]:
            for latent_dim in [8, 16, 24]:
                obj = doe_model(d, m, n=d * 50000, latent_dim=latent_dim)
                if not obj.load("../models/"):
                    obj.generateData()
                    obj.compile()
                    obj.fit(100)
                    obj.save("../models/")
                obj.autoencoder.push_to_hub(f"doe2vec-d{d}-m{m}-ls{latent_dim}")

import os.path
import sys
import warnings
from statistics import mode

import matplotlib.pyplot as plt
import mlflow
import mlflow.tensorflow
import numpy as np
import pandas as pd
import tensorflow as tf
from datasets import load_dataset
from huggingface_hub import from_pretrained_keras
from matplotlib import cm
from mpl_toolkits import mplot3d
from numpy.random import seed
from scipy.stats import qmc
from sklearn import manifold
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from tensorflow.keras import layers, losses
from tensorflow.keras.models import Model

import bbobbenchmarks as bbob
from models import VAE, Autoencoder
from modulesRandFunc import generate_exp2fun as genExp2fun
from modulesRandFunc import generate_tree as genTree
from modulesRandFunc import generate_tree2exp as genTree2exp


class doe_model:
    def __init__(
        self,
        dim,
        m,
        n=250000,
        latent_dim=32,
        seed_nr=0,
        kl_weight=0.001,
        custom_sample=None,
        use_mlflow=False,
        mlflow_name="Doc2Vec",
        model_type="VAE",
    ):
        """Doe2Vec model to transform Design of Experiments to feature vectors.

        Args:
            dim (int): Number of dimensions of the DOE
            m (int): Power for number of samples used in the Sobol sampler (not used for custom_sample)
            n (int, optional): Number of generated functions to use a training data. Defaults to 1000.
            latent_dim (int, optional): Number of dimensions in the latent space (vector size). Defaults to 16.
            seed_nr (int, optional): Random seed. Defaults to 0.
            kl_weight (float, optional): Defaults to 0.1.
            custom_sample (array, optional): dim-d Array with a custom sample or None to use Sobol sequences. Defaults to None.
            use_mlflow (bool, optional): To use the mlflow backend to log experiments. Defaults to False.
            mlflow_name (str, optional): The name to log the mlflow experiment. Defaults to "Doc2Vec".
            model_type (str, optional): The model to use, either "AE" or "VAE". Defaults to "VAE".
            use_bbob (bool, optional): To use BBOB functions in addition to random generated functions.
        """
        self.dim = dim
        self.m = m
        self.n = n
        self.kl_weight = kl_weight
        self.latent_dim = latent_dim
        self.seed = seed_nr
        self.use_VAE = False
        self.model_type = model_type
        self.fitted = False
        self.autoencoder = None
        if model_type == "VAE":
            self.use_VAE = True
            self.pure_model_type = self.model_type
            self.model_type = self.model_type + str(kl_weight)
        seed(self.seed)
        # generate the DOE using Sobol
        if custom_sample is None:
            self.sampler = qmc.Sobol(d=self.dim, scramble=False, seed=self.seed)
            self.sample = self.sampler.random_base2(m=self.m)
        else:
            self.sample = custom_sample
        self.use_mlflow = use_mlflow
        if use_mlflow:
            mlflow.set_experiment(mlflow_name)
            mlflow.start_run(
                run_name=f"run {self.dim}-{self.m}-{self.latent_dim}-{self.seed}"
            )
            mlflow.log_param("model_type", model_type)
            mlflow.log_param("dim", self.dim)
            mlflow.log_param("kl_weight", self.kl_weight)
            mlflow.log_param("m", self.m)
            mlflow.log_param("latent_dim", self.latent_dim)
            mlflow.log_param("seed", self.seed)

    def load_from_huggingface(self, repo="BasStein"):
        """Load a pre-trained model from a HuggingFace repository.

        Args:
            repo (str, optional): the huggingface repo to load from.
        """
        model_name = f"{repo}/doe2vec-d{self.dim}-m{self.m}-ls{self.latent_dim}-{self.pure_model_type}-kl{self.kl_weight}"
        data_name = f"{repo}/{self.n}-randomfunctions-{self.dim}d"
        dataset = load_dataset(data_name)["train"]
        self.autoencoder = from_pretrained_keras(model_name)
        self.autoencoder.compile(optimizer="adam")
        self.functions = dataset["function"]
        self.Y = []
        array_x = (
            self.sample
        )  # this seemingly unused variable is required by the eval() later on
        if not sys.warnoptions:
            warnings.simplefilter("ignore")
        for fun in self.functions:
            try:
                array_y = eval(fun)
                # normalize the train data (this should be done per row (not per column!))
                array_y = array_y.flatten()
                array_y = (array_y - np.min(array_y)) / (
                    np.max(array_y) - np.min(array_y)
                )
                self.Y.append(array_y)
            except:
                continue
        warnings.simplefilter("default")
        self.setData(self.Y)
        print("Loaded huggingface model and data")
        self.summary()
        self.fitNN()
        return True

    def loadModel(self, dir="models"):
        """Load a pre-trained Doe2vec model.

        Args:
            dir (str, optional): The directory where the model is stored. Defaults to "models".

        Returns:
            bool: True if loaded, else False.
        """
        if os.path.exists(
            f"{dir}/model_{self.dim}-{self.m}-{self.latent_dim}-{self.seed}-{self.model_type}"
        ):
            self.autoencoder = tf.keras.models.load_model(
                f"{dir}/model_{self.dim}-{self.m}-{self.latent_dim}-{self.seed}-{self.model_type}"
            )
            print("Loaded pre-existng model")
            self.summary()
            return True
        return False

    def loadData(self, dir="data"):
        """Load a stored functions file and retrieve all the landscapes.

        Args:
            dir (str, optional): The directory where the data are stored. Defaults to "data".

        Returns:
            bool: True if loaded, else False.
        """
        if os.path.exists(f"{dir}/functions_d{self.dim}-n{self.n}.npy"):
            self.functions = np.load(f"{dir}/functions_d{self.dim}-n{self.n}.npy")
            self.Y = []
            array_x = (
                self.sample
            )  # this seemingly unused variable is required by the eval() later on
            if not sys.warnoptions:
                warnings.simplefilter("ignore")
            for fun in self.functions:
                try:
                    array_y = eval(fun)
                    # normalize the train data (this should be done per row (not per column!))
                    array_y = array_y.flatten()
                    array_y = (array_y - np.min(array_y)) / (
                        np.max(array_y) - np.min(array_y)
                    )
                    self.Y.append(array_y)
                except:
                    continue
            warnings.simplefilter("default")
            self.setData(self.Y)
            print("Loaded data")
            return True
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
        if not sys.warnoptions:
            warnings.simplefilter("ignore")
        while len(self.Y) < self.n:
            tries += 1
            # create an artificial function
            tree = genTree.generate_tree(6, 16)
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
        warnings.simplefilter("default")
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
        self.Y = np.array(Y)
        self.train_data = tf.cast(self.Y[:-50], tf.float32)
        self.test_data = tf.cast(self.Y[-50:], tf.float32)

    def compile(self):
        """Compile the autoencoder architecture."""
        if self.use_VAE:
            self.autoencoder = VAE(
                self.latent_dim, self.Y.shape[1], kl_weight=self.kl_weight
            )
            self.autoencoder.compile(optimizer="adam")
        else:
            self.autoencoder = Autoencoder(self.latent_dim, self.Y.shape[1])
            self.autoencoder.compile(optimizer="adam", loss=losses.MeanSquaredError())

    def fit(self, epochs=100, **kwargs):
        """Fit the autoencoder model.

        Args:
            epochs (int, optional): Number of epochs to train. Defaults to 100.
            **kwargs (dict, optional): optional arguments for the fit procedure.
        """
        if self.autoencoder is None:
            raise AttributeError("Autoencoder model is not compiled yet")
        if self.use_mlflow:
            mlflow.tensorflow.autolog(every_n_iter=1)
        if self.use_VAE:
            self.autoencoder.fit(
                self.train_data,
                epochs=epochs,
                batch_size=128,
                shuffle=True,
                validation_data=(self.test_data, self.test_data),
                **kwargs,
            )
        else:
            self.autoencoder.fit(
                self.train_data,
                self.train_data,
                epochs=epochs,
                batch_size=128,
                shuffle=True,
                validation_data=(self.test_data, self.test_data),
                **kwargs,
            )
        self.fitNN()
        if self.use_mlflow:
            self.plot_label_clusters_bbob()
            self.visualizeTestData()
            mlflow.end_run()

    def fitNN(self):
        """Fit the neirest neighbour tree to find similar functions."""
        self.fitted = True
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
        if self.fitted == False:
            self.fitNN()
        distances, indices = self.nn.kneighbors(features)
        return self.functions[indices[0]][0], distances[0]

    def save(self, model_dir="model", data_dir="data"):
        """Save the model and random functions used for training

        Args:
            model_dir (str, optional): Directory to store the model. Defaults to "model".
            data_dir (str, optional): Directory to store the random functions. Defaults to "data".
        """
        self.saveModel(model_dir=model_dir)
        self.saveData(data_dir=data_dir)
        return True

    def saveModel(self, model_dir):
        """Save the model

        Args:
            model_dir (str, optional): Directory to store the model. Defaults to "model".
        """
        self.autoencoder.save(
            f"{model_dir}/model_{self.dim}-{self.m}-{self.latent_dim}-{self.seed}-{self.model_type}"
        )

    def saveData(self, data_dir="data"):
        """Save the random functions used for training

        Args:
            data_dir (str, optional): Directory to store the random functions. Defaults to "data".
        """
        np.save(f"{data_dir}/functions_d{self.dim}-n{self.n}.npy", self.functions)

    def encode(self, X):
        """Encode a Design of Experiments.

        Args:
            X (array): The DOE to encode.

        Returns:
            array: encoded feature vector.
        """
        X = tf.cast(X, tf.float32)
        if self.use_VAE:
            encoded_doe, _, __ = self.autoencoder.encoder(X)
            encoded_doe = np.array(encoded_doe)
        else:
            encoded_doe = self.autoencoder.encoder(X).numpy()
        return encoded_doe

    def summary(self):
        """Get a summary of the autoencoder model"""
        self.autoencoder.encoder.summary()

    def plot_label_clusters_bbob(self):
        encodings = []
        fuction_groups = []
        for f in range(1, 25):
            for i in range(100):
                fun, opt = bbob.instantiate(f, i)
                bbob_y = np.asarray(list(map(fun, self.sample)))
                array_x = (bbob_y.flatten() - np.min(bbob_y)) / (
                    np.max(bbob_y) - np.min(bbob_y)
                )
                encoded = self.encode([array_x])
                encodings.append(encoded[0])
                fuction_groups.append(f)

        X = np.array(encodings)
        y = np.array(fuction_groups).flatten()
        mds = manifold.MDS(
            n_components=2,
            random_state=self.seed,
        )
        embedding = mds.fit_transform(X).T
        # display a 2D plot of the bbob functions in the latent space

        plt.figure(figsize=(12, 10))
        plt.scatter(embedding[0], embedding[1], c=y, cmap=cm.jet)
        plt.colorbar()
        plt.xlabel("")
        plt.ylabel("")

        if self.use_mlflow:
            plt.savefig("latent_space.png")
            mlflow.log_artifact("latent_space.png", "img")
        else:
            plt.savefig(
                f"latent_space_{self.dim}-{self.m}-{self.latent_dim}-{self.seed}-{self.model_type}.png"
            )

    def visualizeTestData(self, n=5):
        """Get a visualisation of the validation data.

        Args:
            n (int, optional): The number of validation DOEs to show. Defaults to 5.
        """
        if self.use_VAE:
            encoded_does, _z_log_var, _z = self.autoencoder.encoder(self.test_data)
        else:
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
            plt.savefig("reconstruction.png")
            mlflow.log_artifact("reconstruction.png", "img")
        else:
            plt.show()


if __name__ == "__main__":
    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    obj = doe_model(
        2,
        8,
        n=50000,
        latent_dim=16,
        kl_weight=0.001,
        use_mlflow=False,
        model_type="VAE",
    )
    obj.load_from_huggingface()
    # test the model
    obj.plot_label_clusters_bbob()

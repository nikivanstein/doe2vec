from random import sample

import tensorflow as tf
from tensorflow.keras import layers, losses
from tensorflow.keras.models import Model
import numpy as np
from keras.layers import Dense, Input, Concatenate, Lambda
from keras.utils.vis_utils import plot_model
from sklearn.metrics import pairwise_distances
import keras.backend as K

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

class CustomConnected(Dense):

    def __init__(self,units,connections,**kwargs):

        #this is matrix A
        self.connections = connections                        

        #initalize the original Dense with all the usual arguments   
        super(CustomConnected,self).__init__(units,**kwargs)  

    def call(self, inputs):
        output = K.dot(inputs, self.kernel * self.connections)
        if self.use_bias:
            output = K.bias_add(output, self.bias)
        if self.activation is not None:
            output = self.activation(output)
        return output

class CustomAutoencoder(Model):
    def __init__(self, latent_dim, sample_size, DOE):
        super(CustomAutoencoder, self).__init__()
        self.latent_dim = latent_dim
        self.DOE = DOE
        self.dim = self.DOE.shape[1]
        self.sample_size = sample_size
        self.encoder = self._encoder()
        self.decoder = tf.keras.Sequential(
            [
                layers.Dense(sample_size / 4, activation="relu"),
                layers.Dense(sample_size / 2, activation="relu"),
                layers.Dense(sample_size, activation="sigmoid"),
            ]
        )


    def _encoder(self):
        '''Create a Dense network with shape information from the DOE'''
        #we use knowledge of the space filling design to determine the distance threshold
        inputTensor = Input((self.sample_size,))
        sorted_DOE = np.argsort(self.DOE, axis=0)
        print("sorted DOE", sorted_DOE)
        connections = np.zeros((self.sample_size,self.sample_size))
        pair_distances = pairwise_distances(self.DOE, metric='cityblock')
        print("pair_distances", pair_distances)
        for i in range(0,len(self.DOE)):
            indexes_to_use = np.argsort(pair_distances[i,:])[:self.dim * 2]
            connections[indexes_to_use, i] = 1
            connections[i, i] = 1
        tf_connections = tf.convert_to_tensor(connections, dtype=tf.float32)
        print("connections",connections)
        #x = Concatenate()(groups)
        x = CustomConnected(self.sample_size, tf_connections, activation="relu")(inputTensor)
        x = Dense(self.sample_size / 2, activation="relu")(x)
        x = Dense(self.sample_size / 4, activation="relu")(x)
        x = Dense(self.latent_dim, activation="relu")(x)
        encoder = tf.keras.Model(inputTensor, x, name="encoder")
        encoder.summary()
        return encoder

    def plot(self):
        plot_model(self.encoder, to_file='encoder.png', show_shapes=True, show_layer_names=True)
        plot_model(self.decoder, to_file='decoder.png', show_shapes=True, show_layer_names=True)

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a DOE."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


class VAE(Model):
    """Variational autoencoder

    Args:
        keras (_type_): _description_
    """

    def __init__(self, latent_dim, sample_size, kl_weight=0.1):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim
        self.sample_size = sample_size
        self.encoder = self._encoder()
        self.decoder = self._decoder()
        self.kl_weight = kl_weight
        self.total_loss_tracker = tf.keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = tf.keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = tf.keras.metrics.Mean(name="kl_loss")

    def _encoder(self):
        encoder_inputs = tf.keras.Input(shape=(self.sample_size,))
        x = layers.Dense(self.sample_size / 2, activation="relu")(encoder_inputs)
        x = layers.Dense(self.sample_size / 4, activation="relu")(x)
        z_mean = layers.Dense(self.latent_dim, name="z_mean")(x)
        z_log_var = layers.Dense(self.latent_dim, name="z_log_var")(x)
        z = Sampling()([z_mean, z_log_var])
        encoder = tf.keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
        encoder.summary()
        return encoder

    def _decoder(self):
        latent_inputs = tf.keras.Input(shape=(self.latent_dim,))
        x = layers.Dense(self.sample_size / 4, activation="relu")(latent_inputs)
        x = layers.Dense(self.sample_size / 2, activation="relu")(x)
        decoder_outputs = layers.Dense(self.sample_size, activation="sigmoid")(x)
        decoder = tf.keras.Model(latent_inputs, decoder_outputs, name="decoder")
        decoder.summary()
        return decoder

    def call(self, x, training=False):
        z_mean, z_log_var, z = self.encoder(x, training=training)
        decoded = self.decoder(z, training=training)
        return z_mean, z_log_var, z, decoded

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z, reconstruction = self(data, training=True)
            reconstruction_loss = tf.reduce_mean(
                tf.square(data - reconstruction)
            )  # MeanSquaredError
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(kl_loss)
            total_loss = reconstruction_loss + self.kl_weight * kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }

    def test_step(self, data):
        d, v = data
        z_mean, z_log_var, z, reconstruction = self(d)
        reconstruction_loss = tf.reduce_mean(
            tf.square(d - reconstruction)
        )  # MeanSquaredError
        kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
        kl_loss = tf.reduce_mean(kl_loss)
        total_loss = reconstruction_loss + self.kl_weight * kl_loss
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }


if __name__ == "__main__":
    import os
    from scipy.stats import qmc
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    sampler = qmc.Sobol(d=2, scramble=False, seed=0)
    sample = sampler.random_base2(m=3) #should create 4 samples
    print(sample)
    model = CustomAutoencoder(2,len(sample),sample)
    model.compile(optimizer="adam")
    model.build(input_shape=(2,sample.shape[0]))
    model.summary()
    model.plot()
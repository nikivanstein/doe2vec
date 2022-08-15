from random import sample

import tensorflow as tf
from tensorflow.keras import layers, losses
from tensorflow.keras.models import Model


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

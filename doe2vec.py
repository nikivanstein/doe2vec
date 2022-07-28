from modulesRandFunc import generate_tree as genTree
from modulesRandFunc import generate_tree2exp as genTree2exp
from modulesRandFunc import generate_exp2fun as genExp2fun
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, losses
from tensorflow.keras.models import Model
from scipy.stats import qmc
import numpy as np

d = 2
m = 8 #power of 2 for sample size
seed = 42

#first generate one DOE using SOBOL
sampler = qmc.Sobol(d=d, scramble=False, seed = seed)
sample = sampler.random_base2(m=m)

# create an artificial function
tree = genTree.generate_tree(8, 12)
exp = genTree2exp.generate_tree2exp(tree)
fun = genExp2fun.generate_exp2fun(exp, len(sample), sample.shape[1])

# skip if function generation failed

array_x = sample
array_y = eval(fun)
print(array_y.shape)

# END fun


# AUTOENCODER part


latent_dim = 32

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

autoencoder = Autoencoder(latent_dim, len(array_y))
autoencoder.compile(optimizer='adam', loss=losses.MeanSquaredError())

#normalize the train data

"""
min_val = tf.reduce_min(train_data)
max_val = tf.reduce_max(train_data)

train_data = (train_data - min_val) / (max_val - min_val)
test_data = (test_data - min_val) / (max_val - min_val)

train_data = tf.cast(train_data, tf.float32)
test_data = tf.cast(test_data, tf.float32)
"""

#train the autoencoder (final step)
"""
autoencoder.fit(x_train, x_train,
                epochs=10,
                shuffle=True,
                validation_data=(x_test, x_test))

encoded_imgs = autoencoder.encoder(x_test).numpy()
decoded_imgs = autoencoder.decoder(encoded_imgs).numpy()

n = 10
plt.figure(figsize=(20, 4))
for i in range(n):
  # display original
  ax = plt.subplot(2, n, i + 1)
  plt.imshow(x_test[i])
  plt.title("original")
  plt.gray()
  ax.get_xaxis().set_visible(False)
  ax.get_yaxis().set_visible(False)

  # display reconstruction
  ax = plt.subplot(2, n, i + 1 + n)
  plt.imshow(decoded_imgs[i])
  plt.title("reconstructed")
  plt.gray()
  ax.get_xaxis().set_visible(False)
  ax.get_yaxis().set_visible(False)
plt.show()

"""
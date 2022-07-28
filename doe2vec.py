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
seed = 0

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
                self.Y.append(array_y.flatten())
            except:
                continue
        print(tries)
        return self.Y
        

obj = Doe2Vec(d, m, n=1000)
y = np.array(obj.generateData())
print(y.shape)

a = aaa
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

#normalize the train data (this should be done per row (not per column!))

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
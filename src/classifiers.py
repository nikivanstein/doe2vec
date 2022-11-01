from random import sample

import tensorflow as tf
from tensorflow.keras import layers, losses
from tensorflow.keras.models import Model
import numpy as np
from keras.layers import Dense, Input, Concatenate, Lambda
from keras.utils.vis_utils import plot_model
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import keras.backend as K
from sklearn.metrics import f1_score
import json
from keras.utils import np_utils


class CustomConnected(Dense):

    def __init__(self,units,connections,**kwargs):
        """Custom dense layer with structural information in the 
        connections.
        """
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

class StructuralInformedDense(Model):
    def __init__(self, layer_sizes, num_classes, sample_size, DOE):
        """Dense classifier with custom connections based on the DOE between input and first hidden layer.
        """
        super(StructuralInformedDense, self).__init__()
        self.layer_sizes = layer_sizes
        self.DOE = DOE
        self.num_classes = num_classes
        self.dim = self.DOE.shape[1]
        self.sample_size = sample_size
        self.classifier = self._classifier()


    def _classifier(self):
        '''Create a Dense network with shape information from the DOE'''
        #we use knowledge of the space filling design to determine the distance threshold
        inputTensor = Input((self.sample_size,))
        connections = np.zeros((self.sample_size,self.sample_size))
        pair_distances = pairwise_distances(self.DOE, metric='cityblock')
        for i in range(0,len(self.DOE)):
            indexes_to_use = np.argsort(pair_distances[i,:])[:self.dim * 2 + 1]
            connections[indexes_to_use, i] = 1
        tf_connections = tf.convert_to_tensor(connections, dtype=tf.float32)

        x = CustomConnected(self.sample_size, tf_connections, activation="relu")(inputTensor)
        for num_nodes in self.layer_sizes:
            x = Dense(num_nodes, activation="relu")(x)

        x = Dense(self.num_classes, activation="sigmoid")(x)
        classifier = tf.keras.Model(inputTensor, x, name="StructuralInformedDense")
        return classifier

    def call(self, x):
        return self.classifier(x)


if __name__ == "__main__":
    import os
    import sys
    if not sys.warnoptions:
        import warnings
        warnings.simplefilter("ignore")
    import bbobbenchmarks as bbob
    from scipy.stats import qmc
    seed=0


    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    """Classification experiment for BBOB
    """
    f1_results = {}
    calc_ela = False
    all_dims = [2,5,10,20,40]
    latent_dim = 24
    m=9 #number of samples
    for model_type in ["SID"]:
        f1_results[model_type] = {}
        for dim in all_dims:
            sampler = qmc.Sobol(d=dim, scramble=False, seed=seed)
            sample = sampler.random_base2(m=m)
            sample = sample * 10 - 5
            X = []
            multim_label = []
            global_label = []
            funnel_label = []
            for i in range(120):
                for f in range(1, 25):
                    fun, opt = bbob.instantiate(f, i)
                    bbob_y = np.asarray(list(map(fun, sample)))
                    array_x = (bbob_y.flatten() - np.min(bbob_y)) / (
                        np.max(bbob_y) - np.min(bbob_y)
                    )
                    X.append(array_x)
                    if f in [1,2]:
                        multim_label.append("none")
                        global_label.append("none")
                        funnel_label.append("yes")
                    elif f in [3,4]:
                        multim_label.append("high")
                        global_label.append("strong")
                        funnel_label.append("yes")
                    elif f in [8,9]:
                        multim_label.append("low")
                        global_label.append("none")
                        funnel_label.append("yes")
                    elif f in [5,6,7,10,11,12,13,14]:
                        multim_label.append("none")
                        global_label.append("none")
                        funnel_label.append("yes")
                    elif f in [15,19]:
                        multim_label.append("high")
                        global_label.append("strong")
                        funnel_label.append("yes")
                    elif f in [16]:
                        multim_label.append("high")
                        global_label.append("medium")
                        funnel_label.append("none")
                    elif f in [17,18]:
                        multim_label.append("high")
                        global_label.append("medium")
                        funnel_label.append("yes")
                    elif f in [20]:
                        multim_label.append("medium")
                        global_label.append("deceptive")
                        funnel_label.append("yes")
                    elif f in [21]:
                        multim_label.append("medium")
                        global_label.append("none")
                        funnel_label.append("none")
                    elif f in [22]:
                        multim_label.append("low")
                        global_label.append("none")
                        funnel_label.append("none")
                    elif f in [23]:
                        multim_label.append("high")
                        global_label.append("none")
                        funnel_label.append("none")
                    elif f in [24]:
                        multim_label.append("high")
                        global_label.append("weak")
                        funnel_label.append("yes")
            #create a dense network for each classification problem
            
                    
            X = np.array(X)
            
            y_1 = np.array(multim_label)
            y_2 = np.array(global_label)
            y_3 = np.array(funnel_label)
            #use LabelEncoder or OneHotEncoder
            enc1 = LabelEncoder()
            enc2 = LabelEncoder()
            enc3 = LabelEncoder()

            
            y_1 = enc1.fit_transform(y_1)
            y_2 = enc2.fit_transform(y_2)
            y_3 = enc3.fit_transform(y_3)

            dummy_y1 = np_utils.to_categorical(y_1)
            dummy_y2 = np_utils.to_categorical(y_2)
            dummy_y3 = np_utils.to_categorical(y_3)
            test_size = 20*25
            
            X_train = X[:-test_size]
            X_test = X[-test_size:]

            
            dummy_ys = [dummy_y1,dummy_y2,dummy_y3]
            ys = [y_1,y_2,y_3]
            probs = ["multimodal", "global", "funnel"]
            for i in [0,1,2]:
                y = dummy_ys[i]
                real_y = ys[i]
                prob = probs[i]
                cf = StructuralInformedDense([128,64],y.shape[1],X.shape[1],sample)
                cf.compile(loss='binary_crossentropy', optimizer='adam')
                cf.fit(
                    X_train, y[:-test_size],
                    epochs=50,
                    batch_size=32,
                    shuffle=True,
                    validation_data=(X_test, y[-test_size:]),
                    verbose=0
                )
                y_dummy_pred = cf.predict(X_test)
                y_pred = np.argmax(y_dummy_pred, axis=1)
                score = f1_score(real_y[-test_size:], y_pred, average='macro')
                print("new",dim, prob, score)
                f1_results[model_type][f"d{dim} {prob}"] = score

                ### normal dense network
                denseModel = tf.keras.Sequential(
                    [
                        #layers.Dense(X.shape[1], activation="relu"),
                        layers.Dense(128, activation="relu"),
                        layers.Dense(64, activation="relu"),
                        layers.Dense(y.shape[1], activation="sigmoid"),
                    ]
                )
                denseModel.compile(loss='binary_crossentropy', optimizer='adam')
                denseModel.fit(
                    X_train, y[:-test_size],
                    epochs=50,
                    batch_size=32,
                    shuffle=True,
                    validation_data=(X_test, y[-test_size:]),
                    verbose=0
                )
                y_dummy_pred = denseModel.predict(X_test)
                y_pred = np.argmax(y_dummy_pred, axis=1)
                score = f1_score(real_y[-test_size:], y_pred, average='macro')
                print("dense",dim, prob, score)
                f1_results["dense"][f"d{dim} {prob}"] = score
            


    with open('f1_results_class.json', 'w') as fp:
        json.dump(f1_results, fp)


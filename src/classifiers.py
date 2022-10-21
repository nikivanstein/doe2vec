from random import sample

import tensorflow as tf
from tensorflow.keras import layers, losses
from tensorflow.keras.models import Model
import numpy as np
from keras.layers import Dense, Input, Concatenate, Lambda
from keras.utils.vis_utils import plot_model
from sklearn.metrics import pairwise_distances
import keras.backend as K


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
        sorted_DOE = np.argsort(self.DOE, axis=0)
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
        classifier.summary()
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
            
                    
            X = np.array(encodings)
            y_1 = np.array(multim_label).flatten()
            y_2 = np.array(global_label).flatten()
            y_3 = np.array(funnel_label).flatten()

            test_size = 20*25
            
            X_train = X[:-test_size]
            X_test = X[-test_size:]

            rf = RandomForestClassifier(n_estimators=100)
            rf.fit(X_train, y_1[:-test_size])

            resRf = rf.predict(X_test)
            f1_macro = f1_score(y_1[-test_size:], resRf, average='macro')
            f1_results[model_type+str(latent_dim)][f"d{dim} multimodal"] = f1_macro

            rf = RandomForestClassifier(n_estimators=100)
            rf.fit(X_train, y_2[:-test_size])
            resRf = rf.predict(X_test)
            f1_macro = f1_score(y_2[-test_size:], resRf, average='macro')
            f1_results[model_type+str(latent_dim)][f"d{dim} global"] = f1_macro

            rf = RandomForestClassifier(n_estimators=100)
            rf.fit(X_train, y_3[:-test_size])
            resRf = rf.predict(X_test)
            f1_macro = f1_score(y_3[-test_size:], resRf, average='macro')
            f1_results[model_type+str(latent_dim)][f"d{dim} funnel"] = f1_macro

            print(f1_results)


    with open('f1_results.json', 'w') as fp:
        json.dump(f1_results, fp)


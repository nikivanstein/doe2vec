#


## doe_model
[source](https://github.com/Basvanstein/doe2vec/blob/master/src/doe2vec.py/#L30)
```python 
doe_model(
   dim, m, n = 100000, latent_dim = 32, seed_nr = 0, kl_weight = 0.001,
   custom_sample = None, use_mlflow = False, mlflow_name = 'Doc2Vec',
   model_type = 'VAE'
)
```




**Methods:**


### .load_from_huggingface
[source](https://github.com/Basvanstein/doe2vec/blob/master/src/doe2vec.py/#L93)
```python
.load_from_huggingface(
   repo = 'BasStein'
)
```

---
Load a pre-trained model from a HuggingFace repository.


**Args**

* **repo** (str, optional) : the huggingface repo to load from.


### .loadModel
[source](https://github.com/Basvanstein/doe2vec/blob/master/src/doe2vec.py/#L124)
```python
.loadModel(
   dir = 'models'
)
```

---
Load a pre-trained Doe2vec model.


**Args**

* **dir** (str, optional) : The directory where the model is stored. Defaults to "models".


**Returns**

* **bool**  : True if loaded, else False.


### .loadData
[source](https://github.com/Basvanstein/doe2vec/blob/master/src/doe2vec.py/#L146)
```python
.loadData(
   dir = 'data'
)
```

---
Load a stored functions file and retrieve all the landscapes.


**Args**

* **dir** (str, optional) : The directory where the data are stored. Defaults to "data".


**Returns**

* **bool**  : True if loaded, else False.


### .getSample
[source](https://github.com/Basvanstein/doe2vec/blob/master/src/doe2vec.py/#L179)
```python
.getSample()
```

---
Get the sample DOE used.


**Returns**

* **array**  : Sample


### .generateData
[source](https://github.com/Basvanstein/doe2vec/blob/master/src/doe2vec.py/#L187)
```python
.generateData()
```

---
Generate the random functions for training the autoencoder.


**Returns**

* **array**  : array with evaluated random functions on sample.


### .setData
[source](https://github.com/Basvanstein/doe2vec/blob/master/src/doe2vec.py/#L232)
```python
.setData(
   Y
)
```

---
Helper function to load the data and split in train validation sets.


**Args**

* **Y** (nd array) : the data set to use.


### .compile
[source](https://github.com/Basvanstein/doe2vec/blob/master/src/doe2vec.py/#L242)
```python
.compile()
```

---
Compile the autoencoder architecture.

### .fit
[source](https://github.com/Basvanstein/doe2vec/blob/master/src/doe2vec.py/#L253)
```python
.fit(
   epochs = 100
)
```

---
Fit the autoencoder model.


**Args**

* **epochs** (int, optional) : Number of epochs to train. Defaults to 100.


### .fitNN
[source](https://github.com/Basvanstein/doe2vec/blob/master/src/doe2vec.py/#L286)
```python
.fitNN()
```

---
Fit the neirest neighbour tree to find similar functions.

### .getNeighbourFunction
[source](https://github.com/Basvanstein/doe2vec/blob/master/src/doe2vec.py/#L294)
```python
.getNeighbourFunction(
   features
)
```

---
Get the closest random generated function depending on a set of features (from another function).


**Args**

* **features** (array) : Feature vector (given by the encode() function)


**Returns**

* **tuple**  : random function string, distance


### .save
[source](https://github.com/Basvanstein/doe2vec/blob/master/src/doe2vec.py/#L308)
```python
.save(
   model_dir = 'model', data_dir = 'data'
)
```

---
Save the model and random functions used for training


**Args**

* **model_dir** (str, optional) : Directory to store the model. Defaults to "model".
* **data_dir** (str, optional) : Directory to store the random functions. Defaults to "data".


### .saveModel
[source](https://github.com/Basvanstein/doe2vec/blob/master/src/doe2vec.py/#L319)
```python
.saveModel(
   model_dir
)
```

---
Save the model


**Args**

* **model_dir** (str, optional) : Directory to store the model. Defaults to "model".


### .saveData
[source](https://github.com/Basvanstein/doe2vec/blob/master/src/doe2vec.py/#L329)
```python
.saveData(
   data_dir = 'data'
)
```

---
Save the random functions used for training


**Args**

* **data_dir** (str, optional) : Directory to store the random functions. Defaults to "data".


### .encode
[source](https://github.com/Basvanstein/doe2vec/blob/master/src/doe2vec.py/#L339)
```python
.encode(
   X
)
```

---
Encode a Design of Experiments.


**Args**

* **X** (array) : The DOE to encode.


**Returns**

* **array**  : encoded feature vector.


### .summary
[source](https://github.com/Basvanstein/doe2vec/blob/master/src/doe2vec.py/#L356)
```python
.summary()
```

---
Get a summary of the autoencoder model

### .plot_label_clusters_bbob
[source](https://github.com/Basvanstein/doe2vec/blob/master/src/doe2vec.py/#L360)
```python
.plot_label_clusters_bbob()
```


### .visualizeTestData
[source](https://github.com/Basvanstein/doe2vec/blob/master/src/doe2vec.py/#L409)
```python
.visualizeTestData(
   n = 5
)
```

---
Get a visualisation of the validation data.


**Args**

* **n** (int, optional) : The number of validation DOEs to show. Defaults to 5.


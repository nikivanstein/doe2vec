# DoE2Vec

DoE2Vec is a self-supervised approach to learn exploratory landscape analysis features from design of experiments.
The model can be used for downstream meta-learning tasks such as learninig which optimizer works best on a given optimization landscape.
Or to classify optimization landscapes in function groups.

The approach uses randomly generated functions and can also be used to find a "cheap" reference function given a DOE.
The model uses Sobol sequences as the default sampling method. A custom sampling method can also be used.
Both the samples and the landscape should be scaled between 0 and 1.


## Install package via pip

`pip install doe2vec`

Afterwards you can use the package via:

`from doe2vec import doe_model`

## Load a model from the HuggingFace Hub

Available models can be viewed here: https://huggingface.co/BasStein
A model name is build up like BasStein/doe2vec-d2-m8-ls16-VAE-kl0.001  
Where d is the number of dimensions, 8 the number (2^8) of samples, 16 the latent size, VAE the model type (variational autoencoder) and 0.001 the KL loss weight.

Example code of loading a huggingface model

    obj = doe_model(
                2,
                8,
                n= 50000,
                latent_dim=16,
                kl_weight=0.001,
                use_mlflow=False,
                model_type="VAE"
            )
    obj.load_from_huggingface("BasStein/doe2vec-d2-m8-ls16-VAE-kl0.001")
    #test the model
    obj.plot_label_clusters_bbob()
 
## How to Setup your Environment for Development

- `python3.8 -m venv env` 
- `source ./env/bin/activate`
- `pip install -r requirements.txt`


## Generate the Data Set

To generate the artificial function dataset for a given dimensionality and sample size
run the following code
    from doe2vec inport doe_model

    obj = doe_model(d, m, n=50000, latent_dim=latent_dim)
    if not obj.load():
        obj.generateData()
        obj.compile()
        obj.fit(100)
        obj.save()

Where `d` is the number of dimensions, `m` the number of samples (2^`m`) per DOE, `n` the number of functions generated and `latent_dim` the size of the output encoding vector.

Once a data set and encoder has been trained it can be loaded with the `load()` function.

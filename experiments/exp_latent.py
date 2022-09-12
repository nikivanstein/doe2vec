import os

import numpy as np

import bbobbenchmarks as bbob
from doe2vec import doe_model

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import matplotlib.pyplot as plt
from matplotlib import cm


obj = doe_model(
    2, 8, n=250000, latent_dim=32, model_type="VAE", kl_weight=0.1, use_mlflow=False
)
if not obj.loadModel("../models"):
    obj.generateData()
    obj.compile()
    obj.fit(100)
    obj.save()


obj.plot_label_clusters_bbob()

import os

import numpy as np
from datasets import Dataset
from huggingface_hub import push_to_hub_keras

from doe2vec import doe_model

model_type = "VAE"
kl_weight = 0.001
seed = 0
dir = "../models"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
for d in [5]:
    for m in [8]:
        for latent_dim in [16, 24]:
            obj = doe_model(
                d,
                m,
                n=d * 50000,
                latent_dim=latent_dim,
                kl_weight=kl_weight,
                use_mlflow=False,
            )
            if not obj.load("../models/"):
                obj.generateData()
                obj.compile()
                obj.fit(100)
                obj.save("../models/")
            data = np.load(
                f"{dir}/data_{d}-{m}-{latent_dim}-{seed}-{model_type}{kl_weight}.npy"
            )
            functions = np.load(
                f"{dir}/functions_{d}-{m}-{latent_dim}-{seed}-{model_type}{kl_weight}.npy"
            )
            datadict = {
                "y": data,
                "function": functions,
                "array_x": [obj.sample] * len(functions),
            }
            dataset = Dataset.from_dict(datadict)
            print(dataset)
            push_to_hub_keras(
                obj.autoencoder,
                f"doe2vec-d{d}-m{m}-ls{latent_dim}-{model_type}-kl{kl_weight}",
            )  # , repo_url="https://huggingface.co/BasStein/doe2vec-d2-m8-ls16")
            dataset.push_to_hub(
                f"doe2vec-d{d}-m{m}-ls{latent_dim}-{model_type}-kl{kl_weight}"
            )

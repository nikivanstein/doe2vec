import os
from doe2vec import doe_model
from huggingface_hub import push_to_hub_keras

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
for d in [2]:
    for m in [8]:
        for latent_dim in [16]:
            obj = doe_model(d, m, n=d * 50000, latent_dim=latent_dim, use_mlflow=False)
            if not obj.load("../models/"):
                obj.generateData()
                obj.compile()
                obj.fit(100)
                obj.save("../models/")
            push_to_hub_keras(obj.autoencoder, f"doe2vec-d{d}-m{m}-ls{latent_dim}")#, repo_url="https://huggingface.co/BasStein/doe2vec-d2-m8-ls16")

"""PyTest file for Doe2Vec
"""

import os

import pytest

from .doe2vec import doe_model

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def test_happy_path():
    model = doe_model(2, 8, 200, 24, 0, kl_weight=0.001)
    model.generateData()
    model.compile()
    model.fit(1)
    assert model.save("temp", "temp") == True


def test_fit_before_compile():
    model = doe_model(2, 8, 10000, 24, 0, kl_weight=0.001)
    # fit before compile
    with pytest.raises(AttributeError, match="Autoencoder model is not compiled yet"):
        model.fit(1)


def test_load_from_folder():
    model = doe_model(2, 8, n=200, latent_dim=24, kl_weight=0.001)
    assert model.loadModel("temp") == True
    assert model.loadData("temp") == True


def test_load_from_huggingface():
    model = doe_model(
        2,
        8,
        n=250000,
        latent_dim=24,
        kl_weight=0.001,
    )
    model.load_from_huggingface()

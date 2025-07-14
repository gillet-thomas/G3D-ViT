import src.models.fmriEncoder as fmriEncoder
import numpy as np
import pytest
import torch


@pytest.fixture
def config():
    config = {
        "grid_size": 40,
        "cube_size": 8,
        "grid_noise": 0.0,
        "batch_size": 4,
        "device": "cuda",
        "dropout": 0.1,
        "vit_patch_size": 5,
        "threshold": 5,
    }
    return config


def test_forward_pass(config):
    x = torch.randn(1, config["grid_size"], config["grid_size"], config["grid_size"])

    encoder = fmriEncoder.fmriEncoder(config).eval()
    with torch.no_grad():
        output = encoder(x)

    dim = (config["grid_size"] // config["cube_size"]) ** 3
    assert output.shape == (1, dim)

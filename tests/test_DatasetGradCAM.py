import numpy as np
import pytest
import torch

import src.data.DatasetGradCAM as DatasetGradCAM


@pytest.fixture
def config():
    config = {
        "base_path": "./",
        "gradcam_train_path": "temp_train.pkl",
        "gradcam_val_path": "temp_val.pkl",
        "grid_size": 32,
        "cube_size": 8,
        "grid_noise": 0.0,
        "visualize_samples": False,
        "batch_size": 4,
        "num_samples": 100,
        "output_dir": "/tmp/output",
    }
    return config


@pytest.fixture
def mock_dataset(config):
    return DatasetGradCAM.GradCAMDataset(config, mode="train", generate_data=True)


def test_generate_data(mocker, config):

    # mock_data = [(np.zeros((32, 32, 32)), 0, np.array([0, 0, 0]))]  # fmri volumes, labels, coordinates
    mock_pickle_load = mocker.patch("pickle.load")  # disable pickle.load
    mock_pickle_dump = mocker.patch("pickle.dump")  # disable pickle.dump
    mock_file = mocker.patch("builtins.open", mocker.mock_open())  # Mock file opening

    DatasetGradCAM.GradCAMDataset(config, mode="train", generate_data=True)

    assert mock_pickle_dump.call_count == 2

    train_call, val_call = mock_pickle_dump.call_args_list
    train_data = train_call[0][0]  # First argument of first call
    val_data = val_call[0][0]  # First argument of second call

    assert len(train_data) == int(0.8 * config["num_samples"])
    assert len(val_data) == int(0.2 * config["num_samples"])

    # Verify each sample has the correct structure (volume, label, coordinates)
    for volume, label, coordinates in train_data[:5]:  # Check first 5 samples
        assert volume.shape == (32, 32, 32)
        assert isinstance(label, (int, np.integer))
        assert coordinates.shape == (3,)
        assert 0 <= label < (32 // 8) ** 3  # Valid label range for position classification

    for volume, label, coordinates in val_data[:5]:  # Check first 5 samples
        assert volume.shape == (32, 32, 32)
        assert isinstance(label, (int, np.integer))
        assert coordinates.shape == (3,)
        assert 0 <= label < (32 // 8) ** 3  # Valid label range for position classification


def test_getitem(mock_dataset):
    volume, label, coordinates = mock_dataset[0]

    assert isinstance(volume, torch.Tensor)
    assert volume.dtype == torch.float32
    assert volume.shape == (
        mock_dataset.grid_size,
        mock_dataset.grid_size,
        mock_dataset.grid_size,
    )

    assert isinstance(label, torch.Tensor)
    assert label.dtype == torch.int64
    assert label.shape == torch.Size([])  # Scalar label

    assert isinstance(coordinates, torch.Tensor)
    assert coordinates.dtype == torch.float32  # Check original type, as it's not converted to int64
    assert coordinates.shape == (3,)

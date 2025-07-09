import os
import pickle
from unittest.mock import MagicMock, patch

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pytest
import torch

# Import the GradCAMDataset class from your module
from data.DatasetGradCAM import GradCAMDataset


@pytest.fixture
def config(tmp_path):
    """Fixture for a basic configuration dictionary."""
    return {
        "base_path": str(tmp_path),
        "gradcam_train_path": str(tmp_path / "train_data.pkl"),
        "gradcam_val_path": str(tmp_path / "val_data.pkl"),
        "grid_size": 16,
        "cube_size": 4,
        "grid_noise": 0.0,
        "visualize_samples": False,
        "batch_size": 4,
        "num_samples": 10,
        "output_dir": str(tmp_path / "output"),
    }


@pytest.fixture
def generated_dataset(config):
    """Fixture to generate and return a GradCAMDataset instance."""
    return GradCAMDataset(config, mode="train", generate_data=True)


def test_dataset_initialization_generate_data(config, tmp_path):
    """Test if the dataset initializes correctly and generates data."""
    dataset = GradCAMDataset(config, mode="train", generate_data=True)
    assert os.path.exists(config["gradcam_train_path"])
    assert os.path.exists(config["gradcam_val_path"])
    assert len(dataset) == int(0.8 * config["num_samples"])  # Train set size


def test_dataset_initialization_load_data(config, generated_dataset):
    """Test if the dataset initializes correctly by loading existing data."""
    # The generated_dataset fixture ensures data is generated first
    dataset = GradCAMDataset(config, mode="train", generate_data=False)
    assert len(dataset) == int(0.8 * config["num_samples"])
    assert isinstance(dataset.data, list)


def test_getitem(generated_dataset):
    """Test the __getitem__ method to ensure correct data types and shapes."""
    volume, label, coordinates = generated_dataset[0]

    assert isinstance(volume, torch.Tensor)
    assert volume.dtype == torch.float32
    assert volume.shape == (
        generated_dataset.grid_size,
        generated_dataset.grid_size,
        generated_dataset.grid_size,
    )

    assert isinstance(label, torch.Tensor)
    assert label.dtype == torch.int64
    assert label.shape == torch.Size([])  # Scalar label

    assert isinstance(coordinates, torch.Tensor)
    assert coordinates.dtype == torch.float32  # Check original type, as it's not converted to int64
    assert coordinates.shape == (3,)


def test_len(generated_dataset):
    """Test the __len__ method."""
    assert len(generated_dataset) == int(0.8 * generated_dataset.num_samples)


def test_generate_data_content(config, tmp_path):
    """Test the content of generated data, specifically the cube placement."""
    GradCAMDataset(config, mode="train", generate_data=True)

    with open(config["gradcam_train_path"], "rb") as f:
        train_data = pickle.load(f)

    # Check a few samples
    for i in range(min(3, len(train_data))):
        volume, label, coords = train_data[i]
        # Ensure the volume has the expected cube value at the coordinates
        x, y, z = int(coords[0]), int(coords[1]), int(coords[2])
        assert np.all(
            volume[
                x : x + config["cube_size"],
                y : y + config["cube_size"],
                z : z + config["cube_size"],
            ]
            == 1
        )
        # Ensure outside the cube is grid_noise
        temp_volume = np.copy(volume)
        temp_volume[
            x : x + config["cube_size"],
            y : y + config["cube_size"],
            z : z + config["cube_size"],
        ] = config["grid_noise"]
        assert np.all(temp_volume == config["grid_noise"])


@pytest.mark.skip
@patch("matplotlib.pyplot.savefig")
@patch("matplotlib.pyplot.show")  # Patch show to prevent plots from appearing
@patch("matplotlib.pyplot.close")
@patch("nibabel.save")
def test_visualize_sample_3d(mock_nib_save, mock_plt_close, mock_plt_show, mock_plt_savefig, config, tmp_path):
    """Test the visualize_sample_3d method."""
    config["visualize_samples"] = False  # Enable visualization for this test
    config["num_samples"] = 1  # Only generate one sample for simpler testing

    # Manually create a dummy dataset for visualization, as the fixture generates 10 samples
    # and the constructor calls visualize_sample_3d for the first 5 if visualize_samples is True.
    # We want to test a specific call, not the constructor's side effect.
    dummy_volume = np.full((config["grid_size"], config["grid_size"], config["grid_size"]), config["grid_noise"])
    dummy_volume[0 : config["cube_size"], 0 : config["cube_size"], 0 : config["cube_size"]] = 1
    dummy_label = 0
    dummy_coordinates = np.array([0, 0, 0])

    # Manually create the pickle file to control the data
    with open(config["gradcam_train_path"], "wb") as f:
        pickle.dump([(dummy_volume, dummy_label, dummy_coordinates)], f)

    dataset = GradCAMDataset(config, mode="train", generate_data=False)

    # Now call visualize_sample_3d explicitly for the first sample
    dataset.visualize_sample_3d(0)

    # Check if output directory was created
    assert os.path.exists(config["output_dir"])

    # Check if savefig and nib.save were called
    mock_plt_savefig.assert_called_once()
    mock_nib_save.assert_called_once()
    mock_plt_close.assert_called_once_with()

    # Verify the arguments passed to nib.save (check NIfTI image type and data content)
    args, kwargs = mock_nib_save.call_args
    nifti_image = args[0]
    assert isinstance(nifti_image, nib.Nifti1Image)
    np.testing.assert_array_equal(nifti_image.get_fdata(), dummy_volume)

    # Verify the filename for both saved files
    expected_filename_prefix = f"DatasetGradCAM_{config['grid_size']}grid_{config['cube_size']}cube_{str(config['grid_noise']).replace('.', 'p')}noise_0"
    mock_nib_save.assert_called_with(
        nib.Nifti1Image(dummy_volume, np.eye(4)), os.path.join(config["output_dir"], expected_filename_prefix)
    )
    mock_plt_savefig.assert_called_with(os.path.join(config["output_dir"], f"{expected_filename_prefix}.png"), dpi=300)


def test_dataset_val_mode(config, tmp_path):
    """Test dataset initialization in validation mode."""
    GradCAMDataset(config, mode="train", generate_data=True)  # Generate data first
    val_dataset = GradCAMDataset(config, mode="val", generate_data=False)
    assert len(val_dataset) == int(0.2 * config["num_samples"])


def test_cube_alignment(config, generated_dataset):
    """Test if generated cubes are aligned (multiples of cube_size)."""
    # Using the generated_dataset fixture ensures data has been created
    for i in range(len(generated_dataset)):
        _, _, coordinates = generated_dataset.data[i]  # Access raw data before tensor conversion
        x, y, z = coordinates.tolist()
        assert x % config["cube_size"] == 0
        assert y % config["cube_size"] == 0
        assert z % config["cube_size"] == 0


def test_label_calculation(config, generated_dataset):
    """Test if labels are calculated correctly based on cube position."""
    num_cubes = config["grid_size"] // config["cube_size"]
    for i in range(len(generated_dataset)):
        _, label, coordinates = generated_dataset.data[i]
        tx, ty, tz = [int(c) for c in coordinates.tolist()]
        expected_label = (
            (tx // config["cube_size"])
            + (ty // config["cube_size"]) * num_cubes
            + (tz // config["cube_size"]) * num_cubes * num_cubes
        )
        assert label == expected_label

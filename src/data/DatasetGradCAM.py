import os
import pickle

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset


# GradCAM dataset class (synthetic data)
class GradCAMDataset(Dataset):
    """A PyTorch Dataset for loading and optionally generating synthetic 3D volumetric data.

    This dataset is designed to create synthetic 3D volumes containing a "cube"
    of a specific value within a noisy grid, suitable for tasks like
    predicting the cube's position. It also includes functionality to
    visualize these 3D samples.

    Attributes:
        mode (str): The mode of the dataset ('train' or 'val').
        config (dict): Configuration parameters for the dataset.
        base_path (str): Base directory for dataset files.
        dataset_path (str): Full path to the pickle file containing the dataset.
        grid_size (int): The size of the 3D grid (e.g., 64 for 64x64x64).
        cube_size (int): The size of the cube to be embedded in the grid.
        grid_noise (float): The value used as noise for voxels outside the cube.
        visualize_samples (bool): If True, visualizes a few samples upon initialization.
        batch_size (int): Batch size (from config, though not directly used in Dataset methods).
        num_samples (int): Total number of samples to generate (if `generate_data` is True).
        data (list): A list of tuples loaded from pickle files, where each tuple contains (volume, label, coordinates).
    """

    def __init__(self, config, mode="train", generate_data=False):
        """Initializes the GradCAMDataset.

        Args:
            config (dict): A dictionary containing configuration parameters.
                Expected keys:
                - "base_path" (str): Base directory for dataset files.
                - "gradcam_train_path" (str): Path to the training data pickle file.
                - "gradcam_val_path" (str): Path to the validation data pickle file.
                - "grid_size" (int): The size of the 3D grid.
                - "cube_size" (int): The size of the cube.
                - "grid_noise" (float): The noise value for the grid.
                - "visualize_samples" (bool): Whether to visualize samples.
                - "batch_size" (int): Batch size.
                - "num_samples" (int): Number of samples to generate.
                - "output_dir" (str): Directory to save visualizations.
            mode (str, optional): The mode of the dataset ('train' or 'val'). Defaults to 'train'.
            generate_data (bool, optional): If True, generates synthetic data and saves
                it to pickle files. If False, loads existing data. Defaults to False.
        """

        self.mode = mode
        self.config = config
        self.base_path = config["base_path"]
        self.dataset_path = config["gradcam_train_path"] if mode == "train" else config["gradcam_val_path"]

        self.grid_size = config["grid_size"]
        self.cube_size = config["cube_size"]
        self.grid_noise = config["grid_noise"]
        self.visualize_samples = config["visualize_samples"]

        self.batch_size = config["batch_size"]
        self.num_samples = config["num_samples"]

        if generate_data:
            self.generate_data()

        with open(self.dataset_path, "rb") as f:
            self.data = pickle.load(f)

        if self.visualize_samples:
            self.visualize_sample_3d(1)
            self.visualize_sample_3d(2)
            self.visualize_sample_3d(3)
            self.visualize_sample_3d(4)
            self.visualize_sample_3d(5)

        print(f"Dataset initialized: {len(self.data)} {mode} samples")

    def generate_data(self):
        """Generates synthetic 3D synthetic volumes with embedded cubes.

        This method creates 3D volumes, labels indicating the cube's position, and the cube's coordinates.
        It then splits the generated data into training and validation sets and saves them as pickle files.

        The current implementation generates "grid-aligned" cubes whose starting coordinates are multiples
        of cube_size, ensuring cubes don't overlap and fit perfectly within a regular grid.
        """

        volumes = np.zeros((self.num_samples, self.grid_size, self.grid_size, self.grid_size))
        labels = np.zeros((self.num_samples), dtype=int)
        coordinates = np.zeros((self.num_samples, 3))

        num_cubes = self.grid_size // self.cube_size  # Number of cubes in each dimension

        for i in range(self.num_samples):
            # Aligned cubes
            tx = np.random.randint(0, num_cubes) * self.cube_size
            ty = np.random.randint(0, num_cubes) * self.cube_size
            tz = np.random.randint(0, num_cubes) * self.cube_size

            # Not-aligned cubes
            # tx = np.random.randint(0, self.grid_size - self.cube_size)
            # ty = np.random.randint(0, self.grid_size - self.cube_size)
            # tz = np.random.randint(0, self.grid_size - self.cube_size)

            # Classification task 1 (position)
            volumes[i] = self.grid_noise  # Add background noise for non-cube voxels
            volumes[i, tx : tx + self.cube_size, ty : ty + self.cube_size, tz : tz + self.cube_size] = 1
            labels[i] = (
                (tx // self.cube_size)
                + (ty // self.cube_size) * num_cubes
                + (tz // self.cube_size) * num_cubes * num_cubes
            )
            coordinates[i] = [tx, ty, tz]

            # Classification task 2 (content)
            # value = random.choice([-1, 1])
            # volumes[i] = 0
            # volumes[i, tx:tx+self.cube_size, ty:ty+self.cube_size, tz:tz+self.cube_size] = value
            # labels[i] = 0 if value == -1 else 1
            # coordinates[i] = [tx, ty, tz]

        train_size = int(0.8 * self.num_samples)
        train_samples = [
            (v, l, c) for v, l, c in zip(volumes[:train_size], labels[:train_size], coordinates[:train_size])
        ]
        val_samples = [
            (v, l, c) for v, l, c in zip(volumes[train_size:], labels[train_size:], coordinates[train_size:])
        ]

        print(f"Training volumes: {len(train_samples)}")
        print(f"Validation volumes: {len(val_samples)}")

        # Save to pickle files
        with open(self.config["gradcam_train_path"], "wb") as f:
            pickle.dump(train_samples, f)
        with open(self.config["gradcam_val_path"], "wb") as f:
            pickle.dump(val_samples, f)
        print("Datasets saved!")

    def __getitem__(self, idx):
        """Retrieves a single sample from the dataset.

        Args:
            idx (int): The index of the sample to retrieve.

        Returns:
            tuple: A tuple containing:
                - volume (torch.Tensor): The 3D synthetic volume, converted to float32.
                - label (torch.Tensor): The label for the volume, converted to int64.
                - coordinates (torch.Tensor): The coordinates of the embedded cube, converted to float32.
        """

        volume, label, coordinates = self.data[idx]
        return (
            torch.tensor(volume, dtype=torch.float32),
            torch.tensor(label, dtype=torch.int64),
            torch.tensor(coordinates, dtype=torch.float32),
        )

    def __len__(self):
        """Returns the total number of samples in the dataset.

        Returns:
            int: The number of samples.
        """

        return len(self.data)

    def visualize_sample_3d(self, idx):
        """Visualizes a single 3D sample from the dataset.

        This method creates a 3D scatter plot of the embedded cube within the volume
        and saves it as a PNG image. It also saves the volume as a NIfTI file.
        The method handles tensor-to-numpy conversion and removes extra dimensions if present.

        Args:
            idx (int): The index of the sample to visualize.
        """

        # Create save directory if it doesn't exist
        os.makedirs(self.config["output_dir"], exist_ok=True)

        # Get the data

        volume, label, coordinates = self.data[idx]

        # If data is torch tensor, convert to numpy
        if torch.is_tensor(volume):
            volume = volume.numpy()

        # Remove batch dimension if present
        if volume.ndim == 5:
            volume = volume[0]

        # Remove channel dimension if present
        if volume.ndim == 4:
            volume = volume[0]

        # Create 3D plot
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(
            *np.where(volume == 1), c="red", marker="s", alpha=0.5, s=50
        )  # scatter plot for all points with value 1 (the cube)

        # Add a bounding box for the volume
        ax.set(xlim=(0, volume.shape[0]), ylim=(0, volume.shape[1]), zlim=(0, volume.shape[2]))  # Grid size
        ax.set(xlabel="X axis", ylabel="Y axis", zlabel="Z axis")

        # Save figure and nifti file

        file_name = f"DatasetGradCAM_{self.grid_size}grid_{self.cube_size}cube_{self.grid_noise}noise_{idx}".replace(
            ".", "p"
        )
        plt.title(f"3D Visualization of Target Cube (Label: {label}, coordinates: {coordinates})")
        plt.tight_layout()

        nib.save(nib.Nifti1Image(volume, np.eye(4)), os.path.join(self.config["output_dir"], file_name))
        plt.savefig(os.path.join(self.config["output_dir"], f"{file_name}.png"), dpi=300)
        plt.close()
        print(f"3D visualization saved to {os.path.join(self.config['output_dir'], file_name)}")

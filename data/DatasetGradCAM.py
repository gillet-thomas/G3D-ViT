import os
import torch
import pickle
import numpy as np
import nibabel as nib    
import matplotlib.pyplot as plt 
from torch.utils.data import Dataset


# GradCAM dataset class (fake data)
class GradCAMDataset(Dataset):
    def __init__(self, config, mode='train', generate_data=False):
        self.mode = mode
        self.config = config
        self.base_path = config['base_path']
        self.dataset_path = config['gradcam_train_path'] if mode == 'train' else config['gradcam_val_path']

        self.grid_size = config['grid_size']
        self.cube_size = config['cube_size']
        self.grid_noise = config['grid_noise']
        self.visualize_samples = config['visualize_samples']
        
        self.batch_size = config['batch_size']
        self.num_samples = config['num_samples']

        if generate_data:
            self.generate_data()
        
        with open(self.dataset_path, 'rb') as f:
            self.data = pickle.load(f)

        if self.visualize_samples:
            self.visualize_sample_3d(1)
            self.visualize_sample_3d(2)
            self.visualize_sample_3d(3)
            self.visualize_sample_3d(4)
            self.visualize_sample_3d(5)

        print(f"Dataset initialized: {len(self.data)} {mode} samples")

    def generate_data(self):
        volumes = np.zeros((self.num_samples, self.grid_size, self.grid_size, self.grid_size))
        labels = np.zeros((self.num_samples), dtype=int)
        coordinates = np.zeros((self.num_samples, 3))
        
        num_cubes = self.grid_size // self.cube_size # Number of cubes in each dimension
        
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
            volumes[i] = self.grid_noise # Add noise for other voxels
            volumes[i, tx:tx+self.cube_size, ty:ty+self.cube_size, tz:tz+self.cube_size] = 1
            labels[i] = (tx//self.cube_size) + (ty//self.cube_size) * num_cubes + (tz//self.cube_size) * num_cubes * num_cubes
            coordinates[i] = [tx, ty, tz]

            # Classification task 2 (content)
            # value = random.choice([-1, 1])
            # volumes[i] = 0
            # volumes[i, tx:tx+self.cube_size, ty:ty+self.cube_size, tz:tz+self.cube_size] = value
            # labels[i] = 0 if value == -1 else 1
            # coordinates[i] = [tx, ty, tz]

        train_size = int(0.8 * self.num_samples)
        train_samples = [(v, l, c) for v, l, c in zip(volumes[:train_size], labels[:train_size], coordinates[:train_size])]
        val_samples = [(v, l, c) for v, l, c in zip(volumes[train_size:], labels[train_size:], coordinates[train_size:])]
        
        print(f"Training volumes: {len(train_samples)}")
        print(f"Validation volumes: {len(val_samples)}")
        
        # Save to pickle files
        with open(self.config['gradcam_train_path'], 'wb') as f:
            pickle.dump(train_samples, f)
        with open(self.config['gradcam_val_path'], 'wb') as f:
            pickle.dump(val_samples, f)
        print("Datasets saved!")

    def __getitem__(self, idx):
        volume, label, coordinates = self.data[idx]
        return torch.tensor(volume, dtype=torch.float32), torch.tensor(label, dtype=torch.int64), torch.tensor(coordinates)

    def __len__(self):
        return len(self.data)

    def visualize_sample_3d(self, idx):
        
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
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(*np.where(volume == 1), c='red', marker='s', alpha=0.5, s=50) # scatter plot for all points with value 1 (the cube)

        # Add a bounding box for the volume
        ax.set(xlim=(0, volume.shape[0]), ylim=(0, volume.shape[1]), zlim=(0, volume.shape[2])) # Grid size
        ax.set(xlabel='X axis', ylabel='Y axis', zlabel='Z axis')
        
        # Save figure and nifti file
        file_name = f'DatasetGradCAM_{self.grid_size}grid_{self.cube_size}cube_{self.grid_noise}noise_{idx}'.replace('.', 'p')
        plt.title(f'3D Visualization of Target Cube (Label: {label}, coordinates: {coordinates})')
        plt.tight_layout()

        nib.save(nib.Nifti1Image(volume, np.eye(4)), os.path.join(self.config["output_dir"], file_name))
        plt.savefig(os.path.join(self.config["output_dir"], f'{file_name}.png'), dpi=300)
        plt.close()
        print(f"3D visualization saved to {os.path.join(self.config['output_dir'], file_name)}")
        
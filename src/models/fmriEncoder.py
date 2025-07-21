import numpy as np
import torch.nn as nn
from vit_pytorch.vit_3d import ViT


import torch
import torch.nn.functional as F



class fmriEncoder(nn.Module):
    """A PyTorch module for encoding fMRI data and generating attention maps.

    This class leverages a `ViT3DEncoder` internally and implements hooks to
    capture intermediate activations and gradients, which are then used
    for attention map computation.

    Attributes:
        config (dict): Configuration parameters for the model.
        device (torch.device): The device (e.g., 'cpu' or 'cuda') on which the model resides.
        volume_encoder (ViT3DEncoder): An instance of the 3D Vision Transformer encoder.
        gradients (dict): Stores gradients captured by the backward hook.
        activations (dict): Stores activations captured by the forward hook.
        forward_handle (torch.utils.hooks.RemovableHandle): Handle for the registered forward hook.
        backward_handle (torch.utils.hooks.RemovableHandle): Handle for the registered backward hook.
    """

    def __init__(self, config):
        """Initializes the fmriEncoder model.

        Args:
            config (dict): A dictionary containing configuration parameters for the model.
                Expected keys:
                - "device" (str): 'cpu' or 'cuda'.
                - "grid_size" (int): The size of the 3D fMRI grid (e.g., 64 for 64x64x64).
                - "vit_patch_size" (int): The patch size for the 3D ViT.
                - "threshold" (float): Percentile threshold for attention map visualization (e.g., 90 for top 10%).
        """

        super().__init__()
        self.config = config
        self.device = config["device"]

        self.volume_encoder = ViT3DEncoder(config)
        self.to(self.device)  # Move entire model to device

        # Gradients and activations tracking
        self.gradients = {}
        self.activations = {}
        self.register_hooks()

    def register_hooks(self):
        """Registers forward and backward hooks on the last normalization layer of the 3D ViT.

        These hooks capture the activations and their corresponding gradients
        during the forward and backward passes, respectively, which are
        essential for computing attention maps (e.g., Grad-CAM).
        """

        # Get the last attention layer
        last_attention = self.volume_encoder.vit3d.transformer.layers[-1][0].norm

        def forward_hook(module, input, output):
            """Hook to capture activations."""
            self.activations = output.detach().cpu()

        def backward_hook(module, grad_input, grad_output):
            """Hook to capture gradients."""
            self.gradients = grad_output[0].detach().cpu()

        # Register hooks
        self.forward_handle = last_attention.register_forward_hook(forward_hook)
        self.backward_handle = last_attention.register_backward_hook(backward_hook)

    def forward(self, x):
        """Defines the forward pass of the fmriEncoder.

        Encodes each fMRI volume using the internal 3D Vision Transformer.

        Args:
            x (torch.Tensor): Input fMRI data. Expected shape is
                (batch_size, H, W, D) for 3D volumes,
                where H, W, D are height, width, and depth.

        Returns:
            torch.Tensor: The encoded representation of the input fMRI data.
                The shape depends on the `pool` setting of the ViT,
                typically (batch_size, dim) if 'cls' token pooling is used.
        """

        volume_encodings = self.volume_encoder(x)  # Encode each volume with 3D-ViT
        return volume_encodings

    def get_attention_map(self, x):
        """Computes a 3D attention map highlighting regions of importance.

        The process involves a forward pass, a backward pass to get gradients and
        activations, computation of importance weights, and upsampling the
        resulting attention map to the original volume size.

        Args:
            x (torch.Tensor): The input 3D fMRI volume for which to compute the
                attention map. Expected shape: (batch_size, H, W, D).

        Returns:
            tuple: A tuple containing:
                - cam_3d (torch.Tensor): The 3D attention map, upsampled to the
                    original input volume size. Shape: (H, W, D).
                - class_idx (torch.Tensor): The predicted class index for the input volume.
                    Shape: (batch_size,).
        """

        grid_size = self.config["grid_size"]
        patch_size = self.config["vit_patch_size"]
        threshold = self.config["threshold"]

        # Forward pass to get target class
        output = self.forward(x)
        class_idx = output.argmax(dim=1)

        # Create one-hot vector for target class
        one_hot = torch.zeros_like(output)
        one_hot[torch.arange(output.size(0)), class_idx] = 1

        # Backward pass to get gradients and activations from hooks
        output.backward(gradient=one_hot, retain_graph=True)
        gradients = self.gradients
        activations = self.activations

        # 1. Compute importance weights (global average pooling of gradients)
        weights = gradients.mean(dim=2, keepdim=True)
        # weights = gradients.abs().mean(dim=2, keepdim=True)
        # weights = gradients.max(dim=2, keepdim=True)[0]
        # weights = F.relu(gradients).mean(dim=2, keepdim=True)

        # 2. Weight activations by importance and sum all features
        cam = (weights * activations).sum(dim=2)  # [1, vit_tokens, dim] -> [1, vit_tokens]

        # 3. Remove CLS token and process patches only
        cam = cam[:, 1:]  # [1, vit_tokens-1]

        # 4. Reshape to 3D grid of patches
        cam_size = grid_size // patch_size
        cam = cam.reshape(1, cam_size, cam_size, cam_size)

        # 5. Normalize cam
        cam = F.relu(cam)
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)  # [0, 1]
        threshold_value = np.percentile(cam, 100 - threshold)
        thresholded_map = np.where(cam >= threshold_value, cam, 0)
        thresholded_map = torch.from_numpy(thresholded_map).unsqueeze(0)

        # 6. Upsample to original size
        cam_3d = F.interpolate(
            thresholded_map,
            size=(grid_size, grid_size, grid_size),
            mode="trilinear",
            align_corners=False,
        ).squeeze()

        return cam_3d, class_idx

    def visualize_slice(self, cam_3d, original_volume, slice_dim=0, slice_idx=None):
        """Extracts a 2D slice from the 3D attention map and the original fMRI volume for visualization.

        Args:
            cam_3d (torch.Tensor): The 3D attention map generated by `get_attention_map`.
                Expected shape: (H, W, D).
            original_volume (torch.Tensor): The original 3D fMRI volume.
                Expected shape: (batch_size, H, W, D).
            slice_dim (int, optional): The dimension along which to slice.
                0 for axial (slice along depth),
                1 for coronal (slice along width),
                2 for sagittal (slice along height).
                Defaults to 0.
            slice_idx (int, optional): The index of the slice to extract.
                If `None`, the middle slice along `slice_dim` is chosen.

        Returns:
            tuple: A tuple containing:
                - img (numpy.ndarray): The 2D slice from the original volume.
                - attn (numpy.ndarray): The 2D slice from the attention map.
            Returns (None, None) if there are errors (e.g., `cam_3d` is None,
            shape mismatch, or `slice_idx` is out of bounds).
        """

        # Check if CAM is computed
        if cam_3d is None:
            print("Error: No CAM computed")
            return

        # Process original volume
        original = (
            original_volume.squeeze()
        )  # original = original_volume.squeeze().permute(2, 0, 1) # for fMRIs rehsape [H, W, D] to [D, H, W]
        original = original.detach().cpu().numpy()

        # Verify shapes
        if original.ndim != 3 or cam_3d.ndim != 3:
            print(f"Shape mismatch: original {original.shape}, CAM {cam_3d.shape}")
            return

        # Default to middle slice
        if slice_idx is None:
            slice_idx = original.shape[slice_dim] // 2
        slice_idx = max(0, min(slice_idx, original.shape[slice_dim] - 1))

        # Select slice
        try:
            if slice_dim == 0:  # Axial
                img = original[slice_idx]
                attn = cam_3d[slice_idx]
            elif slice_dim == 1:  # Coronal
                img = original[:, slice_idx]
                attn = cam_3d[:, slice_idx]
            else:  # Sagittal
                img = original[:, :, slice_idx]
                attn = cam_3d[:, :, slice_idx]
        except IndexError:
            print(f"Slice {slice_idx} out of bounds for dim {slice_dim}")
            return

        return img, attn


class ViT3DEncoder(nn.Module):
    """A PyTorch module that encapsulates the 3D Vision Transformer (ViT) model.

    This encoder prepares the input fMRI data for the 3D ViT by ensuring
    the correct tensor dimensions and then passes it through the ViT.

    Attributes:
        device (torch.device): The device (e.g., 'cpu' or 'cuda') on which the model resides.
        dropout (float): Dropout rate for the ViT.
        grid_size (int): The size of the 3D fMRI grid (e.g., 64 for 64x64x64).
        cube_size (int): The size of the cube used for defining the number of classes.
            This seems to relate to a specific task where the ViT predicts
            the position of a cube within the grid.
        patch_size (int): The patch size for the 3D ViT.
        num_cubes (int): The total number of possible cube positions in the grid,
            calculated as (grid_size // cube_size) ** 3. This defines
            the number of output classes for the ViT.
        vit3d (ViT): The 3D Vision Transformer model instance.
    """

    def __init__(self, config):
        super().__init__()
        self.device = config["device"]
        self.dropout = config["dropout"]
        self.grid_size = config["grid_size"]
        self.cube_size = config["cube_size"]
        self.patch_size = config["vit_patch_size"]
        self.num_cubes = (self.grid_size // self.cube_size) ** 3  # num_cubes is number of possible positions of the cube in the grid

        self.vit3d = ViT(
            channels=1,
            image_size=self.grid_size,
            image_patch_size=self.patch_size,
            frames=self.grid_size,
            frame_patch_size=self.patch_size,
            num_classes=self.num_cubes,
            dim=1024,
            depth=6,
            heads=8,
            mlp_dim=2048,
            dropout=self.dropout,
            emb_dropout=self.dropout,
            pool="cls",  # works better than mean pooling
        ).to(self.device)

    def forward(self, x):
        # x is a 3D tensor of shape (batch_size, H, W, D)
        # ViT3D expects (batch_size, channels, frames, height, width)
        volume = x.to(self.device)
        # volume = volume.permute(0, 3, 1, 2) # for fMRIs reshape [batch, H, W, D] to [batch, D, H, W]
        volume = volume.unsqueeze(1)  # Add channel dimension: [batch, H, W, D] -> [batch, 1, H, W, D]

        encoding = self.vit3d(volume)  # output is [batch, dim]
        return encoding

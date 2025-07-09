import pytest
import torch
import torch.nn as nn
import numpy as np
from unittest.mock import Mock, MagicMock, patch
import sys
from io import StringIO

# Mock the vit_pytorch module since it's not available in the test environment
sys.modules["vit_pytorch"] = Mock()
sys.modules["vit_pytorch.vit_3d"] = Mock()


# Mock ViT class
class MockViT(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.num_classes = kwargs.get("num_classes", 8)
        self.dim = kwargs.get("dim", 1024)
        self.depth = kwargs.get("depth", 6)
        self.heads = kwargs.get("heads", 8)

        # Create mock transformer with layers
        self.transformer = Mock()
        self.transformer.layers = []

        # Create mock attention layers
        for i in range(self.depth):
            layer = Mock()
            attention = Mock()
            attention.norm = Mock()
            layer.append(attention)
            self.transformer.layers.append([attention])

        # Mock the final linear layer
        self.linear = nn.Linear(self.dim, self.num_classes)

    def forward(self, x):
        batch_size = x.size(0)
        # Return logits for classification
        return torch.randn(batch_size, self.num_classes, requires_grad=True)

    def to(self, device):
        return self


# Replace the imported ViT with our mock
sys.modules["vit_pytorch.vit_3d"].ViT = MockViT

# Now import the classes to test
from fmriEncoder import fmriEncoder, ViT3DEncoder


class TestViT3DEncoder:
    """Test cases for ViT3DEncoder class."""

    @pytest.fixture
    def config(self):
        """Standard configuration for testing."""
        return {"device": "cpu", "dropout": 0.1, "grid_size": 64, "cube_size": 8, "vit_patch_size": 8, "threshold": 90}

    @pytest.fixture
    def encoder(self, config):
        """Create a ViT3DEncoder instance for testing."""
        return ViT3DEncoder(config)

    def test_init(self, config):
        """Test initialization of ViT3DEncoder."""
        encoder = ViT3DEncoder(config)

        assert encoder.device == config["device"]
        assert encoder.dropout == config["dropout"]
        assert encoder.grid_size == config["grid_size"]
        assert encoder.cube_size == config["cube_size"]
        assert encoder.patch_size == config["vit_patch_size"]
        assert encoder.num_cubes == (config["grid_size"] // config["cube_size"]) ** 3
        assert encoder.vit3d is not None

    def test_num_cubes_calculation(self):
        """Test that num_cubes is calculated correctly."""
        config = {
            "device": "cpu",
            "dropout": 0.1,
            "grid_size": 64,
            "cube_size": 8,
            "vit_patch_size": 8,
            "threshold": 90,
        }
        encoder = ViT3DEncoder(config)
        expected_num_cubes = (64 // 8) ** 3  # 8^3 = 512
        assert encoder.num_cubes == expected_num_cubes

    def test_forward_input_shape(self, encoder):
        """Test forward pass with correct input shape."""
        batch_size = 2
        grid_size = encoder.grid_size
        x = torch.randn(batch_size, grid_size, grid_size, grid_size)

        output = encoder(x)

        assert output.shape == (batch_size, encoder.num_cubes)
        assert output.dtype == torch.float32

    def test_forward_single_batch(self, encoder):
        """Test forward pass with single batch."""
        grid_size = encoder.grid_size
        x = torch.randn(1, grid_size, grid_size, grid_size)

        output = encoder(x)

        assert output.shape == (1, encoder.num_cubes)

    def test_forward_dimension_transformation(self, encoder):
        """Test that input dimensions are correctly transformed."""
        batch_size = 1
        grid_size = encoder.grid_size
        x = torch.randn(batch_size, grid_size, grid_size, grid_size)

        with patch.object(
            encoder.vit3d, "forward", return_value=torch.randn(batch_size, encoder.num_cubes)
        ) as mock_forward:
            encoder(x)

            # Check that vit3d was called with correct shape
            called_args = mock_forward.call_args[0][0]
            expected_shape = (batch_size, 1, grid_size, grid_size, grid_size)
            assert called_args.shape == expected_shape

    def test_device_handling(self):
        """Test that device is handled correctly."""
        config = {
            "device": "cpu",
            "dropout": 0.1,
            "grid_size": 32,
            "cube_size": 4,
            "vit_patch_size": 4,
            "threshold": 90,
        }
        encoder = ViT3DEncoder(config)

        x = torch.randn(1, 32, 32, 32)
        output = encoder(x)

        assert output.device.type == "cpu"


class TestFmriEncoder:
    """Test cases for fmriEncoder class."""

    @pytest.fixture
    def config(self):
        """Standard configuration for testing."""
        return {"device": "cpu", "dropout": 0.1, "grid_size": 64, "cube_size": 8, "vit_patch_size": 8, "threshold": 90}

    @pytest.fixture
    def encoder(self, config):
        """Create a fmriEncoder instance for testing."""
        return fmriEncoder(config)

    def test_init(self, config):
        """Test initialization of fmriEncoder."""
        encoder = fmriEncoder(config)

        assert encoder.config == config
        assert encoder.device == config["device"]
        assert isinstance(encoder.volume_encoder, ViT3DEncoder)
        assert encoder.gradients == {}
        assert encoder.activations == {}
        assert encoder.forward_handle is not None
        assert encoder.backward_handle is not None

    def test_register_hooks(self, encoder):
        """Test that hooks are registered correctly."""
        # Check that hooks are registered
        assert encoder.forward_handle is not None
        assert encoder.backward_handle is not None

        # Test that hooks can be removed
        encoder.forward_handle.remove()
        encoder.backward_handle.remove()

    def test_forward_pass(self, encoder):
        """Test forward pass of fmriEncoder."""
        batch_size = 2
        grid_size = encoder.config["grid_size"]
        x = torch.randn(batch_size, grid_size, grid_size, grid_size)

        output = encoder(x)

        expected_num_classes = (grid_size // encoder.config["cube_size"]) ** 3
        assert output.shape == (batch_size, expected_num_classes)

    def test_forward_single_volume(self, encoder):
        """Test forward pass with single volume."""
        grid_size = encoder.config["grid_size"]
        x = torch.randn(1, grid_size, grid_size, grid_size)

        output = encoder(x)

        expected_num_classes = (grid_size // encoder.config["cube_size"]) ** 3
        assert output.shape == (1, expected_num_classes)

    @patch("sys.stdout", new_callable=StringIO)
    def test_get_attention_map(self, mock_stdout, encoder):
        """Test attention map generation."""
        grid_size = encoder.config["grid_size"]
        x = torch.randn(1, grid_size, grid_size, grid_size)

        # Mock the activations and gradients
        patch_size = encoder.config["vit_patch_size"]
        cam_size = grid_size // patch_size
        num_patches = cam_size**3

        encoder.activations = torch.randn(1, num_patches + 1, 1024)  # +1 for CLS token
        encoder.gradients = torch.randn(1, num_patches + 1, 1024)

        cam_3d, class_idx = encoder.get_attention_map(x)

        # Check output shapes
        assert cam_3d.shape == (grid_size, grid_size, grid_size)
        assert class_idx.shape == (1,)

    def test_get_attention_map_values(self, encoder):
        """Test that attention map values are in correct range."""
        grid_size = encoder.config["grid_size"]
        x = torch.randn(1, grid_size, grid_size, grid_size)

        # Mock the activations and gradients
        patch_size = encoder.config["vit_patch_size"]
        cam_size = grid_size // patch_size
        num_patches = cam_size**3

        encoder.activations = torch.randn(1, num_patches + 1, 1024)
        encoder.gradients = torch.randn(1, num_patches + 1, 1024)

        cam_3d, class_idx = encoder.get_attention_map(x)

        # Check that values are non-negative (after ReLU and thresholding)
        assert torch.all(cam_3d >= 0)
        assert torch.all(cam_3d <= 1)  # Should be normalized to [0, 1]

    def test_visualize_slice_default(self, encoder):
        """Test slice visualization with default parameters."""
        grid_size = encoder.config["grid_size"]
        cam_3d = torch.randn(grid_size, grid_size, grid_size)
        original_volume = torch.randn(1, grid_size, grid_size, grid_size)

        img, attn = encoder.visualize_slice(cam_3d, original_volume)

        assert img is not None
        assert attn is not None
        assert img.shape == (grid_size, grid_size)
        assert attn.shape == (grid_size, grid_size)

    def test_visualize_slice_different_dimensions(self, encoder):
        """Test slice visualization with different slice dimensions."""
        grid_size = encoder.config["grid_size"]
        cam_3d = torch.randn(grid_size, grid_size, grid_size)
        original_volume = torch.randn(1, grid_size, grid_size, grid_size)

        for slice_dim in [0, 1, 2]:
            img, attn = encoder.visualize_slice(cam_3d, original_volume, slice_dim=slice_dim)

            assert img is not None
            assert attn is not None
            assert img.shape == (grid_size, grid_size)
            assert attn.shape == (grid_size, grid_size)

    def test_visualize_slice_custom_index(self, encoder):
        """Test slice visualization with custom slice index."""
        grid_size = encoder.config["grid_size"]
        cam_3d = torch.randn(grid_size, grid_size, grid_size)
        original_volume = torch.randn(1, grid_size, grid_size, grid_size)

        slice_idx = 10
        img, attn = encoder.visualize_slice(cam_3d, original_volume, slice_idx=slice_idx)

        assert img is not None
        assert attn is not None

    def test_visualize_slice_boundary_conditions(self, encoder):
        """Test slice visualization with boundary slice indices."""
        grid_size = encoder.config["grid_size"]
        cam_3d = torch.randn(grid_size, grid_size, grid_size)
        original_volume = torch.randn(1, grid_size, grid_size, grid_size)

        # Test with slice index at boundaries
        img, attn = encoder.visualize_slice(cam_3d, original_volume, slice_idx=0)
        assert img is not None
        assert attn is not None

        img, attn = encoder.visualize_slice(cam_3d, original_volume, slice_idx=grid_size - 1)
        assert img is not None
        assert attn is not None

        # Test with out of bounds index (should be clamped)
        img, attn = encoder.visualize_slice(cam_3d, original_volume, slice_idx=grid_size + 10)
        assert img is not None
        assert attn is not None

    def test_visualize_slice_none_cam(self, encoder):
        """Test slice visualization with None CAM."""
        grid_size = encoder.config["grid_size"]
        original_volume = torch.randn(1, grid_size, grid_size, grid_size)

        result = encoder.visualize_slice(None, original_volume)

        assert result is None

    def test_visualize_slice_wrong_dimensions(self, encoder):
        """Test slice visualization with wrong dimensions."""
        grid_size = encoder.config["grid_size"]
        cam_3d = torch.randn(grid_size, grid_size)  # Wrong dimensions (2D instead of 3D)
        original_volume = torch.randn(1, grid_size, grid_size, grid_size)

        result = encoder.visualize_slice(cam_3d, original_volume)

        assert result is None

    def test_device_consistency(self, encoder):
        """Test that all operations maintain device consistency."""
        grid_size = encoder.config["grid_size"]
        x = torch.randn(1, grid_size, grid_size, grid_size)

        # Ensure input is on correct device
        x = x.to(encoder.device)

        output = encoder(x)

        assert output.device.type == encoder.device


class TestErrorHandling:
    """Test error handling and edge cases."""

    def test_invalid_config_device(self):
        """Test with invalid device configuration."""
        config = {
            "device": "invalid_device",
            "dropout": 0.1,
            "grid_size": 64,
            "cube_size": 8,
            "vit_patch_size": 8,
            "threshold": 90,
        }

        # Should handle gracefully or raise appropriate error
        with pytest.raises(Exception):
            encoder = fmriEncoder(config)

    def test_mismatched_grid_and_patch_sizes(self):
        """Test with grid size not divisible by patch size."""
        config = {
            "device": "cpu",
            "dropout": 0.1,
            "grid_size": 65,  # Not divisible by patch_size
            "cube_size": 8,
            "vit_patch_size": 8,
            "threshold": 90,
        }

        # Should handle gracefully or raise appropriate error
        encoder = fmriEncoder(config)
        assert encoder is not None

    def test_zero_threshold(self):
        """Test with zero threshold value."""
        config = {
            "device": "cpu",
            "dropout": 0.1,
            "grid_size": 64,
            "cube_size": 8,
            "vit_patch_size": 8,
            "threshold": 0,
        }

        encoder = fmriEncoder(config)
        assert encoder.config["threshold"] == 0


class TestIntegration:
    """Integration tests for the complete workflow."""

    @pytest.fixture
    def config(self):
        """Configuration for integration tests."""
        return {
            "device": "cpu",
            "dropout": 0.1,
            "grid_size": 32,  # Smaller for faster tests
            "cube_size": 4,
            "vit_patch_size": 4,
            "threshold": 90,
        }

    def test_full_workflow(self, config):
        """Test the complete workflow from input to visualization."""
        encoder = fmriEncoder(config)
        grid_size = config["grid_size"]

        # Create input
        x = torch.randn(1, grid_size, grid_size, grid_size)

        # Forward pass
        output = encoder(x)
        assert output.shape[0] == 1

        # Mock activations and gradients for attention map
        patch_size = config["vit_patch_size"]
        cam_size = grid_size // patch_size
        num_patches = cam_size**3

        encoder.activations = torch.randn(1, num_patches + 1, 1024)
        encoder.gradients = torch.randn(1, num_patches + 1, 1024)

        # Get attention map
        cam_3d, class_idx = encoder.get_attention_map(x)
        assert cam_3d.shape == (grid_size, grid_size, grid_size)

        # Visualize slice
        img, attn = encoder.visualize_slice(cam_3d, x)
        assert img is not None
        assert attn is not None

    def test_batch_processing(self, config):
        """Test processing multiple volumes in a batch."""
        encoder = fmriEncoder(config)
        grid_size = config["grid_size"]
        batch_size = 3

        # Create batch input
        x = torch.randn(batch_size, grid_size, grid_size, grid_size)

        # Forward pass
        output = encoder(x)
        assert output.shape[0] == batch_size

        # Note: get_attention_map is designed for single volume
        # Test with single volume from batch
        single_volume = x[:1]

        # Mock activations and gradients
        patch_size = config["vit_patch_size"]
        cam_size = grid_size // patch_size
        num_patches = cam_size**3

        encoder.activations = torch.randn(1, num_patches + 1, 1024)
        encoder.gradients = torch.randn(1, num_patches + 1, 1024)

        cam_3d, class_idx = encoder.get_attention_map(single_volume)
        assert cam_3d.shape == (grid_size, grid_size, grid_size)
        assert class_idx.shape == (1,)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

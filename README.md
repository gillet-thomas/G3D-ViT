# G3D-ViT: 3D GradCAM for Vision Transformers

## Overview
G3D-ViT is a **3D GradCAM implementation for Vision Transformers (ViTs)**, designed to visualize important regions in 3D data by leveraging gradient-based activation maps. This method is particularly useful for understanding model decisions in **fMRI-based classification tasks**.

## Data Input
- The input data consists of **fMRI timepoints**.
- These are **reshaped into 90 × 90 × 90 3D volumes** before being processed by the ViT.

## Custom 3DViT GradCAM
G3D-ViT registers **hooks** to capture **activations** during the forward pass and **gradients** during the backward pass.

- **Forward pass**: Reveals *where* the model detected patterns (**activations**). This answers the question: *"What did the model see?"*
- **Backward pass**: Highlights *importance* (**gradients**) by showing *how much* each feature influenced the decision.

## Implementation Details
1. **Hook Registration**
   - Hooks are registered on the **last normalization layer** of the ViT.
   - Backpropagation is applied to the **class value obtained after the projection head** (producing a vector of size `[batch, 2]` instead of `[batch, 1024]`).

2. **Gradient-Based Weighting**
   - Compute **gradient importance scores** by averaging gradients along the feature dimension.
   - This results in a **single importance score** (**weight**) per spatial location (patch).

3. **Activation Scaling**
   - Multiply each patch’s **weight** by its **1024 activations**.
   - Sum across activation dimensions to obtain a **single attention score per patch**.

## Normalization Methods
For normalizing the **weights**, any of the following four methods can be used (**default is global average pooling**):

```python
weights = gradients.mean(dim=2, keepdim=True)  # Global Average Pooling (Default)
weights = gradients.max(dim=2, keepdim=True)[0]  # Max Pooling
weights = gradients.abs().mean(dim=2, keepdim=True)  # Absolute Mean Pooling
weights = F.relu(gradients).mean(dim=2, keepdim=True)  # ReLU Activation Pooling
```

## Usage
1. Prepare **fMRI data** as 90 × 90 × 90 3D volumes.
2. Pass data through a **Vision Transformer model**.
3. Register **hooks** on the last normalization layer.
4. Compute **GradCAM activation maps** using backpropagation.
5. Visualize **important regions** in 3D space.

## Applications
- **Medical imaging**: Understanding brain activity in **fMRI-based classification**.
- **Neuroscience**: Identifying **regions of interest (ROIs)** in brain scans.
- **Model interpretability**: Gaining insights into how ViTs process **spatial patterns in 3D data**.

---
This implementation allows for an intuitive visualization of feature importance in 3D, making Vision Transformers more interpretable for fMRI and medical imaging applications.


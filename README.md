# G3D-ViT: 3D GradCAM for Vision Transformers

## Overview

G3D-ViT is a **3D GradCAM implementation for Vision Transformers (ViTs)**, designed to visualize important regions in 3D data by leveraging gradient-based activation maps. This method is particularly useful for understanding model decisions in **fMRI-based classification tasks**.

## Origin

This project originated during the development of **fMRI2Vec**. After implementing a 3D Vision Transformer to encode fMRI timepoints and classify them, a need emerged for an interpretability method that could highlight **regions of interest (ROIs)** the ViT focused on during classification. G3D-ViT was developed to fill that gap and provide insights into **which brain regions the model "looked at"** when predicting outcomes such as **gender or age** on the fMRI timepoints.

## State of the Art - Class Activation Maps

Since GradCAM and other explainability tools were originally designed for **CNNs**, a **3D ResNet** was trained to match the performance of the 3D ViT. This allowed testing of standard CNN-based interpretability methods on the ResNet, creating a **ground truth class activation map**. This ground truth would be useful to then replicate that map using existing or custom techniques adapted for the 3D ViT.

To evaluate the correctness of the tested approaches, two main criteria were used: the highlighted regions should be within the brain (inside the skull), and for age classification tasks, regions such as the ventricles should be highlighted since ventricular enlargement is a known biomarker that correlates with aging.

### Evaluation of CNN Interpretability Methods

Here are all the CNN interpretability methods* that were evaluated:  

- **pytorch-grad-cam GradCAM**: Not consistent across subjects; highlighted regions often outside the brain.
- **pytorch-grad-cam LayerCAM**: Consistent results, effective for both age and gender.
- **pytorch-grad-cam GradCAMElementWise**: Similar to LayerCAM; consistent and reliable.
- **pytorch-grad-cam HiResCAM, EigenCAM, EigenGradCAM**: Worked well for age prediction, not for gender.
- **pytorch-grad-cam GradCAM++, FullGrad, ScoreCAM, XGradCAM**: Did not work as expected due to reshaping issues.
- **CAPTUM LayerGradCam**: Not consistent across subjects; highlighted regions outside the brain.
- **CAPTUM Integrated Gradients**: Produced random-looking attributions.
- **SHAP (on 3D ResNet50)**: Did not yield interpretable results.

**pytorch-grad-cam GradCAM**, **LayerCAM**, and **GradCAMElementWise** came out as the most accurate methods, with consistent results for both age and gender classification.

\*_methods tested on a trained 3D ResNet achieving 100% accuracy on training and 98% on validation for age and gender classification_

### Evaluation of Vision Transformer (ViT) Interpretability Methods

After generating a reliable ground truth class activation map using a 3D ResNet model, I evaluated several interpretability methods tailored for Vision Transformers. The goal was to assess how well these methods could replicate the attention patterns from the 3D ResNet model. Here is a summary of the tested ViT explainability methods* and their results:  

- **GradCAM**  
  → *Best-performing and most practical option for ViTs.*

- **Beyond Attention** [Paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Chefer_Transformer_Interpretability_Beyond_Attention_Visualization_CVPR_2021_paper.pdf) [GitHub](https://github.com/hila-chefer/Transformer-Explainability/tree/main)  
  - Requires using a custom ViT-LRP model provided by the authors.  
  - Model retraining is necessary, and scaling to 3D adds complexity.

- **Partial LRP** [Paper](https://www.google.com/search?client=firefox-b-d&q=Analyzing+multi-head+self-attention%3A+Spe-+cialized+heads+do+the+heavy+lifting%2C+the+rest+can+be+pruned)  
  - Promising approach, but the referenced paper is not specifically focused on Partial LRP.  
  - Typically requires full LRP as a prerequisite, increasing integration overhead.

- **Rollout** [GitHub](https://github.com/jacobgil/vit-explain)  
  - Code is not functional out of the box, there is a problem with the `attentions` parameter in the `rollout` function, even in the official GitHub repository version.

- **LeGrad** [GitHub](https://github.com/WalBouss/LeGrad)  
  - Depends on OpenCLIP and supports only a fixed set of pretrained models.  
  - Incompatible with custom ViT architectures.

- **Transformers-Interpret** [GitHub](https://github.com/cdpierse/transformers-interpret?tab=readme-ov-file)  
  - Built for Hugging Face's transformer models; does not support custom ViTs or 3D data.
  
None of the tested methods for 3D ViT enabled the reproduction of the ground truth activation map generetad on the 3D ResNet, motivating the development of a custom solution.

\*_methods tested on a trained 3D ViT achieving 100% accuracy on training and 90%+ on validation for age and gender classification)_
  
## G3D-ViT Implementation Details
G3D-ViT works on a trained 3D ViT for classification. It registers **hooks** to capture **activations** during the forward pass of the model and **gradients** during the backward pass.

The 3D Vision Transformer model used in this project is a direct implementation of the [3D ViT by lucidrains](https://github.com/lucidrains/vit-pytorch?tab=readme-ov-file#3d-vit), which was used without modification to encode and classify the fMRI volumes.

1. **Hook Registration and Calling**
Both hooks are registered on the **last normalization layer** of the ViT (`vit3d.transformer.layers[-1][0].norm`).  
This specific layer was chosen because it consistently produced the most interpretable results compared to other layers.  
This finding aligns with recommendations from [jacobgil's guide on GradCAM for Vision Transformers](https://jacobgil.github.io/pytorch-gradcam-book).

- **Forward hook** captures the output of the last normalization layer, revealing *where* the model detected patterns (**activations**). This forward hook is called during the forward pass triggered by the classification of the input sample.
- **Backward hook** captures the gradient of the loss (after backpropgating the logit of the predicted class), revealing the importance (**gradients**) of each parameter for the final prediction of that class. This backward hook is called during the backward pass which is manually triggered in the `get_attention_map` function.

2. **Gradient-Based Weighting Computation**
   - Compute **gradient importance scores** by averaging gradients along the feature dimension.
   - This results in a **single importance score** (**weight**) per spatial location (ViT patch).

3. **Activation Scaling**
   - Weight each path's activatoins by multiplying them with its **weight**.
   - Sum across patch activation dimension to obtain a **single attention score per patch**.

4. **Create final 3D GradCAM**
   - Remove the CLS token to process the patches value only.
   - Reshape patches to 3D grid.
   - Normalize values with relu and scale between 0 and 1.
5. **Thresholding** (optional)
To refine the attention maps, an optional **thresholding** step is included in the `get_attention_map` function. By default, only the **top 10% of gradients** are kept to focus on the most important regions.

### Normalization Methods
For **normalizing the weights**, the following four methods are available:

```python
weights = gradients.mean(dim=2, keepdim=True)  # Global Average Pooling (Default)
weights = gradients.max(dim=2, keepdim=True)[0]  # Max Pooling
weights = gradients.abs().mean(dim=2, keepdim=True)  # Absolute Mean Pooling
weights = F.relu(gradients).mean(dim=2, keepdim=True)  # ReLU Activation Pooling
```

In practice, Max Pooling and ReLU Activation Pooling provided the most reliable results. Global Average Pooling was inconsistent across tasks and Absolute Mean Pooling was discarded, as it is not accurate to assign equal importance to both positive and negative gradients.

For all evaluations, ReLU Activation Pooling was used by default, followed by a thresholding to keep the 10% most important gradients.

## Usage
The input data consists of **fMRI timepoints (volumes)** which are **reshaped into 90 × 90 × 90 3D volumes** before being processed by the 3D ViT.

1. Prepare **fMRI data** as 90 × 90 × 90 3D volumes.
2. Train the 3D ViT on a classification task.
3. Compute the **GradCAM activation maps** using the trained model.
4. Visualize **important regions** in 3D space.

### Applications

- **Medical imaging**: Understanding brain activity in **fMRI-based classification**.
- **Neuroscience**: Identifying **regions of interest (ROIs)** in brain scans.
- **Model interpretability**: Gaining insights into how ViTs process **spatial patterns in 3D data**.

## Validation: Mock Dataset Experiment
To verify G3D-ViT's accuracy, a mock dataset was created:

### Dataset Description
- A large 3D cube filled with zeros.
- A smaller target cube filled with ones embedded inside.
- Two classification tasks:
   1. Positional Classification: Assigns a class based on the target cube’s location.
   2. Binary Content Classification: Target cube is filled with -1 or 1, label is 0 or 1 accordingly.

### Experiment Flow
1. Train a 3D ViT on this mock dataset.
2. Achieve 100% accuracy on both training and validation.
3. Run G3D-ViT to verify that the attention maps highlight the correct regions.
4. Evaluate interpretability on both:
   - Position-based classification (tests spatial sensitivity).
   - Value-based classification (tests non-spatial sensitivity).
  
### Summary
G3D-ViT extends traditional GradCAM to 3D data and is tailored for Vision Transformers applied to fMRI. It addresses the shortcomings of existing tools by offering:
- 3D-aware interpretability.
- Support for ViT-specific architecture.
- Verified reliability via a controlled mock dataset.

---

This implementation allows for an intuitive visualization of feature importance in 3D, making Vision Transformers more interpretable for fMRI and medical imaging applications.


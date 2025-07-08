import os
import warnings
from datetime import datetime

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import torch
import yaml

from data.DatasetGradCAM import GradCAMDataset
from fmriEncoder import fmriEncoder


def get_sample_gradcam(id, save_sample_attention=False):
    """Retrieves a sample, computes its Grad-CAM attention map, and optionally saves it.

    This function fetches a specific sample from the dataset, passes it through
    the `fmriEncoder` to obtain the attention map, and can save the 3D attention
    map as a NIfTI file and a 3D scatter plot.

    Args:
        id (int): The index of the sample to retrieve from the dataset.
        save_sample_attention (bool, optional): If True, saves the 3D attention map
            as a NIfTI file and a 3D scatter plot. Defaults to False.

    Returns:
        tuple: A tuple containing:
            - id (int): The ID of the processed sample.
            - img (numpy.ndarray): A 2D slice of the original input volume.
            - attn (numpy.ndarray): A 2D slice of the attention map.
            - class_idx (torch.Tensor): The predicted class index for the sample.
            - sample_label (torch.Tensor): The true label of the sample.
    """

    sample = dataset[id]  # sample is tuple (input_tensor, label, coordinates)
    input_tensor = sample[0].to(config["device"]).unsqueeze(0)
    print(f"ID: {id} - Label: {sample[1].item()}, Coordinates: {sample[2].tolist()}")

    cube_size = config["cube_size"]
    patch_x = int(sample[2][0] + cube_size // 2)
    patch_y = int(sample[2][1] + cube_size // 2)
    patch_z = int(sample[2][2] + cube_size // 2)
    # print(f"Patch coordinates: {patch_x}, {patch_y}, {patch_z}")

    # Get attention map
    attention_map, class_idx = model.get_attention_map(input_tensor)
    img, attn = model.visualize_slice(attention_map, input_tensor, slice_dim=0, slice_idx=patch_x)

    if save_sample_attention:
        nib.save(
            nib.Nifti1Image(attention_map.cpu().numpy(), np.eye(4)),
            f'{config["output_dir"]}/DatasetGradCAM_3Dattention_{id}.nii',
        )
        save_gradcam_3d(attention_map, id, sample)

    return id, img, attn, class_idx, sample[1]


def create_gradcam_plot(save_sample_attention=False):
    """Generates and saves a combined plot of Grad-CAM visualizations for multiple samples.

    This function iterates through a predefined list of sample IDs, computes
    their Grad-CAM attention maps, and then arranges these visualizations
    into a single matplotlib figure, which is saved as a PNG image.

    Args:
        save_sample_attention (bool, optional): If True, also calls `save_gradcam_3d`
            for each sample to save individual 3D attention maps. Defaults to False.
    """

    results = [get_sample_gradcam(id, save_sample_attention=save_sample_attention) for id in ids]

    # Create combined plot
    n = len(results)
    cols = 4
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(20, 5 * rows))
    fig.suptitle(
        f'DatasetGradCAM {config["grid_size"]}grid {config["cube_size"]}cube {config["vit_patch_size"]}patch {config["grid_noise"]}noise',
        fontsize=16,
    )

    # Plot each subject's results
    for idx, (ID, image, attention, class_idx, sample) in enumerate(results):
        row = idx // cols
        col = idx % cols
        ax = axes[col] if rows == 1 else axes[row, col]

        ax.imshow(
            -image + 1 if config["grid_noise"] < 1 else image, cmap="gray"
        )  # INVERSE BRIGHTNESS
        heatmap = ax.imshow(attention, cmap="jet", alpha=0.4)
        fig.colorbar(heatmap, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(f"Subject {ID} (Class {class_idx.item()})")
        ax.axis("off")

    # Hide empty subplots
    for idx in range(n, rows * cols):
        row = idx // cols
        col = idx % cols
        (axes[col] if rows == 1 else axes[row, col]).axis("off")

    file_name = f'DatasetGradCAM_{config["grid_size"]}grid_{config["cube_size"]}cube_{config["vit_patch_size"]}patch_{config["grid_noise"]}noise_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}'.replace(
        ".", "p"
    )
    plt.tight_layout()
    plt.savefig(os.path.join(config["output_dir"], f"{file_name}.png"), dpi=300)
    plt.close()
    print(f"All results saved to {file_name}.png")


def save_gradcam_3d(attention_map, id, sample):
    """Saves a 3D scatter plot visualization of the Grad-CAM attention map.

    This function creates a 3D scatter plot where points represent regions
    with attention values above a certain threshold, colored by their attention intensity.
    The plot is saved as a PNG image.

    Args:
        attention_map (torch.Tensor): The 3D attention map tensor.
        id (int): The ID of the sample associated with this attention map.
        sample (tuple): The original sample tuple containing (input_tensor, label, coordinates).
    """

    # Create 3D plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    threshold = 0.2
    coords = np.argwhere(attention_map.cpu().numpy() > threshold)
    values = attention_map.cpu().numpy()[attention_map.cpu().numpy() > threshold]

    # Scatter plot for the regions with attention above the threshold
    if coords.size > 0:
        sc = ax.scatter(
            coords[:, 0],
            coords[:, 1],
            coords[:, 2],
            c=values,
            cmap="jet",
            marker="s",
            alpha=0.6,
            s=50,
        )
        fig.colorbar(sc, ax=ax, shrink=0.5, aspect=10, label="Attention Value")
    else:
        print(f"No attention values above threshold {threshold} for sample {id}")

    # Add a bounding box for the volume
    ax.set(
        xlim=(0, attention_map.shape[0]),
        ylim=(0, attention_map.shape[1]),
        zlim=(0, attention_map.shape[2]),
    )  # Grid size
    ax.set(xlabel="X axis", ylabel="Y axis", zlabel="Z axis")

    # Save figure and nifti file
    save_path = config["output_dir"]
    file_name = f'DatasetGradCAM_{config["grid_size"]}grid_{config["cube_size"]}cube_{config["grid_noise"]}noise_3Dattention_{id}'.replace(
        ".", "p"
    )
    plt.title(f"3D GradCAM (Label: {sample[1].item()}, coordinates: {sample[2].tolist()})")
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f"{file_name}.png"), dpi=300)
    plt.close()


if __name__ == "__main__":
    """Main execution block for loading the model, dataset, and generating Grad-CAM plots.

    This block initializes the configuration, loads the pre-trained `fmriEncoder` model,
    sets up the `GradCAMDataset`, and then calls `create_gradcam_plot` to generate
    and save the visualizations.
    """

    # Config
    warnings.simplefilter(action="ignore", category=FutureWarning)
    config = yaml.safe_load(open("./config.yaml"))
    config["device"] = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load Model and Dataset
    model = fmriEncoder(config).to(config["device"]).eval()
    model.load_state_dict(
        torch.load(config["best_model_path"], map_location=config["device"]), strict=False
    )
    dataset = GradCAMDataset(config, mode="val", generate_data=False)

    ids = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    create_gradcam_plot(save_sample_attention=config["save_gradcam_attention"])

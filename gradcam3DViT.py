import os
import yaml
import torch
import warnings
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from datetime import datetime
import xml.etree.ElementTree as ET
from nilearn.image import load_img
from fmriEncoder import fmriEncoder


def create_itksnap_workspace(original_nifti, gradcam_nifti, output_dir):
    # Create the root element
    registry = ET.Element("registry")
    
    # Add basic entries
    ET.SubElement(registry, "entry", key="SaveLocation", value=output_dir)
    ET.SubElement(registry, "entry", key="Version", value=datetime.now().strftime("%Y%m%d"))
    
    # Add annotations
    annotations = ET.SubElement(registry, "folder", key="Annotations")
    ET.SubElement(annotations, "entry", key="Format", value="ITK-SNAP Annotation File")
    ET.SubElement(annotations, "entry", key="FormatDate", value="20150624")
    
    # Add layers
    layers = ET.SubElement(registry, "folder", key="Layers")
    
    # Main layer (original image)
    layer0 = ET.SubElement(layers, "folder", key="Layer[000]")
    ET.SubElement(layer0, "entry", key="AbsolutePath", value=original_nifti)
    ET.SubElement(layer0, "entry", key="Role", value="MainRole")
    ET.SubElement(layer0, "entry", key="Tags", value="")
    
    # Layer metadata for main image
    layer_meta0 = ET.SubElement(layer0, "folder", key="LayerMetaData")
    ET.SubElement(layer_meta0, "entry", key="Alpha", value="255")
    ET.SubElement(layer_meta0, "entry", key="CustomNickName", value="")
    ET.SubElement(layer_meta0, "entry", key="Sticky", value="0")
    
    # Display mapping for main image
    display_mapping0 = ET.SubElement(layer_meta0, "folder", key="DisplayMapping")
    color_map0 = ET.SubElement(display_mapping0, "folder", key="ColorMap")
    ET.SubElement(color_map0, "entry", key="Preset", value="Grayscale")
    
    # Overlay layer (gradcam)
    layer1 = ET.SubElement(layers, "folder", key="Layer[001]")
    ET.SubElement(layer1, "entry", key="AbsolutePath", value=gradcam_nifti)
    ET.SubElement(layer1, "entry", key="Role", value="OverlayRole")
    ET.SubElement(layer1, "entry", key="Tags", value="")
    
    # Layer metadata for overlay
    layer_meta1 = ET.SubElement(layer1, "folder", key="LayerMetaData")
    ET.SubElement(layer_meta1, "entry", key="Alpha", value="0.5")
    ET.SubElement(layer_meta1, "entry", key="CustomNickName", value="")
    ET.SubElement(layer_meta1, "entry", key="Sticky", value="1")
    
    # Display mapping for overlay
    display_mapping1 = ET.SubElement(layer_meta1, "folder", key="DisplayMapping")
    color_map1 = ET.SubElement(display_mapping1, "folder", key="ColorMap")
    ET.SubElement(color_map1, "entry", key="Preset", value="Jet")
    
    # Add TimePointProperties
    time_props = ET.SubElement(registry, "folder", key="TimePointProperties")
    ET.SubElement(time_props, "entry", key="FormatVersion", value="1")
    
    time_points = ET.SubElement(time_props, "folder", key="TimePoints")
    ET.SubElement(time_points, "entry", key="ArraySize", value="1")
    
    time_point = ET.SubElement(time_points, "folder", key="TimePoint[1]")
    ET.SubElement(time_point, "entry", key="Nickname", value="")
    ET.SubElement(time_point, "entry", key="Tags", value="")
    ET.SubElement(time_point, "entry", key="TimePoint", value="1")
    
    # Create the XML tree
    tree = ET.ElementTree(registry)
    
    # Add XML declaration and doctype
    xml_str = '<?xml version="1.0" encoding="UTF-8" ?>\n'
    xml_str += '<!--ITK-SNAP (itksnap.org) Project File-->\n\n'
    xml_str += '<!DOCTYPE registry [\n'
    xml_str += '<!ELEMENT registry (entry*,folder*)>\n'
    xml_str += '<!ELEMENT folder (entry*,folder*)>\n'
    xml_str += '<!ELEMENT entry EMPTY>\n'
    xml_str += '<!ATTLIST folder key CDATA #REQUIRED>\n'
    xml_str += '<!ATTLIST entry key CDATA #REQUIRED>\n'
    xml_str += '<!ATTLIST entry value CDATA #REQUIRED>\n'
    xml_str += ']>\n'
    xml_str += ET.tostring(registry, encoding='unicode')
    
    # Save to file
    output_path = os.path.join(output_dir, os.path.splitext(os.path.basename(original_nifti))[0] + ".itksnap")
    with open(output_path, 'w') as f:
        f.write(xml_str)
    
    return output_path

def main(ID=151, slice_dim=0, slice_idx=45):
    warnings.simplefilter(action='ignore', category=FutureWarning)

    # Load Config
    BASE_PATH = ""
    FMRI_PATH = f""
    config = yaml.safe_load(open(BASE_PATH + "/configs/config.yaml"))
    config["device"] = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    # Load Model and GradCAM
    model = fmriEncoder(config).to(config["device"])
    model.load_state_dict(torch.load(config["best_model_path"], map_location=config["device"]), strict=False)
    
    # Load and Preprocess fMRI Data
    fmri_img = load_img(FMRI_PATH)
    fmri_data = fmri_img.get_fdata(dtype=np.float32)                 # Shape: (91, 109, 91, 146)
    fmri_data = fmri_data[1:, 10:-9, 1: , 70]                        # CROP Shape: (90, 90, 90)
    fmri_norm = (fmri_data - np.mean(fmri_data)) / np.std(fmri_data) # Normalize
    input_tensor = torch.tensor(fmri_norm).to(config["device"])
    input_tensor = input_tensor.unsqueeze(0)                         # Shape (1, 90, 90, 90)

    # Ensure model is in train mode
    model.eval()
    attention_map, class_idx = model.get_attention_map(input_tensor)       # output [90, 90, 90]
    img, attn = model.visualize_slice(attention_map, input_tensor, slice_dim=slice_dim, slice_idx=slice_idx)
    nib.save(nib.Nifti1Image(attention_map, fmri_img.affine), f'{BASE_PATH}/explainability/xAi_gradcam3DViT/gender/{ID}_gradcam_3dd.nii')

    # Save fMRI image for visualization
    fmri_slice = fmri_norm[ :, :, slice_idx]        # Choose middle slice, output shape: (90, 90)
    fmri_rgb = np.stack([fmri_slice] * 3, axis=-1)  # Convert to RGB, output shape: (90, 90, 3)
    fmri_rgb = (fmri_rgb - np.min(fmri_rgb)) / (np.max(fmri_rgb) - np.min(fmri_rgb))
    nib.save(nib.Nifti1Image(fmri_data, fmri_img.affine), f'{BASE_PATH}/explainability/xAi_gradcam3DViT/gender/{ID}_fmri.nii')
    print("GradCAM completed.")

    # Create ITK-SNAP workspace with GradCAM overlayed on fMRI
    create_itksnap_workspace(f'/Downloads/{ID}_fmri.nii',
                            f'/Downloads/{ID}_gradcam_3dd.nii',
                            f'{BASE_PATH}/explainability/xAi_gradcam3DViT/')

    return ID, img, attn, class_idx


if __name__ == '__main__':
    ids = [151, 153, 154, 155, 501, 502, 503, 504, 505, 507, 508, 509, 510]
    results = []
    for i in ids:
        results.append(main(i))
        print(f"Completed {i}")

    # Create combined plot
    n = len(results)
    cols = 4
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(20, 5*rows))
    fig.suptitle('GradCAM Results Across Subjects', fontsize=16)
    
    # Plot each subject's result
    for idx, (ID, image, attention, class_idx) in enumerate(results):
        row = idx // cols
        col = idx % cols
        
        if rows == 1:
            ax = axes[col]
        else:
            ax = axes[row, col]
        
        ax.imshow(image, cmap='gray')
        heatmap = ax.imshow(attention, cmap='jet', alpha=0.4)
        fig.colorbar(heatmap, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(f'Subject {ID} (Class {class_idx.item()})')
        ax.axis('off')
    
    # Hide empty subplots
    for idx in range(n, rows*cols):
        row = idx // cols
        col = idx % cols
        if rows == 1:
            axes[col].axis('off')
        else:
            axes[row, col].axis('off')

    plt.tight_layout()
    plt.savefig(f'{BASE_PATH}/xAi_gradcam3DViT/gender/{ids[0]}_vit_gender.png')
    plt.close()
    print("All results saved in single plot.")

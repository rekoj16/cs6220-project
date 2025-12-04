import torch
import torch.nn as nn
from torchvision import models, transforms
from peft import PeftModel
from PIL import Image
import numpy as np
import cv2
import os
import matplotlib.pyplot as plot
import pandas as panda
from tqdm import tqdm
import gc
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

#setup
NUM_CLASSES = 11
CLASSIFICATIONS = [
    'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 
    'Enlarged Cardiomediastinum', 'Lung Lesion', 'Lung Opacity',
    'Pneumonia', 'Pneumothorax', 'Pleural Effusion', 'Pleural Other'
]
ADAPTER_DIR = "/home/hice1/achen448/scratch/CS6220/cs6220-project/imagenet/results/lora5/"
VALID_ROOT = "/home/hice1/achen448/scratch/CS6220/cs6220-project/dataset/CheXpert-v1.0-small/valid/"
VAL_CSV = "/home/hice1/achen448/scratch/CS6220/cs6220-project/dataset/CheXpert-v1.0-small/valid.csv"
HEATMAP_OUTPUT = "/home/hice1/achen448/scratch/CS6220/cs6220-project/imagenet/heatmaps_refined_batch1"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TOP_N_PREDICTIONS = 3

#loading the models
def load_models():
    print("loading Baseline Model")
    baseline = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
    feature_size = baseline.classifier[1].in_features
    baseline.classifier[1] = nn.Linear(in_features=feature_size, out_features=NUM_CLASSES)
    baseline.to(DEVICE)
    baseline.eval()

    print(f"loading the Finetuned Model")
    base_for_peft = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
    base_for_peft.classifier[1] = nn.Linear(in_features=feature_size, out_features=NUM_CLASSES)
    try:
        finetuned = PeftModel.from_pretrained(base_for_peft, ADAPTER_DIR)
    except Exception as e:
        print(f"Error loading adapters: {e}")
        exit()
    finetuned.to(DEVICE)
    finetuned.eval()
    
    return baseline, finetuned

#processing each image and fetching its ground truth labels
def get_image_data(image_full_path, global_df):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    try:
        orig_resized = Image.open(image_full_path).convert("RGB").resize((224, 224))
    except Exception as e:
        return None, None, None, [], []

    transform_mod = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    input_tensor = transform_mod(orig_resized).unsqueeze(0).to(DEVICE)
    rbg_img = np.array(orig_resized) / 255.0
    ground_indices = []
    ground_labels = []
    path_parts = image_full_path.replace("\\", "/").split("/")
    if len(path_parts) >= 3:
        uni_suffix = "/".join(path_parts[-3:])
    else:
        uni_suffix = image_full_path.replace("\\", "/")

    matches = global_df[global_df['Path'].str.contains(uni_suffix, regex=False)]
    
    if not matches.empty:
        row = matches.iloc[0]
        for idx, class_name in enumerate(CLASSIFICATIONS):
            if row[class_name] == 1.0 or row[class_name] == -1.0: 
                ground_indices.append(idx)
                ground_labels.append(class_name)
        if not ground_labels:
            ground_labels.append("No Positive Findings")
    else:
        ground_labels.append("N/A (Not Found in CSV)")

    return input_tensor, rbg_img, orig_resized, ground_indices, ground_labels

# get top N predictions overall
def get_overall_top_predictions(model, input_tensor, top_n=TOP_N_PREDICTIONS):
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.sigmoid(outputs).squeeze().cpu().numpy()
    
    predictions = []
    for idx in np.argsort(probabilities)[::-1][:top_n]:
        predictions.append(f"{CLASSIFICATIONS[idx]} (p={probabilities[idx]:.2f})")
    return predictions

#generating heatmap for a specific class
def generate_heatmap_overlay(model, target_layer, input_tensor, rgb_img, class_idx):
    class ModelWrapper(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
        def forward(self, x): return self.model(x)

    wrap_model = ModelWrapper(model)
    g_cam = GradCAM(model=wrap_model, target_layers=[target_layer])
    input_tensor.requires_grad_(True)
    grayscale_cam = g_cam(input_tensor=input_tensor, targets=[ClassifierOutputTarget(class_idx)])[0, :]
    visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
    
    del g_cam, wrap_model
    return visualization

def plot_and_save_comparison(pil_img, gt_labels, 
                             baseline_heatmap_vis, baseline_top_preds_text,
                             finetuned_heatmap_vis, finetuned_top_preds_text,
                             class_name_for_heatmap, image_id, save_path):
    
    fig, axes = plot.subplots(1, 3, figsize=(20, 12)) 
    axes[0].imshow(pil_img)
    axes[0].set_title(f"Original Image\nSample {image_id}", fontsize=14, pad=20)
    axes[0].axis('off')

    # ground Truth Text (Left Panel)
    gt_text_str = "Ground Truth:\n" + "\n".join([f"• {l}" for l in gt_labels])
    fig.text(0.15, 0.88, gt_text_str, fontsize=11, verticalalignment='top', 
             bbox=dict(boxstyle="square,pad=0.5", fc="white", ec="gray", alpha=0.9))
    
    # baseline Heatmap (Middle Panel)
    axes[1].imshow(baseline_heatmap_vis)
    axes[1].set_title(f"Baseline Heatmap\n(Visualizing: {class_name_for_heatmap})", fontsize=14, pad=20)
    axes[1].axis('off')
    baseline_pred_text_str = "Top 3 Predictions:\n" + "\n".join(baseline_top_preds_text)
    fig.text(0.42, 0.88, baseline_pred_text_str, fontsize=11, verticalalignment='top',
             bbox=dict(boxstyle="square,pad=0.5", fc="white", ec="gray", alpha=0.9))

    # finetuned Heatmap (Right Panel)
    axes[2].imshow(finetuned_heatmap_vis)
    axes[2].set_title(f"Finetuned Heatmap\n(Visualizing: {class_name_for_heatmap})", fontsize=14, pad=20)
    axes[2].axis('off')
    finetuned_pred_text_str = "Top 3 Predictions:\n" + "\n".join(finetuned_top_preds_text)
    fig.text(0.70, 0.88, finetuned_pred_text_str, fontsize=11, verticalalignment='top',
             bbox=dict(boxstyle="square,pad=0.5", fc="white", ec="gray", alpha=0.9))
    plot.tight_layout(rect=[0.02, 0.02, 0.98, 0.82]) 
    
    plot.savefig(save_path, bbox_inches='tight', dpi=150)
    plot.close(fig)
# Entry point for batch processing and heatmap generation

if __name__ == "__main__":
    print("refined Batch heatmap generation started")
    print(f"loading valid csv ...")
    GLOBAL_DF = panda.read_csv(VAL_CSV)

    baseline_model, finetuned_model = load_models()
    baseline_target_layer = baseline_model.features[-1] 
    finetuned_target_layer = finetuned_model.model.features[-1]

    print(f"scanning for images in {VALID_ROOT}...")
    all_image_paths = []
    for root, dirs, files in os.walk(VALID_ROOT):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                all_image_paths.append(os.path.join(root, file))
    
    print(f"Found {len(all_image_paths)} images. Starting processing")

    for img_path in tqdm(all_image_paths, desc="Processing Images"):
        
        input_tensor, rgb_img, p_img, ground_indices, ground_labels = get_image_data(img_path, GLOBAL_DF)
        if input_tensor is None: continue

        # get top 3 predictions per image
        baseline_top_preds_formatted = get_overall_top_predictions(baseline_model, input_tensor.clone())
        finetuned_top_preds_formatted = get_overall_top_predictions(finetuned_model, input_tensor.clone())
        
        # determine relevant classes for heatmap generation
        # union of Ground Truth indices and Top N predicted indices from both models
        _, base_model_top_n = torch.topk(torch.sigmoid(baseline_model(input_tensor)).squeeze(), TOP_N_PREDICTIONS)
        _, finetuned_top_n = torch.topk(torch.sigmoid(finetuned_model(input_tensor)).squeeze(), TOP_N_PREDICTIONS)
        
        temp_indices = set(ground_indices) | set(base_model_top_n.cpu().numpy()) | set(finetuned_top_n.cpu().numpy())
        
        # extract a unique ID for the image (patientXXXXX/studyX/viewX_frontal)
        norm_path = img_path.replace("\\", "/")
        path_parts = norm_path.split("/")
        if len(path_parts) >= 3:
            folder_name = os.path.join(path_parts[-3], path_parts[-2], os.path.splitext(path_parts[-1])[0])
            file_title = os.path.splitext(path_parts[-1])[0].replace("view","") # e.g., "1_frontal"
        else:
            folder_name = os.path.splitext(path_parts[-1])[0]
            file_title = os.path.splitext(path_parts[-1])[0]
        output_dir = os.path.join(HEATMAP_OUTPUT, folder_name)
        os.makedirs(output_dir, exist_ok=True)

        # generate and save one comparison plot per relevent class
        for class_idx in temp_indices:
            class_name = CLASSIFICATIONS[class_idx]
            
            save_path = os.path.join(output_dir, f"{class_name}_comparison.png")
            if os.path.exists(save_path): continue
            try:
                # generate specific heatmaps for the current class_idx
                baseline_heatmap_graph = generate_heatmap_overlay(
                    baseline_model, baseline_target_layer, input_tensor.clone(), rgb_img, class_idx
                )
                finetuned_heatmap_graph = generate_heatmap_overlay(
                    finetuned_model, finetuned_target_layer, input_tensor.clone(), rgb_img, class_idx
                )
                
                plot_and_save_comparison(
                    p_img, ground_labels,
                    baseline_heatmap_graph, baseline_top_preds_formatted,
                    finetuned_heatmap_graph, finetuned_top_preds_formatted,
                    class_name, file_title, save_path
                )
                
            except Exception as e:
                print(f"failed to generate {class_name} for {img_path}: {e}")
        
        #delete tensors and clear cache after each image
        del input_tensor, rgb_img, p_img
        gc.collect()
        torch.cuda.empty_cache()

    print(f"check results in: {HEATMAP_OUTPUT}")
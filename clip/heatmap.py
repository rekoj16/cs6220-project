import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from PIL import Image
from transformers import CLIPModel, CLIPProcessor
import pandas as pd

CHEXPERT_LABELS = [
    'No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly',
    'Lung Opacity', 'Lung Lesion', 'Edema', 'Consolidation',
    'Pneumonia', 'Atelectasis', 'Pneumothorax', 'Pleural Effusion',
    'Pleural Other', 'Fracture', 'Support Devices'
]

class ViTGradCAM:
    def __init__(self, model):
        self.model = model
        self.gradients = None
        self.activations = None

        target_block = model.vision_model.encoder.layers[-1].self_attn.out_proj
        target_block.register_forward_hook(self._save_activation)
        target_block.register_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate(self, image_tensor, text_tensor):
        image_tensor = image_tensor.unsqueeze(0).cuda()
        image_tensor.requires_grad = True

        with torch.enable_grad():
            outputs = self.model(pixel_values=image_tensor, input_ids=text_tensor)
            logits = outputs.logits_per_image

            score = logits[0, 0]

            self.model.zero_grad()
            score.backward(retain_graph=True)

        gradients = self.gradients
        activations = self.activations

        weights = gradients.mean(dim=1, keepdim=True)
        cam = torch.relu(torch.matmul(activations, weights.transpose(-1, -2)))
        cam = cam.squeeze(-1).squeeze(0)

        cam = cam[1:]
        h = w = int(cam.shape[0] ** 0.5)
        cam = cam.reshape(h, w)

        cam -= cam.min()
        cam /= cam.max() + 1e-6
        return cam.detach().cpu()

def overlay_heatmap(cam, image_tensor):
    img = image_tensor.permute(1, 2, 0).cpu().numpy()
    img = (img - img.min()) / (img.max() + 1e-6)

    cam_resized = np.array(
        Image.fromarray(cam.cpu().numpy()).resize((img.shape[1], img.shape[0]), Image.BILINEAR)
    )

    cam_color = cm.jet(cam_resized)[..., :3]
    heatmap = 0.4 * cam_color + 0.6 * img
    return np.clip(heatmap, 0, 1)

def load_image(path, processor):
    img = Image.open(path).convert("RGB")
    pixel_values = processor(images=img, return_tensors="pt")["pixel_values"][0]
    return pixel_values, img

def get_ground_truth_labels(row):
    gt_labels = []
    for label in CHEXPERT_LABELS:
        if label in row and row[label] == 1.0:
            gt_labels.append(label)
    return gt_labels if gt_labels else ["No positive findings"]

def get_top_predictions(logits, top_k=3, threshold=None):
    sorted_indices = torch.argsort(logits, descending=True)
    predictions = []
    
    for i in range(min(top_k, len(sorted_indices))):
        idx = sorted_indices[i]
        score = float(logits[idx])
        if threshold is None or score > threshold:
            predictions.append(f"{CHEXPERT_LABELS[idx]} ({score:.3f})")
    
    return predictions if predictions else ["None"]


def main():
    baseline = CLIPModel.from_pretrained("openai/clip-vit-base-patch16").cuda()
    finetuned_path = "/home/hice1/rma96/scratch/clip_lora_chexpert"
    finetuned = CLIPModel.from_pretrained(finetuned_path).cuda()

    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")

    val_csv = "/home/hice1/rma96/scratch/CheXpert-v1.0-small/valid.csv"
    val_df = pd.read_csv(val_csv)

    data_root = "/home/hice1/rma96/scratch"

    output_dir = "heatmaps"
    os.makedirs(output_dir, exist_ok=True)

    text_inputs = processor(
        text=CHEXPERT_LABELS,
        return_tensors="pt",
        padding=True
    ).input_ids.cuda()

    baseline_cam = ViTGradCAM(baseline)
    finetuned_cam = ViTGradCAM(finetuned)

    for idx, row in val_df.iterrows():
        img_rel_path = row["Path"]

        if img_rel_path.startswith(data_root):
            img_rel_path = img_rel_path[len(data_root)+1:]

        img_abs_path = os.path.join(data_root, img_rel_path)

        if not os.path.exists(img_abs_path):
            print(f"Missing: {img_abs_path}")
            continue

        print(f"\n[{idx+1}/{len(val_df)}] Processing {img_abs_path}")

        image_tensor, orig_img = load_image(img_abs_path, processor)
        image_tensor = image_tensor.cuda()

        gt_labels = get_ground_truth_labels(row)

        with torch.no_grad():
            base_logits = baseline(pixel_values=image_tensor.unsqueeze(0),
                                   input_ids=text_inputs).logits_per_image[0]
            fine_logits = finetuned(pixel_values=image_tensor.unsqueeze(0),
                                    input_ids=text_inputs).logits_per_image[0]

        base_preds = get_top_predictions(base_logits, top_k=3)
        fine_preds = get_top_predictions(fine_logits, top_k=3)

        clean_name = img_rel_path.replace("/", "_").replace("\\", "_")
        save_path = os.path.join(output_dir, f"{clean_name}_comparison.png")

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        orig_array = np.array(orig_img)
        axes[0].imshow(orig_array)
        gt_text = "Ground Truth:\n" + "\n".join(f"• {label}" for label in gt_labels)
        axes[0].set_title(gt_text, fontsize=10, ha='center')
        axes[0].axis("off")

        top_base_idx = torch.argmax(base_logits)
        text_tensor_base = text_inputs[top_base_idx].unsqueeze(0)
        cam_base = baseline_cam.generate(image_tensor, text_tensor_base)
        heat_base = overlay_heatmap(cam_base, image_tensor)
        axes[1].imshow(heat_base)
        base_text = f"Baseline Model\n\nTop Predictions:\n" + "\n".join(f"• {p}" for p in base_preds)
        axes[1].set_title(base_text, fontsize=10, ha='center')
        axes[1].axis("off")

        top_fine_idx = torch.argmax(fine_logits)
        text_tensor_fine = text_inputs[top_fine_idx].unsqueeze(0)
        cam_fine = finetuned_cam.generate(image_tensor, text_tensor_fine)
        heat_fine = overlay_heatmap(cam_fine, image_tensor)
        axes[2].imshow(heat_fine)
        fine_text = f"Finetuned Model\n\nTop Predictions:\n" + "\n".join(f"• {p}" for p in fine_preds)
        axes[2].set_title(fine_text, fontsize=10, ha='center')
        axes[2].axis("off")

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Saved {save_path}")
        print(f"Ground Truth: {gt_labels}")
        print(f"Baseline Top-3: {base_preds}")
        print(f"Finetuned Top-3: {fine_preds}")


if __name__ == "__main__":
    main()
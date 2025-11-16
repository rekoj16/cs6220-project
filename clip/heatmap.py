import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms.functional import to_pil_image
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from pathlib import Path
from transformers import CLIPModel, CLIPProcessor

CHEXPERT_LABELS = [
    'No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly',
    'Lung Opacity', 'Lung Lesion', 'Edema', 'Consolidation',
    'Pneumonia', 'Atelectasis', 'Pneumothorax', 'Pleural Effusion',
    'Pleural Other', 'Fracture', 'Support Devices'
]

# -----------------------------
# GradCAM for CLIP Vision Encoder
# -----------------------------
class ViTGradCAM:
    def __init__(self, model):
        self.model = model
        self.gradients = None
        self.activations = None

        # ---- FIX: hook onto out_proj, which always exists ----
        target_block = model.vision_model.encoder.layers[-1].self_attn.out_proj

        target_block.register_forward_hook(self._save_activation)
        target_block.register_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        # activation after attention projection
        self.activations = output       # [B, N, C]

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0] # [B, N, C]

    def generate(self, image_tensor, text_tensor):
        """Runs forward + backward + GradCAM for a single image"""
        image_tensor = image_tensor.unsqueeze(0)
        outputs = self.model(pixel_values=image_tensor, input_ids=text_tensor)
        logits = outputs.logits_per_image  # similarity score

        score = logits[0, 0]
        score.backward()

        gradients = self.gradients          # [1, N, C]
        activations = self.activations      # [1, N, C]

        weights = gradients.mean(dim=1, keepdim=True)   # [1,1,C]
        cam = torch.relu(torch.matmul(activations, weights.transpose(-1, -2)))  # [1, N,1]
        cam = cam.squeeze(-1).squeeze(0)  # [N]

        # remove CLS token
        cam = cam[1:]

        h = w = int(cam.shape[0] ** 0.5)
        cam = cam.reshape(h, w)

        cam -= cam.min()
        cam /= cam.max() + 1e-6
        return cam.detach().cpu()


# -----------------------------
# Heatmap Overlay
# -----------------------------
def overlay_heatmap(cam, image_tensor):
    img = image_tensor.permute(1, 2, 0).cpu().numpy()
    img = (img - img.min()) / (img.max())

    cam = np.stack([cam, cam, cam], axis=-1)
    heat = 0.4 * cam + 0.6 * img
    heat = np.clip(heat, 0, 1)
    return heat


# -----------------------------
# Load image
# -----------------------------
def load_image(path, processor):
    img = Image.open(path).convert("RGB")
    pixel_values = processor(images=img, return_tensors="pt")["pixel_values"][0]
    return pixel_values, img


# -----------------------------
# Main
# -----------------------------
def main():

    # === Load models ===
    baseline = CLIPModel.from_pretrained("openai/clip-vit-base-patch16").cuda()
    finetuned = CLIPModel.from_pretrained("my_lora_checkpoint").cuda()

    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")

    # Path to test images
    img_path = "example_input.png"
    image_tensor, orig_img = load_image(img_path, processor)
    image_tensor = image_tensor.cuda()

    # Convert text prompts once
    text_inputs = processor(
        text=CHEXPERT_LABELS,
        return_tensors="pt",
        padding=True
    ).input_ids.cuda()

    # 14x2 grid
    fig, axes = plt.subplots(14, 2, figsize=(10, 40))

    baseline_cam = ViTGradCAM(baseline)
    finetuned_cam = ViTGradCAM(finetuned)

    for i, pathology in enumerate(CHEXPERT_LABELS):
        print(f"[{i+1}/14] {pathology}")

        # Select correct text token for this pathology
        text_tensor = text_inputs[i].unsqueeze(0)

        # --- Baseline CAM ---
        cam_base = baseline_cam.generate(image_tensor, text_tensor)
        heat_base = overlay_heatmap(cam_base, image_tensor.cpu()[0])

        # --- Finetuned CAM ---
        cam_fine = finetuned_cam.generate(image_tensor, text_tensor)
        heat_fine = overlay_heatmap(cam_fine, image_tensor.cpu()[0])

        axes[i, 0].imshow(heat_base)
        axes[i, 0].set_title(f"Baseline – {pathology}")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(heat_fine)
        axes[i, 1].set_title(f"Finetuned – {pathology}")
        axes[i, 1].axis("off")

    plt.tight_layout()
    plt.savefig("heatmap_grid_14x2.png", dpi=300)
    print("Saved heatmap_grid_14x2.png")


if __name__ == "__main__":
    main()

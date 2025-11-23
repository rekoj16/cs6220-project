import os
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from typing import Tuple, List
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

def compute_gradcam(model: torch.nn.Module, img: torch.Tensor, class_idx: int, device: str) -> torch.Tensor:
    """Compute normalized Grad-CAM using pytorch-grad-cam library.

    Parameters
    ----------
    model: the pre-trained or finetuned model to analyze.
    img: Tensor shape (1, 1, H, W) representing a single X-ray image.
    class_idx: Index of pathology/class to visualize.
    device: 'cuda' or 'cpu'. 

    Returns
    -------
    cam_up : torch.Tensor
        Upsampled heatmap (H, W) in [0,1].
    raw_cam : torch.Tensor
        Raw spatial heatmap before upsampling, in [0,1].
    """
    model.eval()
    img = img.to(device)
    
    target_layers = []
    if hasattr(model, 'features'):
        # For DenseNet, target the last layer in features
        target_layers = [model.features[-1]]
    elif hasattr(model, 'base_model') and hasattr(model.base_model, 'features'):
        # For LoRA wrapped models
        target_layers = [model.base_model.features[-1]]
    else:
        raise AttributeError("Cannot find 'features' attribute for Grad-CAM generation.")
    
    cam = GradCAM(model=model, target_layers=target_layers)
    # Define target class
    targets = [ClassifierOutputTarget(class_idx)]
    
    try:
        grayscale_cam = cam(input_tensor=img, targets=targets)
        cam_result = torch.from_numpy(grayscale_cam[0])  # Shape: (H, W)

        # The library already returns normalized [0,1] values and upsampled to input size
        cam_up = cam_result
        raw_cam = cam_result
        
        if hasattr(cam, 'activations_and_grads'):
            cam.activations_and_grads.release()
        
        return cam_up, raw_cam

    except Exception as e:
        print(f"Grad-CAM computation failed: {e}")
        if hasattr(cam, 'activations_and_grads'):
            cam.activations_and_grads.release()
        # Fallback: return zero heatmap
        h, w = img.shape[-2:]
        zero_cam = torch.zeros((h, w))
        return zero_cam, zero_cam


def _prepare_original(img: torch.Tensor) -> np.ndarray:
    """Convert single-channel tensor to 3-channel numpy for plotting."""
    # img expected shape (1,1,H,W) or (1,H,W)
    if img.ndim == 4:
        img = img.squeeze(0)
    if img.ndim == 3:  # (1,H,W)
        img = img.squeeze(0)
    arr = img.detach().cpu().numpy()
    arr = arr - arr.min()
    if arr.max() > 0:
        arr = arr / arr.max()
    rgb = np.repeat(arr[None, ...], 3, axis=0)  # (3,H,W)
    rgb = np.transpose(rgb, (1, 2, 0))  # (H,W,3)
    return rgb


def visualize_sample(
    baseline_model: torch.nn.Module,
    finetuned_model: torch.nn.Module,
    img: torch.Tensor,
    device: str,
    sample_id: str,
    output_dir: str,
    ground_truth: torch.Tensor, 
    dataset_pathologies: List[str],
    patient_id: str = None,
    positive_label_values: Tuple[float, ...] = (1.0,),
    top_k: int = 5,
) -> str:
    """Generate and save composite visualization (single figure with 3 panels)."""
    os.makedirs(output_dir, exist_ok=True)
    baseline_model.eval(); finetuned_model.eval()

    with torch.no_grad():
        base_out = torch.sigmoid(baseline_model(img.to(device)))  # (1, num_labels_baseline)
        ft_out = torch.sigmoid(finetuned_model(img.to(device)))   # (1, num_labels_finetuned)

    base_probs = base_out.squeeze(0)
    ft_probs = ft_out.squeeze(0)
    base_idx = int(torch.argmax(base_probs).item())
    ft_idx = int(torch.argmax(ft_probs).item())

    def get_name_list(model) -> List[str]:
        if hasattr(model, 'pathologies'):
            return list(model.pathologies)
        # fallback generic names
        return [f'class_{i}' for i in range(base_probs.shape[0])]

    base_names = get_name_list(baseline_model)
    ft_names = get_name_list(finetuned_model)

    def get_name(model, idx, names_list):
        if idx < len(names_list):
            return names_list[idx]
        return f'class_{idx}'

    base_name = get_name(baseline_model, base_idx, base_names)
    ft_name = get_name(finetuned_model, ft_idx, ft_names)

    # Compute Grad-CAMs for top predicted class only
    base_cam_up, _ = compute_gradcam(baseline_model, img, base_idx, device)
    ft_cam_up, _ = compute_gradcam(finetuned_model, img, ft_idx, device)

    # Prepare original
    orig = _prepare_original(img)

    # Ground truth positives
    gt_pos_indices = [i for i, v in enumerate(ground_truth.detach().cpu().tolist()) if v in positive_label_values]
    gt_pos_names = [dataset_pathologies[i] for i in gt_pos_indices] if gt_pos_indices else []
    gt_title_part = ("GT: \n" + "\n".join(gt_pos_names)) if gt_pos_names else "GT: (none positive)"

    # Build top-k prediction strings (vertical layout)
    def topk_str(probs: torch.Tensor, names: List[str], k: int) -> str:
        k = min(k, probs.shape[0])
        vals, idxs = torch.topk(probs, k)
        # Format each predicted label with its probability for clearer readability (vertical)
        parts = [f"{names[int(i)]}(p={float(v):.2f})" for v, i in zip(vals, idxs)]
        return "\n".join(parts)

    base_topk = topk_str(base_probs, base_names, top_k)
    ft_topk = topk_str(ft_probs, ft_names, top_k)

    # Figure 1x3 layout
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Panel 1: Original
    axes[0].imshow(orig, cmap='gray')
    axes[0].set_title(f'Original\n{gt_title_part}', fontsize=10)
    axes[0].axis('off')

    # Helper overlay
    def overlay(ax, orig_img, cam_up, main_title, sub_title):
        ax.imshow(orig_img, cmap='gray')
        ax.imshow(cam_up.detach().cpu().numpy(), cmap='jet', alpha=0.4)
        ax.set_title(f'{main_title}\n{sub_title}', fontsize=9)
        ax.axis('off')

    overlay(axes[1], orig, base_cam_up, 'Baseline', f'Top {top_k}:\n{base_topk}')
    overlay(axes[2], orig, ft_cam_up, 'Finetuned', f'Top {top_k}:\n{ft_topk}')

    fig.suptitle(f'Sample {sample_id} Comparison', fontsize=12)
    fig.tight_layout()
    
    # Generate filename with patient ID if available
    if patient_id:
        filename = f'sample_{sample_id}_{patient_id}_comparison.png'
    else:
        filename = f'sample_{sample_id}_comparison.png'
    
    out_path = os.path.join(output_dir, filename)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path

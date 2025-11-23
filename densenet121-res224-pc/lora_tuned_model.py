"""
This module provides utilities to reconstruct a LoRA fine-tuned DenseNet model from a checkpoint.
Then it can be used to load the model and run inference or evaluation on the CheXpert dataset.
"""
from __future__ import annotations

import os
from typing import List, Optional, Sequence, Tuple, Dict
import torch
import torch.nn as nn
import torchxrayvision as xrv
from peft import LoraConfig, get_peft_model

CHEXPERT_LABELS: List[str] = [
    'Atelectasis','Cardiomegaly','Consolidation','Edema','Enlarged Cardiomediastinum',
    'Fracture','Lung Lesion','Lung Opacity','Effusion','Pleural Other','Pneumonia',
    'Pneumothorax','Support Devices'
]

DEFAULT_LORA_CONFIG: Dict[str, object] = {
    'r': 16,
    'lora_alpha': 32,
    'lora_dropout': 0.1,
    # target_modules will be filled dynamically based on backbone
}


def _build_base_model(pretrained_weights: str = "densenet121-res224-pc") -> nn.Module:
    """Load base DenseNet (pretrained) from torchxrayvision, swap classifier to 13 CheXpert labels.

    Copies matching weights/bias from original 18‑class head when labels overlap.
    """
    base = xrv.models.DenseNet(weights=pretrained_weights)
    orig_head = base.classifier
    in_features = orig_head.in_features
    new_head = nn.Linear(in_features, len(CHEXPERT_LABELS))

    orig_labels = list(base.pathologies)
    copied = 0
    with torch.no_grad():
        for new_i, lab in enumerate(CHEXPERT_LABELS):
            if lab in orig_labels:
                old_i = orig_labels.index(lab)
                new_head.weight[new_i] = orig_head.weight[old_i]
                new_head.bias[new_i] = orig_head.bias[old_i]
                copied += 1
    base.classifier = new_head
    base.pathologies = CHEXPERT_LABELS  # ensure downstream code sees 13 labels
    base.op_threshs = None  # disable thresholds to avoid shape mismatches
    return base


def _select_lora_target_modules(model: nn.Module) -> List[str]:
    """Replicate the training selection logic for Conv2d layers under 'features'.

    Strategy: gather all Conv2d under features; prefer grouping conv2 (3x3) + conv1 (1x1).
    Fall back to last 8 conv layers if selection becomes too small.
    """
    conv_leaf_names: List[str] = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d) and name.startswith('features'):
            conv_leaf_names.append(name)
    if not conv_leaf_names:
        return []
    conv1_layers = [n for n in conv_leaf_names if n.endswith('conv1')]
    conv2_layers = [n for n in conv_leaf_names if n.endswith('conv2')]
    selected = conv2_layers + conv1_layers
    if len(selected) < 4:
        selected = conv_leaf_names[-8:]
    return selected


def _apply_lora(model: nn.Module, lora_config: Optional[Dict[str, object]] = None) -> nn.Module:
    """Apply LoRA to the selected Conv2d target modules and return wrapped model."""
    cfg = dict(DEFAULT_LORA_CONFIG)
    if lora_config:
        cfg.update({k: v for k, v in lora_config.items() if k in DEFAULT_LORA_CONFIG or k == 'target_modules'})
    target_modules = _select_lora_target_modules(model)
    cfg['target_modules'] = target_modules
    if not target_modules:
        print("[LoRA Loader] Warning: No Conv2d layers found for LoRA; returning unmodified model.")
        return model
    peft_cfg = LoraConfig(**cfg)
    wrapped = get_peft_model(model, peft_cfg)
    print(f"[LoRA Loader] Applied LoRA to {len(target_modules)} Conv2d layers:")
    for name in target_modules:
        print(f"  - {name}")
    return wrapped


def _load_checkpoint_state(checkpoint_path: str) -> Dict[str, torch.Tensor]:
    """Load a checkpoint which may be: pure state_dict, dict with 'state_dict', or full model object."""
    obj = torch.load(checkpoint_path, map_location='cpu')
    if isinstance(obj, dict) and 'state_dict' in obj:
        return obj['state_dict']
    if isinstance(obj, dict):
        return obj
    return obj.state_dict()


def _adapt_state_dict_keys(state_dict: Dict[str, torch.Tensor], model: nn.Module) -> Dict[str, torch.Tensor]:
    """Attempt to adapt checkpoint keys to current model naming.

    Fine-tuning used a custom wrapper class (`LoRAModel`) which introduced an
    extra leading `base_model.` prefix before the PEFT model's own `base_model.*`.
    That produces keys like:
        base_model.base_model.model.features.conv0.weight

    Our reconstruction returns the bare PEFT model, whose keys look like:
        base_model.model.features.conv0.weight

    This function rewrites keys with the double prefix to the single prefix
    when the rewritten key exists in the current model state dict.
    """
    model_keys = set(model.state_dict().keys())
    adapted: Dict[str, torch.Tensor] = {}
    rewrites = 0
    for k, v in state_dict.items():
        if k not in model_keys and k.startswith('base_model.base_model.'):
            new_k = k.replace('base_model.base_model.', 'base_model.', 1)
            if new_k in model_keys:
                adapted[new_k] = v
                rewrites += 1
                continue
        adapted[k] = v
    if rewrites:
        print(f"[LoRA Loader] Adapted {rewrites} keys (removed duplicate 'base_model.' prefix).")
    return adapted


def reconstruct_lora_model(checkpoint_path: str,
                           device: Optional[torch.device | str] = None,
                           pretrained_weights: str = "densenet121-res224-pc",
                           lora_config: Optional[Dict[str, object]] = None,
                           strict: bool = False) -> nn.Module:
    """Reconstruct LoRA fine‑tuned DenseNet model and load weights.

    Parameters
    ----------
    checkpoint_path : str
        Path to the saved state_dict from fine‑tuning.
    device : torch.device | str | None
        Device to move the model to; defaults to CUDA if available else CPU.
    pretrained_weights : str
        Base pretrained weights name used during fine‑tuning.
    lora_config : dict | None
        Optional overrides for DEFAULT_LORA_CONFIG (e.g., {'r': 8}).
    strict : bool
        Whether to enforce strict key matching when loading.

    Returns
    -------
    nn.Module
        Ready-to-infer model.
    """
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    elif isinstance(device, str):
        device = torch.device(device)

    print("[LoRA Loader] Building base model and classifier head...")
    base = _build_base_model(pretrained_weights=pretrained_weights)
    print("[LoRA Loader] Applying LoRA adapters...")
    model = _apply_lora(base, lora_config=lora_config)

    print(f"[LoRA Loader] Loading checkpoint: {checkpoint_path}")
    raw_state_dict = _load_checkpoint_state(checkpoint_path)
    state_dict = _adapt_state_dict_keys(raw_state_dict, model)
    missing, unexpected = model.load_state_dict(state_dict, strict=strict)
    print("[LoRA Loader] Checkpoint load summary:")
    print(f"  Missing keys   : {len(missing)}")
    print(f"  Unexpected keys: {len(unexpected)}")
    if missing:
        print("  (INFO) Missing (often harmless: classifier or LoRA re-init diffs):")
        for k in missing:
            print(f"    - {k}")
    if unexpected:
        print("  (INFO) Unexpected (ignored):")
        for k in unexpected:
            print(f"    - {k}")

    model.to(device)
    model.eval()
    total_model_keys = len(model.state_dict())
    loaded_keys = total_model_keys - len(missing)
    coverage = 100.0 * loaded_keys / total_model_keys if total_model_keys else 0.0
    print(f"[LoRA Loader] Parameter coverage: {loaded_keys}/{total_model_keys} ({coverage:.2f}%)")
    print("[LoRA Loader] Model ready on", device)
    return model


def load_lora_finetuned_model(checkpoint_path: str,
                              device: Optional[torch.device | str] = None) -> nn.Module:
    return reconstruct_lora_model(checkpoint_path=checkpoint_path, device=device, strict=False)


def run_dummy_forward(model: nn.Module, batch_size: int = 2) -> Tuple[torch.Size, torch.Size]:
    """Optional sanity check forward pass with random tensor (not X-ray normalized)."""
    with torch.no_grad():
        dummy = torch.randn(batch_size, 1, 224, 224, device=next(model.parameters()).device)
        out = model(dummy)
    print(f"[LoRA Loader] Dummy forward output shape: {out.shape}")
    return dummy.shape, out.shape


if __name__ == "__main__":
    # Testing model reconstrcutions
    ckpt = "/home/hice1/ymai8/scratch/cs6220-project/densenet121-res224-pc/result(4)_batch_32_epoch_10_lr_1e-4_rank_16_alpha_32_dropout_0.1_all-conv-layers/best_lora_model.pth"
    if os.path.isfile(ckpt):
        m = load_lora_finetuned_model(ckpt)
        run_dummy_forward(m)
    else:
        print("No checkpoint found at example path; please supply a valid path.")

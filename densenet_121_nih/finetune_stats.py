import torch
import torch.nn as nn
import torchxrayvision as xrv
import numpy as np
import json

# ============================================================
# CONFIG
# ============================================================
FINETUNED_MODEL_PATH = "results_full_finetune_8e/models/densenet121nih_chexpert14_finetuned.pt"

CHEXPERT_CLASSES = [
    "No Finding",
    "Enlarged Cardiomediastinum",
    "Cardiomegaly",
    "Lung Opacity",
    "Lung Lesion",
    "Edema",
    "Consolidation",
    "Pneumonia",
    "Atelectasis",
    "Pneumothorax",
    "Pleural Effusion",
    "Pleural Other",
    "Fracture",
    "Support Devices",
]

NIH_CLASSES = [
    "Atelectasis",
    "Cardiomegaly",
    "Effusion",
    "Infiltration",
    "Mass",
    "Nodule",
    "Pneumonia",
    "Pneumothorax",
]

# ============================================================
# HELPERS
# ============================================================
def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable, (trainable / total) * 100.0


# ============================================================
# LOAD BASELINE NIH MODEL
# ============================================================
def load_baseline():
    print("\n================ BASELINE NIH DENSENET121 ================")
    m = xrv.models.DenseNet(weights="densenet121-res224-nih")
    m.op_norm = None
    m.op_threshs = None

    total, trainable, pct = count_parameters(m)

    print(f"Model Name: densenet121-res224-nih")
    print(f"Classifier Head: {m.classifier}")
    print(f"Labels Trained On (8): {NIH_CLASSES}")
    print(f"Total Parameters: {total:,}")
    print(f"Trainable Parameters: {trainable:,}")
    print(f"Trainable %: {pct:.2f}%")

    return m, total, trainable, pct


# ============================================================
# LOAD FINETUNED MODEL
# ============================================================
def load_finetuned(path):
    print("\n================ FINETUNED CHEXPERT14 MODEL ================")
    ckpt = torch.load(path, map_location="cpu")

    m = xrv.models.DenseNet(weights="densenet121-res224-nih")
    m.op_norm = None
    m.op_threshs = None

    num_out = len(ckpt["classes"])
    m.classifier = nn.Linear(m.classifier.in_features, num_out)

    m.load_state_dict(ckpt["state_dict"])

    total, trainable, pct = count_parameters(m)

    print(f"Model Name: {ckpt.get('model_name', 'finetuned_model')}")
    print(f"Classifier Head: {m.classifier}")
    print(f"Labels Trained On ({num_out}): {ckpt['classes']}")
    print(f"Total Parameters: {total:,}")
    print(f"Trainable Parameters: {trainable:,}")
    print(f"Trainable %: {pct:.2f}%")

    return m, total, trainable, pct, ckpt


# ============================================================
# MAIN
# ============================================================
def main():
    base_m, base_total, base_train, base_pct = load_baseline()
    fin_m, fin_total, fin_train, fin_pct, ckpt = load_finetuned(FINETUNED_MODEL_PATH)

    print("\n================ PARAMETER COMPARISON SUMMARY ================")
    print(f"{'Metric':30s} | {'Baseline NIH':15s} | {'Finetuned 14-label':15s}")
    print("-" * 75)
    print(f"{'Total Params':30s} | {base_total:,} | {fin_total:,}")
    print(f"{'Trainable Params':30s} | {base_train:,} | {fin_train:,}")
    print(f"{'Trainable %':30s} | {base_pct:.2f}% | {fin_pct:.2f}%")
    print(f"{'Output Classes':30s} | {len(NIH_CLASSES)} | {len(ckpt['classes'])}")
    print(f"{'Classifier Head':30s} | {str(base_m.classifier)} | {str(fin_m.classifier)}")

    print("\nDone.\n")


if __name__ == "__main__":
    main()

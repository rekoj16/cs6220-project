import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import torchxrayvision as xrv
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from PIL import Image
import torchvision.transforms as T
import matplotlib.pyplot as plt

# ================================================
# CONFIGURATION
# ================================================
DATA_DIR = "CheXpert-v1.0-small"
TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
VAL_CSV = os.path.join(DATA_DIR, "valid.csv")

RESULTS_DIR = "results_full_finetune1"
MODEL_DIR = os.path.join(RESULTS_DIR, "models")
BASELINE_DIR = os.path.join(RESULTS_DIR, "baseline_nih8_overlap5")
FINETUNED_DIR = os.path.join(RESULTS_DIR, "finetuned_14")
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")

for d in [RESULTS_DIR, MODEL_DIR, BASELINE_DIR, FINETUNED_DIR, PLOTS_DIR]:
    os.makedirs(d, exist_ok=True)

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

NIH_PATHOLOGIES = [
    "Atelectasis",
    "Cardiomegaly",
    "Effusion",
    "Infiltration",
    "Mass",
    "Nodule",
    "Pneumonia",
    "Pneumothorax",
]

OVERLAP_PAIRS = [
    ("Atelectasis", "Atelectasis"),
    ("Cardiomegaly", "Cardiomegaly"),
    ("Effusion", "Pleural Effusion"),
    ("Pneumonia", "Pneumonia"),
    ("Pneumothorax", "Pneumothorax"),
]
OVERLAP_CHEX_LABELS = [dst for _, dst in OVERLAP_PAIRS]

IMG_SIZE = 224
BATCH_SIZE = 64
EPOCHS = 3
LR = 1e-4
THRESHOLD = 0.6


# ================================================
# DATASET
# ================================================
class CheXpertDataset(torch.utils.data.Dataset):
    def __init__(self, df, root_dir):
        self.df = df.reset_index(drop=True)
        self.root_dir = root_dir
        self.transform = T.Compose([
            T.Resize((IMG_SIZE, IMG_SIZE)),
            T.ToTensor(),
            T.Normalize([0.5], [0.25]),
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        rel_path = row["Path"]
        if rel_path.startswith(self.root_dir + "/"):
            rel_path = rel_path[len(self.root_dir) + 1:]
        img_path = os.path.join(self.root_dir, rel_path)
        img = Image.open(img_path).convert("L")
        img = self.transform(img)
        labels = torch.tensor(row[CHEXPERT_CLASSES].values.astype(np.float32))
        return img, labels


# ================================================
# LOAD SPLITS
# ================================================
def load_chexpert_splits():
    train_df = pd.read_csv(TRAIN_CSV)
    val_df = pd.read_csv(VAL_CSV)
    for df in [train_df, val_df]:
        df[CHEXPERT_CLASSES] = df[CHEXPERT_CLASSES].fillna(0)
        df[CHEXPERT_CLASSES] = df[CHEXPERT_CLASSES].replace(-1, 0)
        df[CHEXPERT_CLASSES] = df[CHEXPERT_CLASSES].astype(np.float32)
    return train_df, val_df


# ================================================
# EVALUATION HELPERS
# ================================================
def evaluate_baseline_nih8_overlap5(model, loader, device, chex_idx, nih_idx):
    model.eval()
    all_probs, all_true = [], []
    with torch.no_grad():
        for imgs, labs in tqdm(loader, desc="Baseline NIH8→CheXpert5 Eval"):
            imgs = imgs.to(device)
            labs = labs.to(device)
            out = model(imgs)
            probs = torch.sigmoid(out)
            all_probs.append(probs[:, nih_idx].cpu().numpy())
            all_true.append(labs[:, chex_idx].cpu().numpy())
    return np.vstack(all_true), np.vstack(all_probs)


def evaluate_14(model, loader, device):
    model.eval()
    all_probs, all_true = [], []
    with torch.no_grad():
        for imgs, labs in tqdm(loader, desc="Finetuned 14-label Eval"):
            imgs = imgs.to(device)
            labs = labs.to(device)
            probs = torch.sigmoid(model(imgs)).cpu().numpy()
            all_probs.append(probs)
            all_true.append(labs.cpu().numpy())
    return np.vstack(all_true), np.vstack(all_probs)


def compute_per_class_metrics_from_labels(class_names, y_true, y_prob, threshold):
    metrics = {}
    for i, cls in enumerate(class_names):
        t = y_true[:, i]
        p = y_prob[:, i]
        preds = (p > threshold).astype(int)
        tp = int(((preds == 1) & (t == 1)).sum())
        fp = int(((preds == 1) & (t == 0)).sum())
        tn = int(((preds == 0) & (t == 0)).sum())
        fn = int(((preds == 0) & (t == 1)).sum())
        prec = tp / (tp + fp + 1e-8)
        rec = tp / (tp + fn + 1e-8)
        f1 = 2 * prec * rec / (prec + rec + 1e-8)
        try:
            auc = roc_auc_score(t, p) if len(np.unique(t)) > 1 else None
        except:
            auc = None
        metrics[cls] = {
            "precision": float(prec),
            "recall": float(rec),
            "f1": float(f1),
            "auc": float(auc) if auc is not None else None,
            "tp": tp, "fp": fp, "tn": tn, "fn": fn
        }
    return metrics


def summarize_macro_metrics(metrics, class_names):
    f1s = [metrics[c]["f1"] for c in class_names]
    aucs = [metrics[c]["auc"] for c in class_names if metrics[c]["auc"] is not None]
    return float(np.mean(f1s)), float(np.mean(aucs)) if aucs else 0.0


# ================================================
# PLOTTING HELPERS
# ================================================
def plot_auroc_bar(class_names, metrics, path, title):
    x = np.arange(len(class_names))
    auc_vals = [(metrics[c]["auc"] or 0) for c in class_names]
    plt.figure(figsize=(18, 6))
    plt.bar(x, auc_vals)
    plt.xticks(x, class_names, rotation=45, ha="right")
    plt.ylabel("AUC")
    plt.title(title)
    plt.ylim(0, 1)
    plt.axhline(0.5, color="red", linestyle="--", linewidth=1, label="Random (AUC=0.5)")
    plt.legend()
    for i, v in enumerate(auc_vals):
        plt.text(i, v + 0.02, f"{v:.3f}", ha="center", va="bottom", fontsize=8, rotation=90)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_f1_bar(class_names, metrics, path, title):
    x = np.arange(len(class_names))
    f1_vals = [metrics[c]["f1"] for c in class_names]
    plt.figure(figsize=(18, 6))
    plt.bar(x, f1_vals)
    plt.xticks(x, class_names, rotation=45, ha="right")
    plt.ylabel("F1")
    plt.title(title)
    plt.ylim(0, 1)
    for i, v in enumerate(f1_vals):
        plt.text(i, v + 0.02, f"{v:.3f}", ha="center", va="bottom", fontsize=8, rotation=90)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_macro_summary(macro_f1, macro_auc, path, stage_name):
    names = [f"{stage_name} F1", f"{stage_name} AUC"]
    vals = [macro_f1, macro_auc]
    plt.figure(figsize=(6, 6))
    plt.bar(names, vals)
    plt.ylim(0, 1)
    for i, v in enumerate(vals):
        plt.text(i, v + 0.02, f"{v:.3f}", ha="center", va="bottom", fontsize=10)
    plt.title(f"Macro Metrics Summary: {stage_name}")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


# ================================================
# TRAINING
# ================================================
def finetune_full_14(model, train_loader, val_loader, device):
    train_losses = []
    val_losses = []

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    for ep in range(EPOCHS):
        model.train()
        total = 0.0
        for imgs, labs in tqdm(train_loader, desc=f"Finetune Epoch {ep+1}/{EPOCHS}"):
            imgs, labs = imgs.to(device), labs.to(device)
            out = model(imgs)
            loss = criterion(out, labs)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item()
        train_loss = total / len(train_loader)
        train_losses.append(train_loss)
        print(f"[Finetune] Epoch {ep+1} TRAIN loss: {train_loss:.4f}")

        model.eval()
        val_total = 0.0
        with torch.no_grad():
            for imgs, labs in val_loader:
                imgs, labs = imgs.to(device), labs.to(device)
                val_out = model(imgs)
                val_loss = criterion(val_out, labs)
                val_total += val_loss.item()
        val_loss_epoch = val_total / len(val_loader)
        val_losses.append(val_loss_epoch)
        print(f"[Finetune] Epoch {ep+1} VAL loss:   {val_loss_epoch:.4f}")

    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label="Training Loss", marker='o')
    plt.plot(val_losses, label="Validation Loss", marker='o')
    plt.title("Training vs Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "loss_curve.png"))
    plt.close()

    return train_losses, val_losses


# ================================================
# SAVE MODEL
# ================================================
def save_finetuned_model(path, model):
    ckpt = {
        "model_name": "DenseNet121-res224-nih-chexpert14-finetuned",
        "classes": CHEXPERT_CLASSES,
        "img_size": IMG_SIZE,
        "state_dict": model.state_dict()
    }
    torch.save(ckpt, path)


# ================================================
# MAIN
# ================================================
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_df, val_df = load_chexpert_splits()
    train_ds = CheXpertDataset(train_df, DATA_DIR)
    val_ds = CheXpertDataset(val_df, DATA_DIR)

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True
    )

    nih_model = xrv.models.DenseNet(weights="densenet121-res224-nih")
    nih_model.op_norm = None
    nih_model.op_threshs = None
    nih_model.to(device)

    chex_idx = [CHEXPERT_CLASSES.index(dst) for _, dst in OVERLAP_PAIRS]
    nih_idx = [NIH_PATHOLOGIES.index(src) for src, _ in OVERLAP_PAIRS]

    y_true_b5, y_prob_b5 = evaluate_baseline_nih8_overlap5(
        nih_model, val_loader, device, chex_idx, nih_idx
    )

    metrics_b5 = compute_per_class_metrics_from_labels(
        OVERLAP_CHEX_LABELS, y_true_b5, y_prob_b5, THRESHOLD
    )
    macro_f1_b5, macro_auc_b5 = summarize_macro_metrics(metrics_b5, OVERLAP_CHEX_LABELS)

    with open(os.path.join(BASELINE_DIR, "metrics_baseline_overlap5.json"), "w") as f:
        json.dump({
            "threshold": THRESHOLD,
            "macro_f1_overlap5": macro_f1_b5,
            "macro_auc_overlap5": macro_auc_b5,
            "per_class_overlap5": metrics_b5
        }, f, indent=2)

    plot_auroc_bar(
        OVERLAP_CHEX_LABELS,
        metrics_b5,
        os.path.join(PLOTS_DIR, "baseline_nih8_overlap5_auroc.png"),
        "Baseline NIH→CheXpert (Overlap 5) AUC"
    )
    plot_f1_bar(
        OVERLAP_CHEX_LABELS,
        metrics_b5,
        os.path.join(PLOTS_DIR, "baseline_nih8_overlap5_f1.png"),
        "Baseline NIH→CheXpert (Overlap 5) F1"
    )
    plot_macro_summary(
        macro_f1_b5,
        macro_auc_b5,
        os.path.join(PLOTS_DIR, "baseline_nih8_overlap5_macro.png"),
        "Baseline NIH8→CheXpert5"
    )

    model_14 = xrv.models.DenseNet(weights="densenet121-res224-nih")
    model_14.op_norm = None
    model_14.op_threshs = None
    model_14.classifier = nn.Linear(model_14.classifier.in_features, len(CHEXPERT_CLASSES))
    model_14.to(device)

    finetune_full_14(model_14, train_loader, val_loader, device)

    y_true_14, y_prob_14 = evaluate_14(model_14, val_loader, device)
    metrics_14 = compute_per_class_metrics_from_labels(
        CHEXPERT_CLASSES, y_true_14, y_prob_14, THRESHOLD
    )
    macro_f1_14, macro_auc_14 = summarize_macro_metrics(metrics_14, CHEXPERT_CLASSES)

    with open(os.path.join(FINETUNED_DIR, "metrics_finetuned_14.json"), "w") as f:
        json.dump({
            "threshold": THRESHOLD,
            "macro_f1_14": macro_f1_14,
            "macro_auc_14": macro_auc_14,
            "per_class_14": metrics_14
        }, f, indent=2)

    plot_auroc_bar(
        CHEXPERT_CLASSES,
        metrics_14,
        os.path.join(PLOTS_DIR, "finetuned_14_auroc.png"),
        "Finetuned DenseNet NIH→CheXpert14 AUC"
    )
    plot_f1_bar(
        CHEXPERT_CLASSES,
        metrics_14,
        os.path.join(PLOTS_DIR, "finetuned_14_f1.png"),
        "Finetuned DenseNet NIH→CheXpert14 F1"
    )
    plot_macro_summary(
        macro_f1_14,
        macro_auc_14,
        os.path.join(PLOTS_DIR, "finetuned_14_macro.png"),
        "Finetuned NIH→CheXpert14"
    )

    save_finetuned_model(
        os.path.join(MODEL_DIR, "densenet121nih_chexpert14_finetuned.pt"),
        model_14
    )


if __name__ == "__main__":
    main()

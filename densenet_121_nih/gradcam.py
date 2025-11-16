import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchxrayvision as xrv
import pandas as pd
import numpy as np
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as T

# ============================================================
# CONFIG
# ============================================================
CHEXPERT_ROOT = "CheXpert-v1.0-small"
VAL_CSV = os.path.join(CHEXPERT_ROOT, "valid.csv")

FINETUNED_MODEL_PATH = "results_full_finetune1/models/densenet121nih_chexpert14_finetuned.pt"

SAVE_DIR = "results_full_finetune_gradcam1"
os.makedirs(SAVE_DIR, exist_ok=True)

IMG_SIZE = 224
THRESHOLD = 0.6
NUM_IMAGES_PER_CLASS = 10
HEATMAP_ALPHA = 0.45

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

OVERLAP_PAIRS = [
    ("Atelectasis", "Atelectasis"),
    ("Cardiomegaly", "Cardiomegaly"),
    ("Effusion", "Pleural Effusion"),
    ("Pneumonia", "Pneumonia"),
    ("Pneumothorax", "Pneumothorax"),
]

device = "cuda" if torch.cuda.is_available() else "cpu"

# ============================================================
# DATASET
# ============================================================
class CheXpertDataset(torch.utils.data.Dataset):
    def __init__(self, df, root):
        self.df = df.reset_index(drop=True)
        self.root = root
        self.transform = T.Compose([
            T.Resize((IMG_SIZE, IMG_SIZE)),
            T.ToTensor(),
            T.Normalize([0.5], [0.25]),
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        rel = row["Path"]

        if rel.startswith("CheXpert-v1.0-small/"):
            rel = rel[len("CheXpert-v1.0-small/"):]

        img_path = os.path.join(self.root, rel)

        img = Image.open(img_path).convert("L")
        img_t = self.transform(img)
        labels = row[CHEXPERT_CLASSES].values.astype(np.float32)
        return img_t, labels, img_path


# ============================================================
# MODEL LOADERS
# ============================================================
def load_baseline(device):
    m = xrv.models.DenseNet(weights="densenet121-res224-nih")
    m.op_norm = None
    m.op_threshs = None
    return m.to(device)


def load_finetuned(path, device):
    ckpt = torch.load(path, map_location=device)
    m = xrv.models.DenseNet(weights="densenet121-res224-nih")
    m.op_norm = None
    m.op_threshs = None
    m.classifier = nn.Linear(m.classifier.in_features, len(ckpt["classes"]))
    m.load_state_dict(ckpt["state_dict"])
    return m.to(device), ckpt["classes"]


# ============================================================
# EVALUATION
# ============================================================
def evaluate_models(baseline, finetuned, loader, device):
    baseline.eval()
    finetuned.eval()

    all_true = []
    all_prob_b = []
    all_prob_f = []
    all_paths = []

    with torch.no_grad():
        for imgs, labs, paths in tqdm(loader, desc="Eval"):
            imgs = imgs.to(device)

            out_b = baseline(imgs)
            out_f = finetuned(imgs)

            prob_b = torch.sigmoid(out_b).cpu().numpy()
            prob_f = torch.sigmoid(out_f).cpu().numpy()

            all_true.append(labs.numpy())
            all_prob_b.append(prob_b)
            all_prob_f.append(prob_f)
            all_paths.extend(paths)

    y_true = np.vstack(all_true)
    y_prob_b = np.vstack(all_prob_b)
    y_prob_f = np.vstack(all_prob_f)

    return y_true, y_prob_b, y_prob_f, all_paths


# ============================================================
# SAMPLING BASED ON FINETUNED MODEL
# ============================================================
def pick_balanced_indices(y_true, y_prob_f, chex_idx, n=10):
    t = y_true[:, chex_idx].astype(int)
    p = y_prob_f[:, chex_idx]
    pred = (p >= THRESHOLD).astype(int)

    TN = np.where((t == 0) & (pred == 0))[0]
    TP = np.where((t == 1) & (pred == 1))[0]
    FP = np.where((t == 0) & (pred == 1))[0]
    FN = np.where((t == 1) & (pred == 0))[0]

    groups = {"TN": TN, "TP": TP, "FP": FP, "FN": FN}
    per = max(1, n // 4)
    selected = []

    for _, idxs in groups.items():
        if len(idxs) > 0:
            k = min(per, len(idxs))
            chosen = np.random.choice(idxs, k, replace=False)
            selected.extend(chosen.tolist())

    if len(selected) < n:
        remaining = np.setdiff1d(np.arange(len(t)), np.array(selected))
        if len(remaining) > 0:
            extra = np.random.choice(remaining, min(n - len(selected), len(remaining)), replace=False)
            selected.extend(extra.tolist())

    return selected[:n]


# ============================================================
# GRAD-CAM
# ============================================================
def get_last_conv(model):
    return model.features.denseblock4.denselayer16.conv2


def compute_gradcam(model, img_t, target_idx, conv_layer, device, img_size=IMG_SIZE):
    acts = {}
    grads = {}

    def fwd_hook(m, i, o):
        acts["value"] = o

    def bwd_hook(m, gi, go):
        grads["value"] = go[0]

    h1 = conv_layer.register_forward_hook(fwd_hook)
    h2 = conv_layer.register_backward_hook(bwd_hook)

    img = img_t.unsqueeze(0).to(device)
    model.zero_grad()

    out = model(img)
    prob = torch.sigmoid(out)[0, target_idx]
    prob.backward()

    A = acts["value"]
    G = grads["value"]

    weights = G.mean(dim=(2, 3), keepdim=True)

    cam = (weights * A).sum(dim=1, keepdim=True)
    cam = F.relu(cam)

    cam_min = cam.min()
    cam_max = cam.max()
    if (cam_max - cam_min) > 1e-8:
        cam = (cam - cam_min) / (cam_max - cam_min)
    else:
        cam = torch.zeros_like(cam)

    cam_raw = cam[0, 0].detach().cpu().numpy()

    cam_up = F.interpolate(
        cam,
        size=(img_size, img_size),
        mode="bilinear",
        align_corners=False,
    )[0, 0].detach().cpu().numpy()

    h1.remove()
    h2.remove()

    return cam_up, cam_raw, float(prob.item())


# ============================================================
# PANEL PLOT
# ============================================================
def plot_panel(
    orig,
    cam_up_b, cam_raw_b, p_b, outcome_b,
    cam_up_f, cam_raw_f, p_f, outcome_f,
    class_name,
    save_path,
):
    fig, ax = plt.subplots(2, 2, figsize=(10, 10))

    ax[0, 0].imshow(orig, cmap="gray")
    ax[0, 0].imshow(cam_up_b, cmap="jet", alpha=HEATMAP_ALPHA)
    ax[0, 0].set_title(f"Baseline Overlay\np={p_b:.6f} ({outcome_b})")
    ax[0, 0].axis("off")

    ax[0, 1].imshow(cam_raw_b, cmap="viridis", interpolation="nearest")
    ax[0, 1].set_title("Baseline Raw 7×7")
    ax[0, 1].axis("off")

    ax[1, 0].imshow(orig, cmap="gray")
    ax[1, 0].imshow(cam_up_f, cmap="jet", alpha=HEATMAP_ALPHA)
    ax[1, 0].set_title(f"Finetuned Overlay\np={p_f:.6f} ({outcome_f})")
    ax[1, 0].axis("off")

    ax[1, 1].imshow(cam_raw_f, cmap="viridis", interpolation="nearest")
    ax[1, 1].set_title("Finetuned Raw 7×7")
    ax[1, 1].axis("off")

    plt.suptitle(f"Grad-CAM Comparison: {class_name}", fontsize=16)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


# ============================================================
# MAIN
# ============================================================
def main():
    df = pd.read_csv(VAL_CSV)
    df = df.fillna(0.0)

    dataset = CheXpertDataset(df, CHEXPERT_ROOT)
    loader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=False)

    baseline = load_baseline(device)
    finetuned, fin_classes = load_finetuned(FINETUNED_MODEL_PATH, device)

    assert list(fin_classes) == CHEXPERT_CLASSES, "Finetuned ckpt classes != CHEXPERT_CLASSES"

    conv_b = get_last_conv(baseline)
    conv_f = get_last_conv(finetuned)

    y_true, y_prob_b, y_prob_f, paths = evaluate_models(baseline, finetuned, loader, device)

    for nih_label, chex_label in OVERLAP_PAIRS:
        nih_idx = NIH_CLASSES.index(nih_label)
        chex_idx = CHEXPERT_CLASSES.index(chex_label)

        out_dir = os.path.join(SAVE_DIR, chex_label.replace(" ", "_"))
        os.makedirs(out_dir, exist_ok=True)

        idxs = pick_balanced_indices(y_true, y_prob_f, chex_idx, n=NUM_IMAGES_PER_CLASS)

        for k, idx in enumerate(idxs):
            img_t, labels, img_path = dataset[idx]
            orig = Image.open(img_path).convert("L")
            orig = orig.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
            orig_np = np.array(orig)

            t = int(labels[chex_idx])

            p_b = torch.sigmoid(torch.tensor(y_prob_b[idx, nih_idx])).item()
            pred_b = p_b >= THRESHOLD
            outcome_b = (
                "TP" if pred_b and t == 1 else
                "FP" if pred_b and t == 0 else
                "TN" if not pred_b and t == 0 else
                "FN"
            )

            p_f = float(y_prob_f[idx, chex_idx])
            pred_f = p_f >= THRESHOLD
            outcome_f = (
                "TP" if pred_f and t == 1 else
                "FP" if pred_f and t == 0 else
                "TN" if not pred_f and t == 0 else
                "FN"
            )

            cam_up_b, cam_raw_b, _ = compute_gradcam(baseline, img_t, nih_idx, conv_b, device)
            cam_up_f, cam_raw_f, _ = compute_gradcam(finetuned, img_t, chex_idx, conv_f, device)

            save_path = os.path.join(out_dir, f"{chex_label.replace(' ', '_')}_{k}.png")

            plot_panel(
                orig_np,
                cam_up_b, cam_raw_b, p_b, outcome_b,
                cam_up_f, cam_raw_f, p_f, outcome_f,
                chex_label,
                save_path,
            )

    print(f"Saved Grad-CAM panels to: {SAVE_DIR}")


if __name__ == "__main__":
    main()

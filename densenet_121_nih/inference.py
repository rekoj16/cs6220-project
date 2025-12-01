import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchxrayvision as xrv
import pandas as pd
import numpy as np
from PIL import Image
import torchvision.transforms as T

# ============================================================
# CONFIG
# ============================================================
CHEXPERT_ROOT = "CheXpert-v1.0-small"
CSV_PATH = os.path.join(CHEXPERT_ROOT, "valid.csv")

FINETUNED_MODEL_PATH = "results_full_finetune/models/densenet121nih_chexpert14_finetuned.pt"

PATIENT_ID_SUBSTR = "patient64601"

IMG_SIZE = 224
THRESHOLD = 0.6
BATCH_SIZE = 32

device = "cuda" if torch.cuda.is_available() else "cpu"

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
# MAIN
# ============================================================
def main():
    print("Loading CSV:", CSV_PATH)
    df = pd.read_csv(CSV_PATH)
    df = df.fillna(0.0)

    mask = df["Path"].astype(str).str.contains(PATIENT_ID_SUBSTR, case=False, na=False)
    df_patient = df[mask].copy()

    if df_patient.empty:
        print(f"No rows found for PATIENT_ID_SUBSTR = '{PATIENT_ID_SUBSTR}' in CSV.")
        return

    print(f"Found {len(df_patient)} rows for patient substring '{PATIENT_ID_SUBSTR}'.")

    dataset = CheXpertDataset(df_patient, CHEXPERT_ROOT)
    loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    print("Loading baseline NIH model")
    baseline = load_baseline(device)

    print("Loading finetuned model from:", FINETUNED_MODEL_PATH)
    finetuned, fin_classes = load_finetuned(FINETUNED_MODEL_PATH, device)

    if list(fin_classes) != CHEXPERT_CLASSES:
        print("Warning: ckpt classes != CHEXPERT_CLASSES")
        print("Ckpt classes:", fin_classes)

    baseline.eval()
    finetuned.eval()

    print("\n=== Inference results for patient substring:", PATIENT_ID_SUBSTR, "===\n")

    with torch.no_grad():
        for imgs, labs, paths in loader:
            imgs = imgs.to(device)
            labs_np = labs.numpy()

            out_b = baseline(imgs)
            out_f = finetuned(imgs)

            prob_b = torch.sigmoid(out_b).cpu().numpy()
            prob_f = torch.sigmoid(out_f).cpu().numpy()

            for i in range(len(paths)):
                path = paths[i]
                gt = labs_np[i]
                pb = prob_b[i]
                pf = prob_f[i]

                print(f"Image: {path}")

                gt_pos = [name for name, v in zip(CHEXPERT_CLASSES, gt) if v == 1.0]
                if gt_pos:
                    print("  Ground truth positives:", ", ".join(gt_pos))
                else:
                    print("  Ground truth positives: none")

                base_pos = []
                for name, v in zip(NIH_CLASSES, pb):
                    if v >= THRESHOLD:
                        base_pos.append(f"{name}={v:.3f}")
                if base_pos:
                    print(f"  Baseline NIH positives at threshold {THRESHOLD}:")
                    for s in base_pos:
                        print("   -", s)
                else:
                    print(f"  Baseline NIH positives at threshold {THRESHOLD}: none")

                fin_pos = []
                for name, v in zip(CHEXPERT_CLASSES, pf):
                    if v >= THRESHOLD:
                        fin_pos.append(f"{name}={v:.3f}")
                if fin_pos:
                    print(f"  Finetuned positives at threshold {THRESHOLD}:")
                    for s in fin_pos:
                        print("   -", s)
                else:
                    print(f"  Finetuned positives at threshold {THRESHOLD}: none")

                print()

if __name__ == "__main__":
    main()

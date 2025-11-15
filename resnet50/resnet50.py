import os, sys
import numpy as np
import torch
import torchvision
import torchxrayvision as xrv
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve
import torch.nn.functional as F

# --------------------------------------------------
# Grad-CAM for ResNet50
# --------------------------------------------------
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate(self, input_tensor, class_idx):
        self.model.zero_grad()
        output = self.model(input_tensor)
        if isinstance(output, (list, tuple)):
            output = output[0]
        score = output[:, class_idx]
        score.backward(retain_graph=True)

        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * self.activations, dim=1).squeeze()
        cam = F.relu(cam)
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        return cam.cpu().numpy()


# --------------------------------------------------
# Main Script
# --------------------------------------------------
def main():
    try:
        transform = torchvision.transforms.Compose([
            xrv.datasets.XRayCenterCrop(),
            xrv.datasets.XRayResizer(512)
        ])

        dataset_root = "/home/hice1/achen448/CS6220/cs6220-project/dataset/CheXpert-v1.0-small"

        d_chex = xrv.datasets.CheX_Dataset(
            imgpath=dataset_root,
            csvpath=os.path.join(dataset_root, "valid.csv"),
            views=["PA", "AP"],
            unique_patients=False,
            transform=transform
        )

        print("\nDataset loaded:")
        print(d_chex)
        print("-" * 60)

        # --------------------------------------------------
        # Load ResNet50
        # --------------------------------------------------
        model = xrv.models.ResNet(weights="resnet50-res512-all")

        # Normalize label names between model and CheXpert
        name_fix = {
            "Pleural_Thickening": "Pleural Other",
            "Effusion": "Pleural Effusion"
        }

        # Final 10 CheXpert-compatible labels
        VALID_LABELS = [
            "Atelectasis",
            "Consolidation",
            "Pneumothorax",
            "Edema",
            "Pneumonia",
            "Pleural Other",
            "Cardiomegaly",
            "Lung Lesion",
            "Lung Opacity",
            "Enlarged Cardiomediastinum"
        ]

        pathology_mapping = {}

        for m_idx, m_name in enumerate(model.pathologies):
            m_fixed = name_fix.get(m_name, m_name)
            if m_fixed not in VALID_LABELS:
                continue
            for d_idx, d_name in enumerate(d_chex.pathologies):
                if m_fixed.lower() == str(d_name).lower():
                    pathology_mapping[m_idx] = (d_idx, m_fixed)
                    break

        print("\n=== Final Mapped Pathologies (10) ===")
        for k,v in pathology_mapping.items():
            print(f"Model[{k}] -> Dataset[{v[0]}]: {v[1]}")
        print("-" * 60)

        # --------------------------------------------------
        # Device
        # --------------------------------------------------
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"

        model = model.to(device)
        model.eval()
        target_layer = model.model.layer4
        cam_generator = GradCAM(model, target_layer)

        # --------------------------------------------------
        # Inference
        # --------------------------------------------------
        dataloader = DataLoader(d_chex, batch_size=4, shuffle=False, num_workers=2)
        threshold = 0.625

        cached_preds, cached_gt = [], []
        false_positives = []

        print("\nRunning inference...")
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                if batch_idx % 10 == 0:
                    print(f"  Batch {batch_idx}/{len(dataloader)}")

                imgs = batch["img"].to(device)
                gt = batch["lab"].to(device)
                outputs = torch.sigmoid(model(imgs))

                cached_preds.append(outputs.cpu())
                cached_gt.append(gt.cpu())

                for i in range(imgs.shape[0]):
                    idx = batch_idx * dataloader.batch_size + i

                    if idx < len(d_chex.csv):
                        rel_path = d_chex.csv.iloc[idx]["Path"]
                        if rel_path.startswith("CheXpert-v1.0-small"):
                            rel_path = rel_path.replace("CheXpert-v1.0-small/", "")
                        path = os.path.join(dataset_root, rel_path)
                    else:
                        path = "Unknown"

                    for m_idx, (d_idx, name) in pathology_mapping.items():
                        gt_val = gt[i, d_idx].item()
                        pred_prob = outputs[i, m_idx].item()
                        if np.isnan(gt_val):
                            continue
                        if gt_val == 0 and pred_prob > threshold:
                            false_positives.append({
                                "image_path": path,
                                "pathology": name,
                                "prob": pred_prob,
                                "model_idx": m_idx
                            })

        print(f"\nTotal false positives found: {len(false_positives)}")

        # --------------------------------------------------
        # Metrics
        # --------------------------------------------------
        all_preds = torch.cat(cached_preds, dim=0).numpy()
        all_gt = torch.cat(cached_gt, dim=0).numpy()
        metrics = []

        for m_idx, (d_idx, name) in pathology_mapping.items():
            y_pred = all_preds[:, m_idx]
            y_true = all_gt[:, d_idx]
            valid = ~np.isnan(y_true)
            y_pred = y_pred[valid]
            y_true = y_true[valid]

            if len(np.unique(y_true)) < 2:
                continue

            y_bin = (y_pred > threshold).astype(int)
            auc = roc_auc_score(y_true, y_pred)
            prec = precision_score(y_true, y_bin, zero_division=0)
            rec = recall_score(y_true, y_bin, zero_division=0)
            f1 = f1_score(y_true, y_bin, zero_division=0)
            tn, fp, fn, tp = confusion_matrix(y_true, y_bin).ravel()
            fpr = fp / (fp + tn + 1e-8)

            metrics.append({
                "Pathology": name,
                "AUROC": auc,
                "Precision": prec,
                "Recall": rec,
                "F1": f1,
                "Support": len(y_true),
                "FalsePos": fp
            })

        df = pd.DataFrame(metrics).sort_values(by="AUROC", ascending=False)
        df.to_csv("metrics_resnet50_full.csv", index=False)
        print("\nSaved metrics → metrics_resnet50_full.csv")

        fp_df = pd.DataFrame(false_positives)
        fp_df.to_csv("false_positives_resnet50_full.csv", index=False)

        # --------------------------------------------------
        # PLOTTING
        # --------------------------------------------------
        print("\nGenerating graphs...")

        # 1. False Positive Distribution
        plt.figure(figsize=(10, 6))
        sns.countplot(
            y="pathology",
            data=fp_df,
            order=df["Pathology"]
        )
        plt.title("False Positive Distribution (ResNet50)")
        plt.tight_layout()
        plt.savefig("plot_fp_distribution.png")
        plt.close()

        # 2. AUROC
        plt.figure(figsize=(10, 6))
        sns.barplot(x="AUROC", y="Pathology", data=df)
        plt.title("AUROC per Pathology")
        plt.xlim(0, 1)
        plt.tight_layout()
        plt.savefig("plot_auroc.png")
        plt.close()

        # 3. F1 Score
        plt.figure(figsize=(10, 6))
        sns.barplot(x="F1", y="Pathology", data=df)
        plt.title("F1 Score per Pathology")
        plt.xlim(0, 1)
        plt.tight_layout()
        plt.savefig("plot_f1.png")
        plt.close()

        # 4. Precision & Recall
        melted = df.melt(id_vars="Pathology", value_vars=["Precision", "Recall"])
        plt.figure(figsize=(12, 6))
        sns.barplot(data=melted, x="value", y="Pathology", hue="variable")
        plt.title("Precision & Recall per Pathology")
        plt.xlim(0, 1)
        plt.tight_layout()
        plt.savefig("plot_precision_recall.png")
        plt.close()

        # 5. FP Rate vs AUROC
        plt.figure(figsize=(10, 6))
        plt.scatter(df["AUROC"], df["FalsePos"], s=120)
        for i, row in df.iterrows():
            plt.annotate(row["Pathology"], (row["AUROC"], row["FalsePos"]))
        plt.xlabel("AUROC")
        plt.ylabel("False Positives")
        plt.title("False Positives vs AUROC")
        plt.tight_layout()
        plt.savefig("plot_fp_vs_auroc.png")
        plt.close()

        # 6. Support counts
        plt.figure(figsize=(12, 6))
        sns.barplot(y="Pathology", x="Support", data=df)
        plt.title("Support (Samples per Pathology)")
        plt.tight_layout()
        plt.savefig("plot_support.png")
        plt.close()

        # 7. Priority Score = High FP + Low AUROC
        df["PriorityScore"] = (df["FalsePos"] / df["FalsePos"].max()) + (1 - df["AUROC"])
        df["PriorityScore"] /= df["PriorityScore"].max()

        plt.figure(figsize=(12, 6))
        sns.barplot(x="PriorityScore", y="Pathology", data=df)
        plt.title("Fine-tuning Priority Score (Higher = Needs Improvement)")
        plt.tight_layout()
        plt.savefig("plot_priority_score.png")
        plt.close()

        print("\n✓ All graphs generated successfully")

    except Exception as e:
        print("ERROR:", e)


if __name__ == "__main__":
    main()

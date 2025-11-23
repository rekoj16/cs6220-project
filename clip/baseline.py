import torch
import pandas as pd
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve
from transformers import CLIPProcessor, CLIPModel

CACHE_DIR = "/home/hice1/rma96/scratch/transformers_cache"
MODEL_NAME = "openai/clip-vit-base-patch32"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
NUM_WORKERS = 1

CHEXPERT_LABELS = [
    'No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly',
    'Lung Opacity', 'Lung Lesion', 'Edema', 'Consolidation',
    'Pneumonia', 'Atelectasis', 'Pneumothorax', 'Pleural Effusion',
    'Pleural Other', 'Fracture', 'Support Devices'
]

TEMPLATES = [
    "A chest X-ray showing {}.",
    "An X-ray image of {}.",
    "This radiograph demonstrates {}.",
]

class CheXpertDataset(torch.utils.data.Dataset):
    def __init__(self, csv_path, img_root):
        self.df = pd.read_csv(csv_path)
        self.img_root = Path(img_root)
        self.labels = self.df[CHEXPERT_LABELS].fillna(0).replace(-1, 0).values.astype(np.float32)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        rel_path = self.df.iloc[idx]["Path"].replace("CheXpert-v1.0-small/", "")
        img_path = self.img_root / rel_path

        image = Image.open(img_path).convert("RGB")
        labels = torch.tensor(self.labels[idx], dtype=torch.float32)
        return image, labels

def collate_fn(batch):
    images, labels = zip(*batch)
    labels = torch.stack(labels, dim=0)
    return list(images), labels

def prepare_text_embeddings(model, processor, device):
    """Create text prompts (templates) and compute text embeddings once."""
    text_prompts = []
    for label in CHEXPERT_LABELS:
        for t in TEMPLATES:
            text_prompts.append(t.format(label))

    with torch.no_grad():
        tokenized = processor(text=text_prompts, padding=True, return_tensors="pt")
        tokenized = {k: v.to(device) for k, v in tokenized.items()}
        if hasattr(model, "get_text_features"):
            text_emb = model.get_text_features(**tokenized)
        else:
            out = model.get_text_features(**tokenized)
            text_emb = out

        text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)
    return text_emb

def compute_zero_shot_predictions(model, processor, dataloader, device):
    """Compute zero-shot predictions across the dataloader.
       Returns (predictions, labels) where predictions shape=(N, num_labels)."""
    model.eval()
    text_emb = prepare_text_embeddings(model, processor, device)
    num_templates = len(TEMPLATES)
    num_texts = text_emb.shape[0]
    assert num_texts == len(CHEXPERT_LABELS) * num_templates

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(dataloader):
            image_inputs = processor(images=images, return_tensors="pt").to(device)
            if hasattr(model, "get_image_features"):
                image_emb = model.get_image_features(**{"pixel_values": image_inputs["pixel_values"]})
            else:
                out = model.get_image_features(**{"pixel_values": image_inputs["pixel_values"]})
                image_emb = out

            image_emb = image_emb / image_emb.norm(dim=-1, keepdim=True)

            similarity = image_emb @ text_emb.T

            logits_per_label = []
            for i in range(len(CHEXPERT_LABELS)):
                start = i * num_templates
                end = (i + 1) * num_templates

                group = similarity[:, start:end]
                score = group.mean(dim=1)
                logits_per_label.append(score)
            logits_per_label = torch.stack(logits_per_label, dim=1)

            mins = logits_per_label.min(dim=1, keepdim=True).values
            maxs = logits_per_label.max(dim=1, keepdim=True).values
            probs = (logits_per_label - mins) / (maxs - mins + 1e-8)

            all_preds.append(probs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    predictions = np.vstack(all_preds)
    labels = np.vstack(all_labels)
    return predictions, labels


def calculate_auroc(predictions, labels):
    aurocs = {}
    for i, name in enumerate(CHEXPERT_LABELS):
        if len(np.unique(labels[:, i])) > 1:
            try:
                aurocs[name] = roc_auc_score(labels[:, i], predictions[:, i])
            except Exception:
                aurocs[name] = np.nan
        else:
            aurocs[name] = np.nan
    valid = [v for v in aurocs.values() if not np.isnan(v)]
    mean_auroc = np.mean(valid) if valid else np.nan
    return aurocs, mean_auroc

def plot_auroc_bargraph(aurocs, save_path="auroc_barplot.png"):
    labels = list(aurocs.keys())
    values = [aurocs[k] for k in labels]

    clean = [(l, v) for l, v in zip(labels, values) if not np.isnan(v)]
    clean.sort(key=lambda x: x[1])

    sorted_labels = [x[0] for x in clean]
    sorted_values = [x[1] for x in clean]

    plt.figure(figsize=(14, 6))
    plt.bar(sorted_labels, sorted_values)
    plt.ylabel("AUROC")
    plt.xlabel("Pathology")
    plt.title("Zero-Shot CLIP AUROC per Pathology")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved AUROC bar plot → {save_path}")


def main():
    print(f"Using device: {DEVICE}")
    csv_path = "./CheXpert-v1.0-small/valid.csv"
    img_root = "./CheXpert-v1.0-small"

    dataset = CheXpertDataset(csv_path, img_root)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn
    )

    print(f"Dataset size: {len(dataset)}")

    model = CLIPModel.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR).to(DEVICE)
    processor = CLIPProcessor.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR)

    predictions, labels = compute_zero_shot_predictions(model, processor, dataloader, DEVICE)

    aurocs, mean_auroc = calculate_auroc(predictions, labels)
    print("=" * 50)
    print("AUROC Results:")
    print("=" * 50)
    for k, v in aurocs.items():
        print(f"{k:30s}: {v:.4f}" if not np.isnan(v) else f"{k:30s}: N/A")
    print("=" * 50)
    print(f"{'Mean AUROC':30s}: {mean_auroc:.4f}")
    print("=" * 50)

    plot_auroc_bargraph(aurocs)
    pd.DataFrame({"Pathology": list(aurocs.keys()), "AUROC": list(aurocs.values())}).to_csv(
        "clip_chexpert_zero_shot_results.csv", index=False
    )
    print("Results saved to clip_chexpert_zero_shot_results.csv")


if __name__ == "__main__":
    main()

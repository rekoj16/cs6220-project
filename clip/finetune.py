import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm
from peft import LoraConfig, get_peft_model
from transformers import CLIPModel, CLIPProcessor
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt

CACHE_DIR = "/home/hice1/rma96/scratch/transformers_cache"
MODEL_NAME = "openai/clip-vit-base-patch32"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32
LR = 1e-6
EPOCHS = 5

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
    labels = torch.stack(labels)
    return list(images), labels

def calculate_auroc(predictions, labels):
    aurocs = {}
    for i, name in enumerate(CHEXPERT_LABELS):
        if len(np.unique(labels[:, i])) > 1:
            try:
                aurocs[name] = roc_auc_score(labels[:, i], predictions[:, i])
            except:
                aurocs[name] = np.nan
        else:
            aurocs[name] = np.nan
    valid = [v for v in aurocs.values() if not np.isnan(v)]
    mean_auroc = np.mean(valid) if valid else np.nan
    return aurocs, mean_auroc


def prepare_text_embeddings(model, processor, device):
    text_prompts = []
    for label in CHEXPERT_LABELS:
        for t in TEMPLATES:
            text_prompts.append(t.format(label))
    with torch.no_grad():
        tok = processor(text=text_prompts, padding=True, return_tensors="pt").to(device)
        text_emb = model.get_text_features(**tok)
        text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)
    return text_emb


def train_lora():
    print("Loading model...")
    model = CLIPModel.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR)

    for p in model.parameters():
        p.requires_grad = False

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
        lora_dropout=0.1,
        bias="none",
    )

    model = get_peft_model(model, lora_config).to(DEVICE)
    processor = CLIPProcessor.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR)

    text_emb = prepare_text_embeddings(model, processor, DEVICE)
    num_templates = len(TEMPLATES)

    train_ds = CheXpertDataset("./CheXpert-v1.0-small/train.csv",
                               "./CheXpert-v1.0-small/")
    val_ds = CheXpertDataset("./CheXpert-v1.0-small/valid.csv",
                             "./CheXpert-v1.0-small/")

    train_dl = torch.utils.data.DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=4, collate_fn=collate_fn
    )
    val_dl = torch.utils.data.DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=4, collate_fn=collate_fn
    )

    optim = torch.optim.AdamW(model.parameters(), lr=LR)
    loss_fn = nn.BCEWithLogitsLoss()
    scale = nn.Parameter(torch.tensor(10.0)).to(DEVICE)

    train_loss_history = []
    val_loss_history = []
    epoch_aurocs = {}

    print("Starting training...")
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        all_preds = []
        all_labels = []

        for images, labels in tqdm(train_dl):
            labels = labels.to(DEVICE)
            img_tok = processor(images=images, return_tensors="pt").to(DEVICE)
            img_emb = model.get_image_features(**{"pixel_values": img_tok["pixel_values"]})
            img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)

            sim = scale * (img_emb @ text_emb.T)
            logits = []
            for i in range(len(CHEXPERT_LABELS)):
                start = i * num_templates
                end = (i + 1) * num_templates
                logits.append(sim[:, start:end].mean(dim=1))
            logits = torch.stack(logits, dim=1)

            loss = loss_fn(logits, labels)
            optim.zero_grad()
            loss.backward()
            optim.step()

            total_loss += loss.item()
            all_preds.append(logits.detach().cpu().numpy())
            all_labels.append(labels.detach().cpu().numpy())

        avg_train_loss = total_loss / len(train_dl)
        train_loss_history.append(avg_train_loss)

        model.eval()
        val_loss = 0
        val_preds = []
        val_labels = []

        with torch.no_grad():
            for images, labels in val_dl:
                labels = labels.to(DEVICE)
                img_tok = processor(images=images, return_tensors="pt").to(DEVICE)
                img_emb = model.get_image_features(**{"pixel_values": img_tok["pixel_values"]})
                img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)

                sim = scale * (img_emb @ text_emb.T)
                logits = []
                for i in range(len(CHEXPERT_LABELS)):
                    start = i * num_templates
                    end = (i + 1) * num_templates
                    logits.append(sim[:, start:end].mean(dim=1))
                logits = torch.stack(logits, dim=1)

                loss = loss_fn(logits, labels)
                val_loss += loss.item()

                val_preds.append(logits.cpu().numpy())
                val_labels.append(labels.cpu().numpy())

        avg_val_loss = val_loss / len(val_dl)
        val_loss_history.append(avg_val_loss)

        preds = np.vstack(val_preds)
        lbls = np.vstack(val_labels)
        aurocs, mean_auroc = calculate_auroc(preds, lbls)
        epoch_aurocs[epoch + 1] = aurocs

        print(f"Epoch {epoch+1}/{EPOCHS} - Train Loss: {avg_train_loss:.4f}  "
              f"Val Loss: {avg_val_loss:.4f}  Mean AUROC: {mean_auroc:.4f}")

    model.save_pretrained("clip_lora_chexpert")
    print("Saved LoRA fine-tuned model → clip_lora_chexpert/")

    plt.figure(figsize=(8,5))
    plt.plot(range(1, EPOCHS+1), train_loss_history, marker='o', label="Training Loss")
    plt.plot(range(1, EPOCHS+1), val_loss_history, marker='o', label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training & Validation Loss Curve")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("lora_loss_curve.png", dpi=300)
    plt.close()

    print("Saved loss curve → lora_loss_curve.png")

    # single bar graphs
    for e in [1, 3, 5]:
        if e not in epoch_aurocs:
            continue
        auroc_dict = epoch_aurocs[e]
        labels = list(auroc_dict.keys())
        values = [
            auroc_dict[l] if not np.isnan(auroc_dict[l]) else 0.0
            for l in labels
        ]

        plt.figure(figsize=(14,6))
        plt.bar(labels, values)
        plt.xticks(rotation=45, ha='right')
        plt.xlabel("Pathology")
        plt.ylabel("AUROC")
        plt.title(f"Validation AUROC per Pathology — Epoch {e}")
        plt.ylim(0, 1)
        plt.grid(axis='y', linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.savefig(f"lora_auroc_epoch_{e}.png", dpi=300)
        plt.close()

    # combined bar graphs
    for e in [1, 3, 5]:
        if e not in epoch_aurocs:
            print(f"Skipping AUROC grouped graph — epoch {e} missing")
            return

    auroc_e1 = epoch_aurocs[1]
    auroc_e3 = epoch_aurocs[3]
    auroc_e5 = epoch_aurocs[5]

    pathologies = CHEXPERT_LABELS

    values_e1 = [0 if np.isnan(auroc_e1[p]) else auroc_e1[p] for p in pathologies]
    values_e3 = [0 if np.isnan(auroc_e3[p]) else auroc_e3[p] for p in pathologies]
    values_e5 = [0 if np.isnan(auroc_e5[p]) else auroc_e5[p] for p in pathologies]

    x = np.arange(len(pathologies))
    width = 0.25

    plt.figure(figsize=(18, 6))

    plt.bar(x - width, values_e1, width, label="Epoch 1")
    plt.bar(x,         values_e3, width, label="Epoch 3")
    plt.bar(x + width, values_e5, width, label="Epoch 5")

    plt.xticks(x, pathologies, rotation=45, ha='right')
    plt.ylabel("AUROC")
    plt.title("AUROC per Pathology (Epochs 1, 3, 5)")
    plt.ylim(0, 1)
    plt.legend()
    plt.tight_layout()
    plt.savefig("lora_auroc_grouped.png", dpi=300)
    plt.close()
    print("Saved grouped AUROC plot → lora_auroc_grouped.png")


if __name__ == "__main__":
    train_lora()

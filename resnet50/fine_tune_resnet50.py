import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchxrayvision as xrv
from torch.utils.data import DataLoader
import numpy as np

# ----------------------------------------------------------
# 1. Load CheXpert Training Data
# ----------------------------------------------------------
dataset_root = "/home/hice1/achen448/CS6220/cs6220-project/dataset/CheXpert-v1.0-small"

transform = torchvision.transforms.Compose([
    xrv.datasets.XRayCenterCrop(),
    xrv.datasets.XRayResizer(512),
])

train_dataset = xrv.datasets.CheX_Dataset(
    imgpath=dataset_root,
    csvpath=os.path.join(dataset_root, "train.csv"),
    views=["PA", "AP"],
    unique_patients=False,
    transform=transform
)

print("Loaded training dataset:", train_dataset)

# ----------------------------------------------------------
# 2. Load pretrained ResNet50
# ----------------------------------------------------------
model = xrv.models.ResNet(weights="resnet50-res512-all")
original_labels = model.pathologies
print("Model labels:", original_labels)

# ----------------------------------------------------------
# 3. Map labels + exclude Fracture
# ----------------------------------------------------------
chexpert_labels = [str(x) for x in train_dataset.pathologies]
exclude_label = "fracture"

pathology_mapping = {}
for m_idx, m_name in enumerate(original_labels):
    for d_idx, d_name in enumerate(chexpert_labels):
        if m_name.lower() == d_name.lower():
            if m_name.lower() != exclude_label:
                pathology_mapping[m_idx] = d_idx

print("Final mapped pathologies:", pathology_mapping)

# ----------------------------------------------------------
# 4. Freeze everything except layer4 and final FC
# ----------------------------------------------------------
for param in model.parameters():
    param.requires_grad = False

# Unfreeze high-level conv block (layer4)
for param in model.model.layer4.parameters():
    param.requires_grad = True

# Unfreeze classifier
for param in model.model.fc.parameters():      # FIXED
    param.requires_grad = True

print("\nTrainable parameters:")
for name, param in model.named_parameters():
    if param.requires_grad:
        print("  ", name)

# ----------------------------------------------------------
# 5. Training setup
# ----------------------------------------------------------
device = "cuda" if torch.cuda.is_available() else \
         "mps" if torch.backends.mps.is_available() else "cpu"

model = model.to(device)

dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=0)

criterion = nn.BCEWithLogitsLoss(reduction='none')

optimizer = optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=1e-4
)

epochs = 3

# ----------------------------------------------------------
# 6. Training loop
# ----------------------------------------------------------
print("\nStarting fine-tuning...\n")

for epoch in range(epochs):
    model.train()
    running_loss = 0

    for batch in dataloader:
        imgs = batch["img"].to(device)
        labels = batch["lab"].to(device)

        outputs = model(imgs)

        # Extract mapped pathologies only
        preds = []
        trues = []
        for m_idx, d_idx in pathology_mapping.items():
            preds.append(outputs[:, m_idx])
            trues.append(labels[:, d_idx])

        preds = torch.stack(preds, dim=1)
        trues = torch.stack(trues, dim=1)

        mask = ~torch.isnan(trues)
        raw_loss = criterion(preds, trues)
        loss = raw_loss[mask].mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch {epoch+1}/{epochs} — Loss: {running_loss/len(dataloader):.4f}")

# ----------------------------------------------------------
# 7. Save model
# ----------------------------------------------------------
save_path = "resnet50_finetuned_chexpert.pth"
torch.save(model.state_dict(), save_path)
print("\n✓ Fine-tuning complete! Saved to:", save_path)

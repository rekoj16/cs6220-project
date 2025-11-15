import os, argparse, csv
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from transformers import AutoModel, AutoTokenizer, AdamW
from tqdm import tqdm

CHEXPERT_LABELS = [
    "No Finding","Enlarged Cardiomediastinum","Cardiomegaly","Lung Opacity","Lung Lesion",
    "Edema","Consolidation","Pneumonia","Atelectasis","Pneumothorax",
    "Pleural Effusion","Pleural Other","Fracture","Support Devices"
]

def map_uncertainty(val, strategy="ignore"):
    if val in ("-1","U",-1):
        if strategy=="ignore": return np.nan
        return 1.0 if strategy=="one" else 0.0
    try: return float(val)
    except: return np.nan

class CheXpertDataset(Dataset):
    def __init__(self, csv_path, img_root, labels=CHEXPERT_LABELS, transform=None, uncertainty_strategy="ignore"):
        self.rows=[]
        with open(csv_path,'r',newline='') as f:
            reader = csv.DictReader(f)
            for r in reader:
                path = r.get('Path') or r.get('path') or r.get('ImagePath')
                if not path: continue
                labs=[]
                for lbl in labels:
                    val = None
                    for key in (lbl, lbl.lower(), lbl.replace(" ","_"), lbl.replace(" ","")):
                        if key in r:
                            val = r[key]; break
                    if val is None: val = r.get(lbl,"0")
                    labs.append(map_uncertainty(val, strategy=uncertainty_strategy))
                self.rows.append({"path": path, "report": r.get('Report') or r.get('Findings') or "", "labels": np.array(labs,dtype=float)})
        self.img_root = img_root
        self.transform = transform

    def __len__(self): return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        img_path = os.path.join(self.img_root, row['path'])
        img = Image.open(img_path).convert("RGB")
        if self.transform: img = self.transform(img)
        return img, row['labels']

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_csv", required=True)
    parser.add_argument("--val_csv", required=True)
    parser.add_argument("--img_root", required=True)
    parser.add_argument("--model_name", default="microsoft/BiomedVLP-BioViL-T")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--uncertainty_strategy", choices=["ignore","one","zero"], default="ignore")
    parser.add_argument("--out_dir", default="finetune_out")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.485,0.485],[0.229,0.229,0.229])
    ])

    train_ds = CheXpertDataset(args.train_csv, args.img_root, transform=transform, uncertainty_strategy=args.uncertainty_strategy)
    val_ds   = CheXpertDataset(args.val_csv, args.img_root, transform=transform, uncertainty_strategy=args.uncertainty_strategy)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)

    print("Loading model:", args.model_name)
    model = AutoModel.from_pretrained(args.model_name, trust_remote_code=True).to(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)

    optimizer = AdamW(model.parameters(), lr=args.lr)
    criterion = torch.nn.BCEWithLogitsLoss()

    for epoch in range(args.epochs):
        model.train()
        pbar = tqdm(train_loader, desc=f"Train Epoch {epoch+1}")
        for imgs, labels in pbar:
            imgs = imgs.to(args.device)
            labels = labels.to(args.device).float()
            optimizer.zero_grad()
            outputs = model.get_image_embeddings(pixel_values=imgs)
            outputs = torch.sigmoid(outputs)  # convert to 0-1
            # For simplicity: use dot product with label embeddings as logits
            # You could implement a full multimodal classifier here
            text_emb = model.get_projected_text_embeddings(
                **tokenizer([f"There is {lbl.lower()}." for lbl in CHEXPERT_LABELS], 
                            padding=True, truncation=True, return_tensors="pt").to(args.device)
            )
            text_emb = torch.nn.functional.normalize(text_emb, dim=1)
            logits = outputs @ text_emb.T
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            pbar.set_postfix(loss=loss.item())

        # validation
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for imgs, labels in tqdm(val_loader, desc="Validation"):
                imgs = imgs.to(args.device)
                labels = labels.to(args.device).float()
                outputs = model.get_image_embeddings(pixel_values=imgs)
                outputs = torch.nn.functional.normalize(outputs, dim=1)
                text_emb = model.get_projected_text_embeddings(
                    **tokenizer([f"There is {lbl.lower()}." for lbl in CHEXPERT_LABELS], 
                                padding=True, truncation=True, return_tensors="pt").to(args.device)
                )
                text_emb = torch.nn.functional.normalize(text_emb, dim=1)
                logits = outputs @ text_emb.T
                all_preds.append(logits.cpu().numpy())
                all_labels.append(labels.cpu().numpy())
        y_pred = np.vstack(all_preds)
        y_true = np.vstack(all_labels)
        aurocs = []
        for i in range(len(CHEXPERT_LABELS)):
            mask = ~np.isnan(y_true[:,i])
            if mask.sum() < 10: aurocs.append(np.nan); continue
            aurocs.append(roc_auc_score(y_true[mask,i], y_pred[mask,i]))
        print(f"Epoch {epoch+1} AUROCs: {aurocs}")

    torch.save(model.state_dict(), os.path.join(args.out_dir,"finetuned_model.pt"))
    print("Finetuning done.")

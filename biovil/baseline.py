import os, argparse, csv
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from transformers import AutoModel, AutoTokenizer
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve
import matplotlib.pyplot as plt
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
        return img, row['report'], row['labels']

def compute_metrics(y_true, y_score):
    L = y_true.shape[1]
    aurocs=[]
    auprcs=[]
    for i in range(L):
        mask = ~np.isnan(y_true[:,i])
        if mask.sum() < 10:
            aurocs.append(np.nan); auprcs.append(np.nan)
            continue
        try:
            au = roc_auc_score(y_true[mask,i], y_score[mask,i])
            ap = average_precision_score(y_true[mask,i], y_score[mask,i])
        except Exception:
            au=np.nan; ap=np.nan
        aurocs.append(au); auprcs.append(ap)
    return np.array(aurocs), np.array(auprcs)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--chexpert_csv", required=True)
    parser.add_argument("--img_root", required=True)
    parser.add_argument("--model_name", default="microsoft/BiomedVLP-BioViL-T")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--uncertainty_strategy", choices=["ignore","one","zero"], default="ignore")
    parser.add_argument("--out_dir", default="baseline_out")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.485,0.485],[0.229,0.229,0.229])
    ])

    ds = CheXpertDataset(args.chexpert_csv, args.img_root, transform=transform, uncertainty_strategy=args.uncertainty_strategy)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=4)

    print("Loading model:", args.model_name)
    vlp = AutoModel.from_pretrained(args.model_name, trust_remote_code=True).to(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)

    # create text prompts
    label_prompts = [f"There is {lbl.lower()}." for lbl in CHEXPERT_LABELS]
    vlp.eval()
    with torch.no_grad():
        tok = tokenizer(label_prompts, padding=True, truncation=True, return_tensors="pt").to(args.device)
        text_emb = vlp.get_projected_text_embeddings(input_ids=tok['input_ids'], attention_mask=tok['attention_mask'])
        text_emb = torch.nn.functional.normalize(text_emb, dim=1)

    all_scores, all_labels = [], []

    for imgs, reports, labels in tqdm(loader, desc="Zero-shot eval"):
        imgs = imgs.to(args.device)
        with torch.no_grad():
            img_emb = vlp.get_image_embeddings(pixel_values=imgs)
            img_emb = torch.nn.functional.normalize(img_emb, dim=1)
            sims = (img_emb @ text_emb.T).cpu().numpy()
        all_scores.append(sims)
        all_labels.append(labels.numpy())

    y_score = np.vstack(all_scores)
    y_true = np.vstack(all_labels)

    np.save(os.path.join(args.out_dir,"preds_val.npy"), y_score)
    np.save(os.path.join(args.out_dir,"labels_val.npy"), y_true)

    print("Saved predictions and labels.")
    aurocs, auprcs = compute_metrics(y_true, y_score)
    for lbl, a, p in zip(CHEXPERT_LABELS, aurocs, auprcs):
        print(f"{lbl}: AUROC={a}  AUPRC={p}")

    # plot ROC & PR curves
    for i,lbl in enumerate(CHEXPERT_LABELS):
        mask = ~np.isnan(y_true[:,i])
        if mask.sum() < 10: continue
        fpr, tpr, _ = roc_curve(y_true[mask,i], y_score[mask,i])
        prec, rec, _ = precision_recall_curve(y_true[mask,i], y_score[mask,i])
        plt.figure()
        plt.plot(fpr, tpr); plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title(f"ROC - {lbl}")
        plt.savefig(os.path.join(args.out_dir, f"roc_{i}_{lbl.replace(' ','_')}.png")); plt.close()
        plt.figure()
        plt.plot(rec, prec); plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title(f"PR - {lbl}")
        plt.savefig(os.path.join(args.out_dir, f"pr_{i}_{lbl.replace(' ','_')}.png")); plt.close()

    print("Baseline evaluation done.")

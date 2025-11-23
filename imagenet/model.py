import torch
import torch.nn as nn
import torchvision.models as models
from torchmetrics.classification import MultilabelAUROC
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# import from the chexpert_loader.py
try:
    from chexpert_loader import val_loader, NUM_CLASSES, CLASS_NAMES
    print("Imported 'val_loader' and 'NUM_CLASSES' from chexpert_loader.py")
    if NUM_CLASSES != 12 or len(CLASS_NAMES) != 12:
        print(f"Warning: Expected 12 classes, but found {NUM_CLASSES} and {len(CLASS_NAMES)} labels.")
    print(f"Found {NUM_CLASSES} classes: {CLASS_NAMES}")
except ImportError:
    print("could not import from 'chexpert_loader.py'.")
    exit()
except Exception as e:
    print(f"An error occurred during import: {e}")
    exit()

#seting up the 12-Class Model
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
num_features = model.classifier[1].in_features
model.classifier[1] = nn.Linear(in_features=num_features, out_features=NUM_CLASSES)
print(f"Model classifier replaced. Output features: {NUM_CLASSES}")
model.to(device)

# setting up the auroc metric
auroc_metric = MultilabelAUROC(
    num_labels=NUM_CLASSES, 
    average="none" 
).to(device)


# do the the evaluation
print("\nStarting baseline evaluation")
model.eval()
with torch.no_grad():
    for images, labels in tqdm(val_loader, desc="Evaluating Baseline"):
        images = images.to(device)
        labels = labels.to(device) 
        outputs = model(images)
        preds = torch.sigmoid(outputs)
        auroc_metric.update(preds, labels.int())

# the final scores per class
individual_auroc_scores = auroc_metric.compute()
scores_np = individual_auroc_scores.cpu().numpy()
macro_avg_auroc = np.mean(scores_np)

print("\n Baseline Results")
print(f"Overall (Macro) AUROC: {macro_avg_auroc:.4f}")
print("Individual Scores by Pathology:")

# create a dataframe for later graphs
results_df = pd.DataFrame({
    'Pathology': CLASS_NAMES,
    'AUROC': scores_np
})
results_df = results_df.sort_values(by='AUROC', ascending=False)
print(results_df.to_string(index=False))

csv_file_name = "baseline_auroc.csv"
try:
    results_df.to_csv(csv_file_name, index=False)
    print(f"\nsaved individual scores to: {os.path.abspath(csv_file_name)}")
except Exception as e:
    print(f"Error saving CSV file: {e}")

# generate and save the bar graph
plot_file_name = "baseline_auroc.png"
try:
    plt.figure(figsize=(10, 8))
    plt.barh(results_df['Pathology'], results_df['AUROC'])
    plt.gca().invert_yaxis()
    plt.xlabel('AUROC Score')
    plt.ylabel('Pathology')
    plt.title('Baseline (Zero-Shot) AUROC Scores by Pathology')
    plt.xlim(0, 1.0)
    
    for index, value in enumerate(results_df['AUROC']):
        plt.text(value + 0.01, index, f"{value:.3f}", va='center')
        
    plt.tight_layout()
    plt.savefig(plot_file_name)
    print(f"saved bar graph to: {os.path.abspath(plot_file_name)}")

except Exception as e:
    print(f"Error saving plot: {e}")

auroc_metric.reset()
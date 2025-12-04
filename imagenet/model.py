import torch
import torch.nn as nn
import torchvision.models as models
from torchmetrics.classification import MultilabelAUROC
from tqdm import tqdm
import os
import matplotlib.pyplot as plot
import pandas as panda
import numpy as np

# import from the chexpert_loader.py
try:
    from chexpert_loader import val_loader, NUM_CLASSES, CLASSIFICATIONS
    print("Imported 'val_loader' and 'NUM_CLASSES' from chexpert_loader.py")
    if NUM_CLASSES != 12 or len(CLASSIFICATIONS) != 12:
        print(f"Warning: Expected 12 classes, but found {NUM_CLASSES} and {len(CLASSIFICATIONS)} labels.")
    print(f"Found {NUM_CLASSES} classes: {CLASSIFICATIONS}")
except ImportError:
    print("could not import from 'chexpert_loader.py'.")
    exit()
except Exception as e:
    print(f"An error occurred during import: {e}")
    exit()

#seting up the 12-Class Model
device = "cuda" if torch.cuda.is_available() else "cpu"
model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
feature_size = model.classifier[1].in_features
model.classifier[1] = nn.Linear(in_features=feature_size, out_features=NUM_CLASSES)
print(f"Model classifier replaced. Output features: {NUM_CLASSES}")
model.to(device)

# setting up the auroc metric
auroc = MultilabelAUROC(
    num_labels=NUM_CLASSES, 
    average="none" 
).to(device)


# do the the evaluation
print("\nStarting baseline evaluation")
model.eval()
with torch.no_grad():
    for img, labels in tqdm(val_loader, desc="Evaluating Baseline"):
        img = img.to(device)
        labels = labels.to(device) 
        auroc.update(torch.sigmoid(model(img.to(device))), labels.int())

# the final scores per class
scores_np = auroc.compute().cpu().numpy()
avg_auroc = np.mean(scores_np)

print("\nBaseline Results")
print(f"Overall AUROC: {avg_auroc:.4f}")
print("Individual Scores by Pathology:")

# create a dataframe for later graphs
results_df = panda.DataFrame({
    'Pathology': CLASSIFICATIONS,
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
graph_file_name = "baseline_auroc.png"
try:
    plot.figure(figsize=(10, 8))
    plot.barh(results_df['Pathology'], results_df['AUROC'])
    plot.gca().invert_yaxis()
    plot.xlabel('AUROC Score')
    plot.ylabel('Pathology')
    plot.title('Baseline (Zero-Shot) AUROC Scores by Pathology')
    plot.xlim(0, 1.0)
    
    for index, value in enumerate(results_df['AUROC']):
        plot.text(value + 0.01, index, f"{value:.3f}", va='center')
        
    plot.tight_layout()
    plot.savefig(graph_file_name)
    print(f"saved bar graph to: {os.path.abspath(graph_file_name)}")

except Exception as e:
    print(f"Error saving plot: {e}")

auroc.reset()
import torch
import torch.nn as nn
import torchvision.models as models
from torchmetrics.classification import MultilabelAUROC
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from peft import PeftModel

# import from the dataloader
try:
    from chexpert_loader import val_loader, NUM_CLASSES, CLASSIFICATIONS
    print("Successfully imported 'val_loader' and 'NUM_CLASSES' from chexpert_loader.py")
    print(f"found {NUM_CLASSES} classes: {CLASSIFICATIONS}")
except ImportError:
    print("\n--- ERROR ---")
    print("Could not import from 'chexpert_loader.py'.")
    exit()

# we must first build the same model structure that we trained
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# then load the base EfficientNet-B0 (with ImageNet weights)
base_model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
# replace the final classifier layer (same as before)
num_features = base_model.classifier[1].in_features
base_model.classifier[1] = nn.Linear(in_features=num_features, out_features=NUM_CLASSES)
print(f"base model classifier replaced. Output features: {NUM_CLASSES}")
#  lastly load the trained LoRA adapters
adapter_save_dir = "/home/hice1/achen448/scratch/CS6220/cs6220-project/imagenet/results/lora5/"
print(f"Loading trained LoRA adapters from: {adapter_save_dir}")

try:
    # loads the base model and then injects the saved adapters
    model = PeftModel.from_pretrained(base_model, adapter_save_dir)
    print("loaded LoRA adapters.")
except Exception as e:
    print(f"\n--- ERROR ---")
    print(f"Could not load adapters from {adapter_save_dir}.")
    print(f"Error details: {e}\n")
    exit()

# move the LoRA model to the device
model.to(device)
#seting up the AUROC Metric
auroc_metric = MultilabelAUROC(
    num_labels=NUM_CLASSES, 
    average="none"
).to(device)


#run the evaluation
print("\nStarting fine-tuned evaluation...")
model.eval()
with torch.no_grad():
    for images, labels in tqdm(val_loader, desc="Evaluating Finetuned Model"):
        images = images.to(device)
        labels = labels.to(device) 
        #get predictions from the LoRA model
        outputs = model(images)
        preds = torch.sigmoid(outputs)
        auroc_metric.update(preds, labels.int())

#the final scores per class
individual_auroc_scores = auroc_metric.compute()

scores_np = individual_auroc_scores.cpu().numpy()
macro_avg_auroc = np.mean(scores_np)

print("\nfinetuned Results")
print(f"Overall (Macro) AUROC: {macro_avg_auroc:.4f}")
print("Individual Scores by Pathology:")

results_df = pd.DataFrame({
    'Pathology': CLASSIFICATIONS,
    'AUROC': scores_np
})
results_df = results_df.sort_values(by='AUROC', ascending=False)
print(results_df.to_string(index=False))

csv_file_name = "finetuned_auroc_5.csv"
plot_file_name = "finetuned_auroc_5.png"

try:
    results_df.to_csv(csv_file_name, index=False)
    print(f"\nsaved individual scores to: {os.path.abspath(csv_file_name)}")
except Exception as e:
    print(f"Error saving CSV file: {e}")

# generate and save the bar graph
try:
    plt.figure(figsize=(10, 8))
    plt.barh(results_df['Pathology'], results_df['AUROC'])
    plt.gca().invert_yaxis()
    plt.xlabel('AUROC Score')
    plt.ylabel('Pathology')
    plt.title('Finetuned (LoRA) AUROC Scores by Pathology')
    plt.xlim(0, 1.0)
    for index, value in enumerate(results_df['AUROC']):
        plt.text(value + 0.01, index, f"{value:.3f}", va='center')
    plt.tight_layout()
    plt.savefig(plot_file_name)
    print(f"saved bar graph to: {os.path.abspath(plot_file_name)}")

except Exception as e:
    print(f"Error saving plot: {e}")

auroc_metric.reset()
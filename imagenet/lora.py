import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
from peft import LoraConfig, get_peft_model

TRAIN_CSV_PATH = "/home/hice1/achen448/scratch/CS6220/cs6220-project/dataset/CheXpert-v1.0-small/train.csv"
VAL_CSV_PATH = "/home/hice1/achen448/scratch/CS6220/cs6220-project/dataset/CheXpert-v1.0-small/valid.csv"
IMAGE_ROOT_DIR = "/home/hice1/achen448/scratch/CS6220/cs6220-project/dataset/"

BATCH_SIZE = 32
NUM_CLASSES = 11      # removed fracture
LEARNING_RATE = 1e-5  # lowered to prevent overfitting
NUM_EPOCHS = 5       
WORKER_COUNT = 1 

# settting up labels and processcing function
CLASS_NAMES = [
    'Atelectasis', 
    'Cardiomegaly', 
    'Consolidation', 
    'Edema', 
    'Enlarged Cardiomediastinum',
    'Lung Lesion',
    'Lung Opacity',
    'Pneumonia',
    'Pneumothorax',
    'Pleural Effusion',
    'Pleural Other'
]

#converting the values: NaN to 0, -1 to 1
def process_chexpert_labels(df, class_names):
    df_processed = df.copy()
    df_processed[class_names] = df_processed[class_names].fillna(0)
    df_processed[class_names] = df_processed[class_names].replace(-1, 1)
    return df_processed

# setting up image transformations
IMG_SIZE = 224
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])

val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])

# creating pytorch dataset
class CheXpertDataset(Dataset):
    def __init__(self, df, image_root_dir, class_names, transform):
        self.df = df
        self.image_root_dir = image_root_dir
        self.transform = transform
        self.class_names = class_names

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = os.path.join(self.image_root_dir, row['Path'])
        image = Image.open(image_path).convert("RGB")
        labels = torch.tensor(row[self.class_names].values.astype(np.float32))
        
        if self.transform:
            image = self.transform(image)
            
        return image, labels

print("Setting up DataLoaders...")
# create the dataloaders
train_df_raw = pd.read_csv(TRAIN_CSV_PATH)
train_df = process_chexpert_labels(train_df_raw, CLASS_NAMES)
train_dataset = CheXpertDataset(train_df, IMAGE_ROOT_DIR, CLASS_NAMES, train_transform)
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=WORKER_COUNT
)

# create the validation dataloader
val_df_raw = pd.read_csv(VAL_CSV_PATH)
val_df = process_chexpert_labels(val_df_raw, CLASS_NAMES)
val_dataset = CheXpertDataset(val_df, IMAGE_ROOT_DIR, CLASS_NAMES, val_transform)
val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=WORKER_COUNT
)

print(f"Train loader: {len(train_loader)} batches")
print(f"Valid loader: {len(val_loader)} batches")

# setting up the model and applying LoRA
print("Setting up model and applying LoRA...")
device = "cuda" if torch.cuda.is_available() else "cpu"
model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
num_features = model.classifier[1].in_features
model.classifier[1] = nn.Linear(in_features=num_features, out_features=NUM_CLASSES)
target_modules = []
for name, module in model.named_modules():
    if isinstance(module, nn.Conv2d):
        if module.groups == 1:
            target_modules.append(name)
target_modules.append("classifier.1") 
print(f"found {len(target_modules)} modules to target with LoRA.")
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=target_modules,
    lora_dropout=0.05,
    bias="none",
)

lora_model = get_peft_model(model, lora_config)
lora_model.print_trainable_parameters()
lora_model.to(device)

# start the fine-tuning process with the validation and early stopping
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(lora_model.parameters(), lr=LEARNING_RATE)

# lists to store losses for epoch-level for later graph generation
all_batch_losses = []
all_epoch_train_losses = []
all_epoch_val_losses = []

# for early stopping when validation loss does not improve
best_val_loss = np.inf
epochs_without_improvement = 0
early_stopping_patience = 2 # this would stop after 2 epochs of no improvement
adapter_save_dir = "/home/hice1/achen448/scratch/CS6220/cs6220-project/imagenet/results/lora5"
print("\nstarting LoRA fine-tuning...")
for epoch in range(NUM_EPOCHS):
    print(f"\n--- Epoch {epoch + 1}/{NUM_EPOCHS} ---")
    # start training phase
    lora_model.train()
    running_loss_30_batch = 0.0
    start_time_30_batch = time.time()
    epoch_train_loss = 0.0

    for i, (images, labels) in enumerate(tqdm(train_loader, desc="Training")):
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = lora_model(images)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        current_loss = loss.item()
        epoch_train_loss += current_loss
        
        running_loss_30_batch += current_loss
        all_batch_losses.append(current_loss)

        if (i + 1) % 30 == 0:
            end_time_30_batch = time.time()
            duration = end_time_30_batch - start_time_30_batch
            avg_loss_30_batch = running_loss_30_batch / 30
            
            print(f"  [Epoch {epoch + 1}, Batch {i + 1}]")
            print(f"    - Avg Loss (last 30 batches): {avg_loss_30_batch:.4f}")
            print(f"    - Time (last 30 batches): {duration:.2f} seconds")
            
            running_loss_30_batch = 0.0
            start_time_30_batch = time.time()

    avg_epoch_train_loss = epoch_train_loss / len(train_loader)
    all_epoch_train_losses.append(avg_epoch_train_loss)
    print(f"Epoch {epoch+1} Average Training Loss: {avg_epoch_train_loss:.4f}")

    # validation phase
    print(f"Running validation for Epoch {epoch+1}...")
    lora_model.eval()
    epoch_val_loss = 0.0
    
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Validating"):
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = lora_model(images)
            loss = criterion(outputs, labels)
            
            epoch_val_loss += loss.item()
            
    avg_epoch_val_loss = epoch_val_loss / len(val_loader)
    all_epoch_val_losses.append(avg_epoch_val_loss)
    print(f"Epoch {epoch+1} Average Validation Loss: {avg_epoch_val_loss:.4f}")
    
    # stop if the validation loss did not improve
    if avg_epoch_val_loss < best_val_loss:
        print(f"  Validation loss improved ({best_val_loss:.4f} --> {avg_epoch_val_loss:.4f})")
        print(f"  Saving best model to {adapter_save_dir}...")
        
        lora_model.save_pretrained(adapter_save_dir)
        best_val_loss = avg_epoch_val_loss
        epochs_without_improvement = 0
    else:
        epochs_without_improvement += 1
        print(f"  Validation loss did not improve ({best_val_loss:.4f}).")
        print(f"  Patience: {epochs_without_improvement}/{early_stopping_patience}")

    if epochs_without_improvement >= early_stopping_patience:
        print(f"\n--- Early stopping triggered after {epoch + 1} epochs ---")
        break # stop the training loop

print(f"\nTraining complete. Best model saved in {adapter_save_dir}")

# generate a batch-by-batch training loss graph
print("\nGenerating batch-level training loss curve...")
try:
    loss_df = pd.DataFrame({'batch_loss': all_batch_losses})
    rolling_avg_loss = loss_df['batch_loss'].rolling(window=50).mean()

    plt.figure(figsize=(12, 6))
    plt.plot(loss_df['batch_loss'], label='Loss per Batch', alpha=0.3)
    plt.plot(rolling_avg_loss, label='Smoothed Loss (50-batch avg)', color='red', linewidth=2)
    plt.xlabel('Batch Number')
    plt.ylabel('Loss (BCEWithLogitsLoss)')
    plt.title('Training Loss Curve (Batch-by-Batch)')
    plt.legend()
    plt.grid(True)
    
    plot_file_name = "training_loss_curve.png"
    plt.savefig(plot_file_name)
    print(f"Batch loss curve saved to {plot_file_name}")

except Exception as e:
    print(f"Error generating batch loss plot: {e}")

# generating epoch-level training vs. validation Loss graph
print("\ngenerating epoch-level loss comparison graph")
try:
    epoch_count = len(all_epoch_train_losses)
    epoch_numbers = range(1, epoch_count + 1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(epoch_numbers, all_epoch_train_losses, 'o-', label='Training Loss')
    plt.plot(epoch_numbers, all_epoch_val_losses, 'o-', label='Validation Loss')
    
    plt.xlabel('Epoch')
    plt.ylabel('Average Loss')
    plt.title('Training vs. Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.xticks(epoch_numbers) 
    
    epoch_plot_file = "epoch_loss_comparison.png"
    plt.savefig(epoch_plot_file)
    print(f"Epoch loss comparison curve saved to {epoch_plot_file}")

except Exception as e:
    print(f"Error generating epoch loss plot: {e}")
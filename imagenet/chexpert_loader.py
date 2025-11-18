import torch
import pandas as pd
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

VAL_CSV_PATH = "/home/hice1/achen448/scratch/CS6220/cs6220-project/dataset/CheXpert-v1.0-small/valid.csv" 
IMAGE_ROOT_DIR = "/home/hice1/achen448/scratch/CS6220/cs6220-project/dataset/" 
BATCH_SIZE = 32
NUM_CLASSES = 11
CLASS_NAMES = [
    'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 
    'Enlarged Cardiomediastinum', 'Lung Lesion',
    'Lung Opacity', 'Pneumonia', 'Pneumothorax', 'Pleural Effusion',
    'Pleural Other'
]

def process_chexpert_labels(df, class_names):
    print("Processing labels...")
    df_processed = df.copy()
    # convert NaN to 0
    df_processed[class_names] = df_processed[class_names].fillna(0)
    # convert -1 to 1
    df_processed[class_names] = df_processed[class_names].replace(-1, 1)
    print("Label processing complete.")
    return df_processed

# define Image Transformations, EfficientNet-B0 expects 224x224 imagesm and normalize with ImageNet stats since the model was pretrained on it
IMG_SIZE = 224
val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# create custom pytorch dataset
class CheXpertValidationDataset(Dataset):
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
        # apply transforms
        if self.transform:
            image = self.transform(image)
        return image, labels

print("Setting up validation dataloader...")
# load the CSV
try:
    val_df_raw = pd.read_csv(VAL_CSV_PATH)
except FileNotFoundError:
    print(f"Error: Could not find validation CSV")
# need to process the labels
val_df_processed = process_chexpert_labels(val_df_raw, CLASS_NAMES)
# create the dataset
val_dataset = CheXpertValidationDataset(
    df=val_df_processed,
    image_root_dir=IMAGE_ROOT_DIR,
    class_names=CLASS_NAMES,
    transform=val_transform
)
# create the dataLoader
val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False, 
    num_workers=2
)
print(f"  - Total validation samples: {len(val_dataset)}")
print(f"  - Batches per epoch: {len(val_loader)}")
print(f"  - Batch size: {BATCH_SIZE}")
print(f"  - Number of classes: {NUM_CLASSES}")
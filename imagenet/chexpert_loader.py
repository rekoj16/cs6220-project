import torch
import pandas as panda
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

VAL_CSV = "/home/hice1/achen448/scratch/CS6220/cs6220-project/dataset/CheXpert-v1.0-small/valid.csv" 
ROOT_DIR = "/home/hice1/achen448/scratch/CS6220/cs6220-project/dataset/" 
BATCH_SIZE = 32
NUM_CLASSES = 11
CLASSIFICATIONS = [
    'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Enlarged Cardiomediastinum', 'Lung Lesion',
    'Lung Opacity', 'Pneumonia', 'Pneumothorax', 'Pleural Effusion', 'Pleural Other'
]

def process_chexpert_labels(df, class_names):
    print("Processing labels...")
    processed = df.copy()
    # convert NaN to 0
    processed[class_names] = processed[class_names].fillna(0)
    # convert -1 to 1
    processed[class_names] = processed[class_names].replace(-1, 1)
    print("Label processing complete.")
    return processed

# define Image Transformations, EfficientNet-B0 expects 224x224 imagesm and normalize with ImageNet stats since the model was pretrained on it
size = 224
val_transform = transforms.Compose([
    transforms.Resize((size, size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# create custom pytorch dataset
class CheXpertValidationDataset(Dataset):
    def __init__(self, df, image_root_dir, class_names, transform):
        self.df = df
        self.image_root_dir = image_root_dir
        self.class_names = class_names
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(os.path.join(self.image_root_dir, row['Path'])).convert("RGB")
        labels = torch.tensor(row[self.class_names].values.astype(np.float32))
        if self.transform:
            image = self.transform(image)
        return image, labels

print("Setting up validation dataloader...")
# load the CSV
try:
    val_raw = panda.read_csv(VAL_CSV)
except FileNotFoundError:
    print(f"Error: did not find valid.csv file")
# need to process the labels
val_processed = process_chexpert_labels(val_raw, CLASSIFICATIONS)
# create the dataset
val_dataset = CheXpertValidationDataset(
    df=val_processed,
    image_root_dir=ROOT_DIR,
    class_names=CLASSIFICATIONS,
    transform=val_transform
)
# create the dataLoader
val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False, 
    num_workers=2
)
print(f"Total validation samples: {len(val_dataset)}")
print(f"Batches per epoch: {len(val_loader)}")
print(f"Batch size: {BATCH_SIZE}")
print(f"Number of classes: {NUM_CLASSES}")
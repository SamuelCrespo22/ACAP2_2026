# Dataloader
import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib


class ButterflyDataset(Dataset):
    """
    Class to load images of butterflies.
    """
    def __init__(self, dataframe, img_dir, transform=None):
        """
        dataframe: Pandas DataFrame with filename and label.
        img_dir: Path to the folder where images are stored ('data/train/').
        transform: torchvision transformations to apply to the image.
        """
        self.dataframe = dataframe
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        """
        Given an index, load the corresponding image and label, apply transformations, and return them as tensors.
        """
        img_name = os.path.join(self.img_dir, self.dataframe.iloc[idx, 0])
        image = Image.open(img_name).convert("RGB")
        label = self.dataframe.iloc[idx, 1]

        if self.transform:
            image = self.transform(image)
            
        return image, torch.tensor(label, dtype=torch.long)

def get_dataloaders(csv_path, img_dir, batch_size=32, img_size=224, test_size=0.2):
    """
    Read CSV, convert text labels to numbers, split into train and validation, and return ready-to-use DataLoaders.
    """
    df = pd.read_csv(csv_path)

    col_filename = df.columns[0]
    col_label = df.columns[1]
    
    # Convert labels to numbers from 0 to 74 (75 classes).
    label_encoder = LabelEncoder()
    df['label_encoded'] = label_encoder.fit_transform(df[col_label])
    
    df_clean = df[[col_filename, 'label_encoded']]
    
    # TT 80/20 with stratification.
    train_df, val_df = train_test_split(df_clean, test_size=test_size, stratify=df_clean['label_encoded'], random_state=42)
    
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)

    # Data Augmentation and Normalization
    train_transforms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
        transforms.ToTensor(),             # [C, H, W] with values from 0 to 1.
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transforms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = ButterflyDataset(train_df, img_dir, transform=train_transforms)
    val_dataset = ButterflyDataset(val_df, img_dir, transform=val_transforms)

    num_workers = 2
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, persistent_workers=num_workers > 0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, persistent_workers=num_workers > 0)

    joblib.dump(label_encoder, 'label_encoder.pkl')

    return train_loader, val_loader, label_encoder.classes_
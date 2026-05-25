import os
from PIL import Image
import pandas as pd
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split

class ButterflyDataset(data.Dataset):
    def __init__(self, df, img_dir, transform=None):
        self.img_labels = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform

        self.classes = sorted(self.img_labels['label'].unique())
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_name = self.img_labels.iloc[idx]['filename']
        img_path = os.path.join(self.img_dir, img_name)

        image = Image.open(img_path).convert("RGB")

        label_name = self.img_labels.iloc[idx]['label']
        label_idx = self.class_to_idx[label_name]
        label = torch.tensor(label_idx, dtype=torch.long)

        if self.transform:
            image = self.transform(image)

        return image, label


def get_dataloaders(
    csv_path,
    img_dir,
    batch_size=32,
    img_size=64,
    val_size=0.2,
    test_size=0.0,
    model_type="classifier",
    num_workers=2
):
    """
    Loads and prepares dataloaders for training, validation, and optionally testing.
    
    Args:
        csv_path (str): Path to the CSV file.
        img_dir (str): Path to the images directory.
        batch_size (int): Number of images per batch.
        img_size (int): Size to resize the images to.
        val_size (float): Proportion of data for validation.
        test_size (float): Proportion of data for an internal test set.
        model_type (str): "classifier" for standard preprocessing (no data augmentation).
                          "generative" for preprocessing with data augmentation.
        num_workers (int): Number of CPU subprocesses to use for data loading.
                          
    Returns:
        train_loader, val_loader, test_loader, classes
    """
    assert val_size + test_size < 1.0, "val_size and test_size combined must be less than 1.0 (need data for training)."

    df = pd.read_csv(csv_path)
    
    pin_memory = torch.cuda.is_available()

    base_transforms = [transforms.Resize((img_size, img_size))]

    if model_type == "generative":
        train_transforms = transforms.Compose(base_transforms + [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ToTensor(),
        ])
    else:
        # The classifier cannot use data augmentation
        train_transforms = transforms.Compose(base_transforms + [
            transforms.ToTensor(),
        ])

    # Validation/Test exclusively use standard preprocessing
    eval_transforms = transforms.Compose(base_transforms + [
        transforms.ToTensor(),
    ])

    train_loader, val_loader, test_loader = None, None, None

    # First split: Separate Test set (if requested)
    if test_size > 0:
        train_val_df, test_df = train_test_split(
            df, test_size=test_size, random_state=42, stratify=df['label']
        )
        test_dataset = ButterflyDataset(df=test_df, img_dir=img_dir, transform=eval_transforms)
        test_loader = data.DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory
        )
    else:
        train_val_df = df

    # Second split: Separate Validation set (if requested)
    if val_size > 0:
        # Adjust validation proportion relative to the remaining training+validation data
        relative_val_size = val_size / (1.0 - test_size)
        
        train_df, val_df = train_test_split(
            train_val_df, test_size=relative_val_size, random_state=42, stratify=train_val_df['label']
        )
        
        val_dataset = ButterflyDataset(df=val_df, img_dir=img_dir, transform=eval_transforms)
        val_loader = data.DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory
        )
    else:
        train_df = train_val_df

    # Final Training Dataset
    train_dataset = ButterflyDataset(df=train_df, img_dir=img_dir, transform=train_transforms)
    
    # drop_last=True is very useful for generative models so incomplete batches don't break layers
    train_loader = data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=num_workers, pin_memory=pin_memory
    )
    
    # Get class names from dataset
    classes = train_dataset.classes

    return train_loader, val_loader, test_loader, classes
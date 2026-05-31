import os
from PIL import Image
import pandas as pd
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split


class ButterflyDataset(data.Dataset):
    def __init__(self, df, img_dir, transform=None, classes=None):
        self.img_labels = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform

        self.filenames = self.img_labels["filename"].tolist()
        self.labels = self.img_labels["label"].tolist()
        if "img_dir" in self.img_labels.columns:
            self.img_dirs = self.img_labels["img_dir"].tolist()
        else:
            self.img_dirs = [self.img_dir] * len(self.img_labels)

        if classes is None:
            self.classes = sorted(self.img_labels["label"].unique())
        else:
            self.classes = classes

        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}

        print(f"Pre loading {len(self.filenames)} images to RAM...")
        self.images_cache = []
        for i in range(len(self.filenames)):
            img_path = os.path.join(self.img_dirs[i], self.filenames[i])
            self.images_cache.append(Image.open(img_path).convert("RGB"))
        print("Pre loading completed.")

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        image = self.images_cache[idx]

        label_name = self.labels[idx]
        if label_name not in self.class_to_idx:
            raise ValueError(
                f"Label '{label_name}' not found in classes mapping. "
                f"Expected one of: {self.classes}"
            )

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
    val_size=0.15,
    test_size=0.15,
    augment=False,
    normalize=True,
    aug_csv_path=None,
    aug_img_dir=None,
    num_workers=2,
):
    assert val_size + test_size < 1.0, (
        "val_size and test_size combined must be less than 1.0."
    )

    df = pd.read_csv(csv_path)
    df['img_dir'] = img_dir 
    classes = sorted(df["label"].unique())

    try:
        import torch_directml
        has_dml = torch_directml.is_available()
    except ImportError:
        has_dml = False
    pin_memory = torch.cuda.is_available() or has_dml

    mean = (0.5, 0.5, 0.5)
    std = (0.5, 0.5, 0.5)

    train_transforms = [
        transforms.Resize((img_size, img_size))
    ]

    if augment:
        train_transforms += [
            transforms.RandomHorizontalFlip(p=0.5),

            # Reflect padding before rotation to avoid black corners.
            transforms.Pad(
                padding=8,
                padding_mode="reflect"
            ),

            transforms.RandomRotation(
                degrees=10,
                expand=False,
                fill=0
            ),

            transforms.CenterCrop(img_size),
        ]

    train_transforms += [
        transforms.ToTensor()
    ]

    if normalize:
        train_transforms += [
            transforms.Normalize(mean=mean, std=std)
        ]

    train_transforms = transforms.Compose(train_transforms)

    eval_transforms = [
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor()
    ]

    if normalize:
        eval_transforms += [
            transforms.Normalize(mean=mean, std=std)
        ]

    eval_transforms = transforms.Compose(eval_transforms)

    test_loader = None

    if test_size > 0:
        train_val_df, test_df = train_test_split(
            df,
            test_size=test_size,
            random_state=42,
            stratify=df["label"]
        )

        test_dataset = ButterflyDataset(
            df=test_df,
            img_dir=img_dir,
            transform=eval_transforms,
            classes=classes,
        )

        test_loader = data.DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
    else:
        train_val_df = df

    if val_size > 0:
        relative_val_size = val_size / (1.0 - test_size)

        train_df, val_df = train_test_split(
            train_val_df,
            test_size=relative_val_size,
            random_state=42,
            stratify=train_val_df["label"],
        )

        val_dataset = ButterflyDataset(
            df=val_df,
            img_dir=img_dir,
            transform=eval_transforms,
            classes=classes,
        )

        val_loader = data.DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
    else:
        train_df = train_val_df
        val_loader = None

    if aug_csv_path is not None and aug_img_dir is not None:
        aug_df = pd.read_csv(aug_csv_path)
        aug_df['img_dir'] = aug_img_dir
        aug_df = aug_df[~aug_df['filename'].isin(df['filename'])]
        train_df = pd.concat([train_df, aug_df], ignore_index=True)

    train_dataset = ButterflyDataset(
        df=train_df,
        img_dir=img_dir,
        transform=train_transforms,
        classes=classes,
    )

    train_loader = data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader, test_loader, classes
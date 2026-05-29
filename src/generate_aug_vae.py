import os
import shutil
import torch
import pandas as pd
from torchvision.utils import save_image
from dataset import get_dataloaders
from models.vae import ConvVAE 

def generate_augmented_vae():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    generate_per_class = 50
    output_dir = "data/train_aug_vae"
    output_csv = "data/train_aug_vae.csv"
    os.makedirs(output_dir, exist_ok=True)

    train_loader, _, _, classes = get_dataloaders(
        csv_path="data/train.csv",
        img_dir="data/train",
        val_size=0.15,
        test_size=0.15,
        augment=False,
        normalize=False
    )
    train_df = train_loader.dataset.img_labels
    
    print("Copying original images...")
    for _, row in train_df.iterrows():
        shutil.copy2(os.path.join("data/train", row['filename']), os.path.join(output_dir, row['filename']))

    # Load Model
    model = ConvVAE(latent_dim=128, num_classes=len(classes)).to(device)
    model.load_state_dict(torch.load("results_gen/best_vae.pth", map_location=device))
    model.eval()

    new_rows = []
    
    # Generate Images
    with torch.no_grad():
        for class_idx, class_name in enumerate(classes):
            to_generate = generate_per_class
            print(f"VAE generating {to_generate} images for class: {class_name}...")
            z = torch.randn(to_generate, 128).to(device) # VAE needs (N, latent_dim)
            labels = torch.full((to_generate,), class_idx, dtype=torch.long, device=device)

            samples = model.decode(z, labels) # [0, 1]

            for i in range(to_generate):
                img_name = f"gen_vae_{class_idx}_{i}.jpg"
                save_image(samples[i], os.path.join(output_dir, img_name))
                new_rows.append({"filename": img_name, "label": class_name})

    pd.concat([train_df, pd.DataFrame(new_rows)], ignore_index=True).to_csv(output_csv, index=False)
    print(f"VAE Dataset concluded: {output_csv}")

if __name__ == "__main__":
    generate_augmented_vae()
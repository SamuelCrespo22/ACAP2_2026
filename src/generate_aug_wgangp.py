import os
import shutil
import torch
import pandas as pd
from torchvision.utils import save_image
from dataset import get_dataloaders
from models.wgan_gp import Generator 

def generate_augmented_wgangp():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    z_dim = 100
    output_dir = "data/train_aug_wgangp"
    output_csv = "data/train_aug_wgangp.csv"
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
    full_df = pd.read_csv("data/train.csv")
    
    print("Copying original images...")
    for _, row in train_df.iterrows():
        shutil.copy2(os.path.join("data/train", row['filename']), os.path.join(output_dir, row['filename']))

    # Load Model
    netG = Generator(inputDim=z_dim, num_classes=len(classes)).to(device)
    netG.load_state_dict(torch.load("results_gen/wgan_n5_500ep_padding_final/best_wgangp_g.pth", map_location=device))
    netG.eval()

    new_rows = []
    
    # Generate Images
    with torch.no_grad():
        for class_idx, class_name in enumerate(classes):
            class_count = len(full_df[full_df['label'] == class_name])
            
            if 51 <= class_count <= 60:
                to_generate = int(round(class_count * 0.20))
            elif 61 <= class_count <= 70:
                to_generate = int(round(class_count * 0.15))
            elif 71 <= class_count <= 80:
                to_generate = int(round(class_count * 0.10))
            elif 81 <= class_count <= 90:
                to_generate = int(round(class_count * 0.05))
            else:
                to_generate = 0

            if to_generate <= 0:
                continue

            print(f"WGAN-GP generating {to_generate} images for class: {class_name} (Current count: {class_count})...")

            z = torch.randn(to_generate, z_dim, 1, 1, device=device)
            labels = torch.full((to_generate,), class_idx, dtype=torch.long, device=device)

            samples = netG(z, labels)

            # Revert from Tanh [-1, 1] to [0, 1]
            samples = (samples + 1.0) / 2.0 
            samples = torch.clamp(samples, 0.0, 1.0) 

            for i in range(to_generate):
                img_name = f"gen_wgan_{class_idx}_{i}.jpg"
                save_image(samples[i], os.path.join(output_dir, img_name))
                new_rows.append({"filename": img_name, "label": class_name})

    pd.concat([train_df, pd.DataFrame(new_rows)], ignore_index=True).to_csv(output_csv, index=False)
    print(f"WGAN-GP Dataset concluded: {output_csv}")

if __name__ == "__main__":
    generate_augmented_wgangp()

import os
import shutil
import torch
import pandas as pd
from torchvision.utils import save_image
from dataset import get_dataloaders
from diffusion import UNet64, DDPM 

def generate_augmented_diffusion():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    target_per_class = 150
    output_dir = "data/train_aug_diff"
    output_csv = "data/train_aug_diff.csv"
    os.makedirs(output_dir, exist_ok=True)

    train_loader, _, _, classes = get_dataloaders(
        csv_path="data/train.csv", img_dir="data/train", val_size=0.2, augment=False, normalize=False
    )
    train_df = train_loader.dataset.img_labels
    
    print("A copiar imagens originais...")
    for _, row in train_df.iterrows():
        shutil.copy2(os.path.join("data/train", row['filename']), os.path.join(output_dir, row['filename']))

    model = UNet64(time_dim=128, num_classes=len(classes)).to(device)
    model.load_state_dict(torch.load("results_gen/best_diffusion.pth", map_location=device))
    model.eval()
    ddpm = DDPM(model=model, T=1000, device=device)

    new_rows = []
    batch_size_gen = 32 # Ajusta isto se der "Out of Memory" (Baixa para 16 ou 8)
    
    for class_idx, class_name in enumerate(classes):
        current_count = len(train_df[train_df['label'] == class_name])
        to_generate = target_per_class - current_count
        
        if to_generate <= 0: continue
            
        print(f"Diffusion a gerar {to_generate} imagens para: {class_name} (Demorado...)")
        
        generated_count = 0
        while generated_count < to_generate:
            n_samples = min(batch_size_gen, to_generate - generated_count)
            labels = torch.full((n_samples,), class_idx, dtype=torch.long, device=device)
            
            # --- Gerar imagens (Processo iterativo pesado) ---
            samples = ddpm.sample(n=n_samples, labels=labels) 
            
            # Reverter de [-1, 1] para [0, 1]
            samples = (samples + 1.0) / 2.0 
            samples = torch.clamp(samples, 0.0, 1.0)
            
            for i in range(n_samples):
                img_name = f"gen_diff_{class_idx}_{generated_count}.jpg"
                save_image(samples[i], os.path.join(output_dir, img_name))
                new_rows.append({"filename": img_name, "label": class_name})
                generated_count += 1

    pd.concat([train_df, pd.DataFrame(new_rows)], ignore_index=True).to_csv(output_csv, index=False)
    print(f"Dataset Diffusion concluído: {output_csv}")

if __name__ == "__main__":
    generate_augmented_diffusion()
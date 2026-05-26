import os
import csv
import torch
import torch.optim as optim
from dataset import get_dataloaders
from models.vae import ConvVAE, vae_loss
from utils import GenerativeEvaluator

def train_vae():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"A treinar VAE no dispositivo: {device}")

    # Configurations
    batch_size = 32
    epochs = 100 
    learning_rate = 1e-3
    eval_every = 5  # Calculate FID/IS every 5 epochs 
    os.makedirs("results_gen", exist_ok=True)
    csv_path = "results_gen/vae_metrics.csv"

    # Initialize metrics CSV
    with open(csv_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Epoch', 'Train_Loss', 'Val_Loss', 'FID', 'IS_mean', 'SSIM'])

    # Load Data
    train_loader, val_loader, _, _ = get_dataloaders(
        csv_path="data/train.csv",
        img_dir="data/train",
        batch_size=batch_size,
        img_size=64,
        val_size=0.15,
        test_size=0.15,
        augment=False,
        normalize=False, # VAE with BCE Loss needs [0, 1]
    )

    num_classes = len(val_loader.dataset.classes)
    
    # 2. Inicializar Modelo e Avaliador
    model = ConvVAE(latent_dim=128, num_classes=num_classes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    evaluator = GenerativeEvaluator(device=device)

    best_fid = float('inf')

    # 3. Loop de Treino
    for epoch in range(1, epochs + 1):
        model.train()
        running_train_loss = 0.0
        
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            
            recon_images, mu, logvar = model(images, labels)
            loss = vae_loss(recon_images, images, mu, logvar)
            
            loss.backward()
            optimizer.step()
            running_train_loss += loss.item()

        avg_train_loss = running_train_loss / len(train_loader)

        # Validação
        model.eval()
        running_val_loss = 0.0
        
        # Listas para acumular imagens para as métricas (limitado para não estoirar RAM)
        real_imgs_list = []
        recon_imgs_list = []
        
        with torch.no_grad():
            for i, (images, labels) in enumerate(val_loader):
                images = images.to(device)
                labels = labels.to(device)
                recon_images, mu, logvar = model(images, labels)
                
                loss = vae_loss(recon_images, images, mu, logvar)
                running_val_loss += loss.item()
                
                # Guardar um subset (ex: 500 imagens) para calcular FID e IS
                if len(real_imgs_list) * batch_size < 500:
                    real_imgs_list.append(images.cpu())
                    recon_imgs_list.append(recon_images.cpu())

        avg_val_loss = running_val_loss / len(val_loader)
        
        fid_val, is_mean, ssim_val = None, None, None

        # Calcular métricas generativas a cada 'eval_every' épocas ou na última
        if epoch % eval_every == 0 or epoch == epochs:
            print("A calcular métricas generativas (FID, IS, SSIM)...")
            real_tensor = torch.cat(real_imgs_list, dim=0)
            fake_tensor = torch.cat(recon_imgs_list, dim=0)
            
            # Gerar imagens totalmente novas a partir do espaço latente para o IS e FID
            with torch.no_grad():
                z = torch.randn(fake_tensor.size(0), model.latent_dim).to(device)
                random_labels = torch.randint(0, num_classes, (fake_tensor.size(0),)).to(device)
                generated_imgs = model.decode(z, random_labels).cpu()
            
            # SSIM (usa reconstruções vs reais)
            ssim_val = evaluator.compute_ssim(fake_tensor, real_tensor, is_tanh=False)
            
            # FID e IS (usam imagens geradas do zero)
            fid_val = evaluator.compute_fid(real_tensor, generated_imgs)
            is_mean, _ = evaluator.compute_is(generated_imgs)
            
            print(f"Epoch [{epoch}/{epochs}] Train Loss: {avg_train_loss:.2f} | Val Loss: {avg_val_loss:.2f} | FID: {fid_val:.2f} | IS: {is_mean:.2f} | SSIM: {ssim_val:.4f}")

            # Guardar o modelo se o FID melhorar
            if fid_val < best_fid:
                best_fid = fid_val
                torch.save(model.state_dict(), "results_gen/best_vae.pth")
                print(" -> Melhor VAE guardado (Novo melhor FID)!")
        else:
            print(f"Epoch [{epoch}/{epochs}] Train Loss: {avg_train_loss:.2f} | Val Loss: {avg_val_loss:.2f}")

        # Guardar logs no CSV
        with open(csv_path, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch, 
                avg_train_loss, 
                avg_val_loss, 
                fid_val if fid_val else '', 
                is_mean if is_mean else '', 
                ssim_val if ssim_val else ''
            ])

if __name__ == "__main__":
    train_vae()
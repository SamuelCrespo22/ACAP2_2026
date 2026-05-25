# train_gen_model.py
import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

# Importações dos teus ficheiros fornecidos
from dataset import get_dataloaders
from models.vae import ConvVAE, vae_loss
from utils import GenerativeEvaluator

def train_vae_one_epoch(model, dataloader, optimizer, device):
    model.train()
    running_loss = 0.0
    
    for images, _ in dataloader:
        images = images.to(device)
        
        optimizer.zero_grad()
        recon_images, mu, logvar = model(images)
        
        loss = vae_loss(recon_images, images, mu, logvar)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()

    return running_loss / len(dataloader.dataset)

def validate_vae(model, dataloader, evaluator, device):
    model.eval()
    total_ssim = 0.0
    
    with torch.no_grad():
        for images, _ in dataloader:
            images = images.to(device)
            recon_images, _, _ = model(images)
            
            # Calcular o SSIM para este batch (comparando reconstrução com original)
            batch_ssim = evaluator.compute_ssim(recon_images, images, is_tanh=False)
            total_ssim += batch_ssim * images.size(0)
            
    return total_ssim / len(dataloader.dataset)

def main():
    # Parâmetros de Configuração
    csv_path = "data/train.csv"
    img_dir = "data/train"
    
    batch_size = 64
    img_size = 64        # Definido para a resolução ideal de 64x64
    latent_dim = 128
    epochs = 30
    lr = 0.001
    
    # Proporções do teu Corte Único Detalhado (Ex: 70/15/15)
    val_size = 0.15
    test_size = 0.15

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"A usar o dispositivo: {device}")

    # Carregar os Dataloaders usando o teu dataset.py
    # O test_loader conterá os teus 15% intocáveis.
    train_loader, val_loader, test_loader, classes = get_dataloaders(
        csv_path=csv_path,
        img_dir=img_dir,
        batch_size=batch_size,
        img_size=img_size,
        val_size=val_size,
        test_size=test_size,
        model_type="generative" # Ativa transformações básicas se necessário
    )
    
    print(f"Amostras de Treino: {len(train_loader.dataset)}")
    print(f"Amostras de Validação: {len(val_loader.dataset)}")
    print(f"Amostras de Teste Ocultas: {len(test_loader.dataset)}")

    # Inicializar o Modelo e Otimizador
    model = ConvVAE(latent_dim=latent_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Inicializar o nosso módulo de métricas generativas
    evaluator = GenerativeEvaluator(device=device)
    
    best_ssim = -1.0
    os.makedirs("checkpoints", exist_ok=True)

    # Loop de Treino e Validação
    for epoch in range(epochs):
        train_loss = train_vae_one_epoch(model, train_loader, optimizer, device)
        
        # Validação usando a métrica estrutural SSIM
        val_ssim = validate_vae(model, val_loader, evaluator, device)
        
        print(f"Época [{epoch+1}/{epochs}] | Train Loss: {train_loss:.4f} | Val SSIM: {val_ssim:.4f}")
        
        # Guardar o melhor modelo com base na qualidade da reconstrução
        if val_ssim > best_ssim:
            best_ssim = val_ssim
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'latent_dim': latent_dim,
                'best_ssim': best_ssim
            }, "checkpoints/best_vae.pth")
            print(f"-> Novo melhor checkpoint do VAE guardado (SSIM: {best_ssim:.4f})")

    print("\nTreino da Phase 1 Concluído!")
    print(f"Melhor SSIM alcançado na validação: {best_ssim:.4f}")

if __name__ == "__main__":
    main()
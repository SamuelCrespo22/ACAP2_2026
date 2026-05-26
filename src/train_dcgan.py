import os
import csv
import torch
import torch.nn as nn
import torch.optim as optim
from dataset import get_dataloaders
from dcgan import Generator, Discriminator, weights_init # Ajusta os imports se necessário
from utils import GenerativeEvaluator

def train_dcgan():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"A treinar DCGAN no dispositivo: {device}")

    # Configurações
    batch_size = 32
    epochs = 200
    lr = 2e-4
    beta1 = 0.5
    z_dim = 100
    eval_every = 5 # Calcular FID/IS a cada 5 épocas
    os.makedirs("results_gen", exist_ok=True)
    csv_path = "results_gen/dcgan_metrics.csv"

    # Inicializar CSV
    with open(csv_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Epoch', 'Loss_D', 'Loss_G', 'FID', 'IS_mean'])

    # Carregar Dados (normalize=False porque os dados do loader vêm em [0, 1])
    train_loader, val_loader, _, _ = get_dataloaders(
        csv_path="data/train.csv",
        img_dir="data/train",
        batch_size=batch_size,
        img_size=64,
        val_size=0.2,
        augment=False,
        normalize=False, 
    )

    num_classes = len(train_loader.dataset.classes)
    conditional = num_classes > 1

    netG = Generator(inputDim=z_dim, num_classes=num_classes if conditional else None).to(device)
    netD = Discriminator(num_classes=num_classes if conditional else None).to(device)
    netG.apply(weights_init)
    netD.apply(weights_init)

    criterion = nn.BCELoss()
    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
    evaluator = GenerativeEvaluator(device=device)

    best_fid = float('inf')
    real_label, fake_label = 1.0, 0.0

    for epoch in range(1, epochs + 1):
        running_lossD, running_lossG = 0.0, 0.0
        
        for i, (real_imgs, labels) in enumerate(train_loader):
            real_imgs = real_imgs.to(device)
            labels = labels.to(device)
            
            # DCGAN usa Tanh, logo precisa de dados em [-1, 1] para o treino
            real_imgs_scaled = real_imgs * 2.0 - 1.0
            cur_batch_size = real_imgs.size(0)

            # --- Treinar Discriminador ---
            netD.zero_grad()
            label_real = torch.full((cur_batch_size,), real_label, device=device)
            output_real = netD(real_imgs_scaled, labels).view(-1)
            lossD_real = criterion(output_real, label_real)

            z = torch.randn(cur_batch_size, z_dim, 1, 1, device=device)
            fake_imgs = netG(z, labels)
            label_fake = torch.full((cur_batch_size,), fake_label, device=device)
            output_fake = netD(fake_imgs.detach(), labels).view(-1)
            lossD_fake = criterion(output_fake, label_fake)

            lossD = lossD_real + lossD_fake
            lossD.backward()
            optimizerD.step()
            running_lossD += lossD.item()

            # --- Treinar Gerador ---
            netG.zero_grad()
            label_gen = torch.full((cur_batch_size,), real_label, device=device)
            output = netD(fake_imgs, labels).view(-1)
            lossG = criterion(output, label_gen)
            lossG.backward()
            optimizerG.step()
            running_lossG += lossG.item()

        avg_lossD = running_lossD / len(train_loader)
        avg_lossG = running_lossG / len(train_loader)

        fid_val, is_mean = None, None

        # --- Avaliação Generativa ---
        if epoch % eval_every == 0 or epoch == epochs:
            print("A calcular métricas generativas (FID e IS)...")
            netG.eval()
            real_imgs_list, fake_imgs_list = [], []
            
            with torch.no_grad():
                for val_imgs, val_labels in val_loader:
                    if len(real_imgs_list) * batch_size >= 500: # Limitar a ~500 imagens por rapidez
                        break
                    # Guardar imagens reais em [0, 1] para a avaliação
                    real_imgs_list.append(val_imgs) 
                    
                    z = torch.randn(val_imgs.size(0), z_dim, 1, 1, device=device)
                    val_labels = val_labels.to(device)
                    fakes = netG(z, val_labels)
                    
                    # Reverter de [-1, 1] para [0, 1] para o InceptionV3
                    fakes = (fakes + 1.0) / 2.0 
                    fakes = torch.clamp(fakes, 0.0, 1.0)
                    fake_imgs_list.append(fakes.cpu())
                    
            real_tensor = torch.cat(real_imgs_list, dim=0)
            fake_tensor = torch.cat(fake_imgs_list, dim=0)
            
            fid_val = evaluator.compute_fid(real_tensor, fake_tensor)
            is_mean, _ = evaluator.compute_is(fake_tensor)
            
            print(f"Epoch [{epoch}/{epochs}] Loss D: {avg_lossD:.4f} | Loss G: {avg_lossG:.4f} | FID: {fid_val:.2f} | IS: {is_mean:.2f}")

            if fid_val < best_fid:
                best_fid = fid_val
                torch.save(netG.state_dict(), "results_gen/best_dcgan_g.pth")
                torch.save(netD.state_dict(), "results_gen/best_dcgan_d.pth")
                print(" -> Melhor DCGAN guardada (Novo melhor FID)!")
            netG.train()
        else:
            print(f"Epoch [{epoch}/{epochs}] Loss D: {avg_lossD:.4f} | Loss G: {avg_lossG:.4f}")

        # Registar métricas
        with open(csv_path, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch, avg_lossD, avg_lossG, fid_val if fid_val else '', is_mean if is_mean else ''])

if __name__ == "__main__":
    train_dcgan()
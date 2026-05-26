import os
import csv
import torch
import torch.optim as optim
from dataset import get_dataloaders
from wgan_gp import Generator, Critic, weights_init # Ajusta os imports se necessário
from utils import GenerativeEvaluator

def compute_gradient_penalty(critic, real_samples, fake_samples, labels, device):
    alpha = torch.rand((real_samples.size(0), 1, 1, 1), device=device)
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)

    d_interpolates = critic(interpolates, labels)
    fake = torch.ones_like(d_interpolates, device=device)

    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

def train_wgangp():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"A treinar WGAN-GP no dispositivo: {device}")

    # Configurações
    batch_size = 32
    epochs = 200
    lr = 1e-4
    z_dim = 100
    n_critic = 5
    lambda_gp = 10
    eval_every = 5
    os.makedirs("results_gen", exist_ok=True)
    csv_path = "results_gen/wgangp_metrics.csv"

    with open(csv_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Epoch', 'Loss_C', 'Loss_G', 'FID', 'IS_mean'])

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

    netG = Generator(inputDim=z_dim, num_classes=num_classes).to(device)
    netC = Critic(num_classes=num_classes).to(device)
    netG.apply(weights_init)
    netC.apply(weights_init)

    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(0.0, 0.9))
    optimizerC = optim.Adam(netC.parameters(), lr=lr, betas=(0.0, 0.9))
    evaluator = GenerativeEvaluator(device=device)

    best_fid = float('inf')

    for epoch in range(1, epochs + 1):
        running_lossC, running_lossG = 0.0, 0.0
        batches_G = 0
        
        for i, (real_imgs, labels) in enumerate(train_loader):
            real_imgs = real_imgs.to(device)
            labels = labels.to(device)
            real_imgs_scaled = real_imgs * 2.0 - 1.0
            cur_batch_size = real_imgs.size(0)

            # --- Treinar Critic ---
            for _ in range(n_critic):
                optimizerC.zero_grad()
                z = torch.randn(cur_batch_size, z_dim, 1, 1, device=device)
                fake_imgs = netG(z, labels)

                critic_real = netC(real_imgs_scaled, labels).view(-1)
                critic_fake = netC(fake_imgs.detach(), labels).view(-1)

                gp = compute_gradient_penalty(netC, real_imgs_scaled, fake_imgs.detach(), labels, device)
                loss_C = critic_fake.mean() - critic_real.mean() + lambda_gp * gp

                loss_C.backward()
                optimizerC.step()
                running_lossC += loss_C.item()

            # --- Treinar Gerador ---
            optimizerG.zero_grad()
            z = torch.randn(cur_batch_size, z_dim, 1, 1, device=device)
            fake_imgs = netG(z, labels)

            critic_fake = netC(fake_imgs, labels).view(-1)
            loss_G = -critic_fake.mean()

            loss_G.backward()
            optimizerG.step()
            running_lossG += loss_G.item()
            batches_G += 1

        avg_lossC = running_lossC / (len(train_loader) * n_critic)
        avg_lossG = running_lossG / batches_G

        fid_val, is_mean = None, None

        # --- Avaliação Generativa ---
        if epoch % eval_every == 0 or epoch == epochs:
            print("A calcular métricas generativas (FID e IS)...")
            netG.eval()
            real_imgs_list, fake_imgs_list = [], []
            
            with torch.no_grad():
                for val_imgs, val_labels in val_loader:
                    if len(real_imgs_list) * batch_size >= 500:
                        break
                    real_imgs_list.append(val_imgs)
                    
                    z = torch.randn(val_imgs.size(0), z_dim, 1, 1, device=device)
                    val_labels = val_labels.to(device)
                    fakes = netG(z, val_labels)
                    
                    fakes = (fakes + 1.0) / 2.0
                    fakes = torch.clamp(fakes, 0.0, 1.0)
                    fake_imgs_list.append(fakes.cpu())
                    
            real_tensor = torch.cat(real_imgs_list, dim=0)
            fake_tensor = torch.cat(fake_imgs_list, dim=0)
            
            fid_val = evaluator.compute_fid(real_tensor, fake_tensor)
            is_mean, _ = evaluator.compute_is(fake_tensor)
            
            print(f"Epoch [{epoch}/{epochs}] Loss C: {avg_lossC:.4f} | Loss G: {avg_lossG:.4f} | FID: {fid_val:.2f} | IS: {is_mean:.2f}")

            if fid_val < best_fid:
                best_fid = fid_val
                torch.save(netG.state_dict(), "results_gen/best_wgangp_g.pth")
                torch.save(netC.state_dict(), "results_gen/best_wgangp_c.pth")
                print(" -> Melhor WGAN-GP guardada (Novo melhor FID)!")
            netG.train()
        else:
            print(f"Epoch [{epoch}/{epochs}] Loss C: {avg_lossC:.4f} | Loss G: {avg_lossG:.4f}")

        with open(csv_path, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch, avg_lossC, avg_lossG, fid_val if fid_val else '', is_mean if is_mean else ''])

if __name__ == "__main__":
    train_wgangp()
import argparse
import os

import torch
import torch.optim as optim

from torchvision.utils import save_image
from dataset import get_dataloaders
from models.wgan_gp import Generator, Critic, weights_init
from utils import GenerativeEvaluator, append_generative_metrics_to_csv


def compute_gradient_penalty(critic, real_samples, fake_samples, labels, device):
    alpha = torch.rand(
        (real_samples.size(0), 1, 1, 1),
        device=device
    )

    interpolates = (
        alpha * real_samples + ((1 - alpha) * fake_samples)
    ).requires_grad_(True)

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

    gradient_penalty = (
        (gradients.norm(2, dim=1) - 1) ** 2
    ).mean()

    return gradient_penalty


def parse_args():
    parser = argparse.ArgumentParser(description="Train Conditional WGAN-GP")

    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--z_dim", type=int, default=100)
    parser.add_argument("--n_critic", type=int, default=5)
    parser.add_argument("--lambda_gp", type=float, default=10)
    parser.add_argument("--eval_every", type=int, default=2)

    parser.add_argument("--save_dir", default="results_gen/wgangp")
    parser.add_argument("--experiment_name", default="WGAN-GP n_critic 5")

    return parser.parse_args()


def train_wgangp():
    args = parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs("results", exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {device}")
    print(f"Experiment: {args.experiment_name}")

    train_loader, val_loader, _, _ = get_dataloaders(
        csv_path="data/train.csv",
        img_dir="data/train",
        batch_size=args.batch_size,
        img_size=64,
        val_size=0.15,
        test_size=0.15,
        augment=True,
        normalize=False,
        num_workers=0,
    )

    num_classes = len(train_loader.dataset.classes)

    netG = Generator(
        inputDim=args.z_dim,
        num_classes=num_classes
    ).to(device)

    netC = Critic(
        num_classes=num_classes
    ).to(device)

    netG.apply(weights_init)
    netC.apply(weights_init)

    optimizerG = optim.Adam(
        netG.parameters(),
        lr=args.lr,
        betas=(0.0, 0.9)
    )

    optimizerC = optim.Adam(
        netC.parameters(),
        lr=args.lr,
        betas=(0.0, 0.9)
    )

    evaluator = GenerativeEvaluator(device=device)

    best_fid = float("inf")
    best_is = ""
    best_loss_g = ""
    best_loss_c = ""

    epochs_no_improve = 0
    patience = 20

    for epoch in range(1, args.epochs + 1):
        netG.train()
        netC.train()

        running_lossC = 0.0
        running_lossG = 0.0
        batches_G = 0

        for real_imgs, labels in train_loader:
            real_imgs = real_imgs.to(device)
            labels = labels.to(device)

            real_imgs_scaled = real_imgs * 2.0 - 1.0
            cur_batch_size = real_imgs.size(0)

            # Train Critic multiple times
            for _ in range(args.n_critic):
                optimizerC.zero_grad()

                z = torch.randn(
                    cur_batch_size,
                    args.z_dim,
                    1,
                    1,
                    device=device
                )

                fake_imgs = netG(z, labels)

                critic_real = netC(real_imgs_scaled, labels).view(-1)
                critic_fake = netC(fake_imgs.detach(), labels).view(-1)

                gp = compute_gradient_penalty(
                    critic=netC,
                    real_samples=real_imgs_scaled,
                    fake_samples=fake_imgs.detach(),
                    labels=labels,
                    device=device,
                )

                loss_C = (
                    critic_fake.mean()
                    - critic_real.mean()
                    + args.lambda_gp * gp
                )

                loss_C.backward()
                optimizerC.step()

                running_lossC += loss_C.item()

            # Train Generator
            optimizerG.zero_grad()

            z = torch.randn(
                cur_batch_size,
                args.z_dim,
                1,
                1,
                device=device
            )

            fake_imgs = netG(z, labels)
            critic_fake = netC(fake_imgs, labels).view(-1)

            loss_G = -critic_fake.mean()

            loss_G.backward()
            optimizerG.step()

            running_lossG += loss_G.item()
            batches_G += 1

        avg_lossC = running_lossC / (len(train_loader) * args.n_critic)
        avg_lossG = running_lossG / batches_G

        if epoch % args.eval_every == 0 or epoch == args.epochs:
            netG.eval()

            real_imgs_list = []
            fake_imgs_list = []

            with torch.no_grad():
                for val_imgs, val_labels in val_loader:
                    real_imgs_list.append(val_imgs)

                    val_labels = val_labels.to(device)

                    z = torch.randn(
                        val_imgs.size(0),
                        args.z_dim,
                        1,
                        1,
                        device=device
                    )

                    fakes = netG(z, val_labels)
                    fakes = (fakes + 1.0) / 2.0
                    fakes = torch.clamp(fakes, 0.0, 1.0)

                    fake_imgs_list.append(fakes.cpu())

            real_tensor = torch.cat(real_imgs_list, dim=0)
            fake_tensor = torch.cat(fake_imgs_list, dim=0)

            grid_path = os.path.join(args.save_dir, f"grid_epoch_{epoch}.png")
            save_image(fake_tensor[:32], grid_path, nrow=8)

            fid_val = evaluator.compute_fid(real_tensor, fake_tensor)
            is_mean, _ = evaluator.compute_is(fake_tensor)

            print(
                f"Epoch [{epoch}/{args.epochs}] "
                f"Loss C: {avg_lossC:.4f} | "
                f"Loss G: {avg_lossG:.4f} | "
                f"FID: {fid_val:.4f} | "
                f"IS: {is_mean:.4f}"
            )

            if fid_val < best_fid:
                best_fid = fid_val
                best_is = is_mean
                best_loss_c = avg_lossC
                best_loss_g = avg_lossG

                epochs_no_improve = 0

                torch.save(
                    netG.state_dict(),
                    os.path.join(args.save_dir, "best_wgangp_g.pth")
                )

                torch.save(
                    netC.state_dict(),
                    os.path.join(args.save_dir, "best_wgangp_c.pth")
                )

                print(" -> New best model saved.")

            else:
                epochs_no_improve += 1
                print(f" -> No improvement for {epochs_no_improve} eval(s).")
                
                if epochs_no_improve >= patience and epoch > 100:
                    print(f"Early stopping triggered! Model stopped improving for {patience * args.eval_every} epochs.")
                    break

        else:
            print(
                f"Epoch [{epoch}/{args.epochs}] "
                f"Loss C: {avg_lossC:.4f} | "
                f"Loss G: {avg_lossG:.4f}"
            )

    append_generative_metrics_to_csv(
        experiment_name=args.experiment_name,
        epochs=args.epochs,
        fid=best_fid,
        inception_score=best_is,
        loss_g=best_loss_g,
        loss_c=best_loss_c,
        save_path="results_gen/generative_experiments.csv",
    )


if __name__ == "__main__":
    train_wgangp()
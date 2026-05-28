import argparse
import os
import torch
import torch.optim as optim

from torchvision.utils import save_image
from dataset import get_dataloaders
from models.vae import ConvVAE, vae_loss
from utils import GenerativeEvaluator, append_generative_metrics_to_csv


def parse_args():
    parser = argparse.ArgumentParser(description="Train Conditional VAE")
    parser.add_argument("--latent_dim", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--eval_every", type=int, default=5) 
    parser.add_argument("--save_dir", default="results_gen/vae")
    parser.add_argument("--experiment_name", default="VAE latent 128")
    return parser.parse_args()


def train_vae():
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

    model = ConvVAE(
        latent_dim=args.latent_dim,
        num_classes=num_classes
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    evaluator = GenerativeEvaluator(device=device)

    best_fid = float("inf")
    best_is = ""
    best_ssim = ""
    best_train_loss = ""
    best_val_loss = ""

    epochs_no_improve = 0
    patience = 3 

    for epoch in range(1, args.epochs + 1):
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

        model.eval()
        running_val_loss = 0.0
        real_imgs_list = []
        recon_imgs_list = []

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)

                recon_images, mu, logvar = model(images, labels)
                loss = vae_loss(recon_images, images, mu, logvar)

                running_val_loss += loss.item()

                real_imgs_list.append(images.cpu())
                recon_imgs_list.append(recon_images.cpu())

        avg_val_loss = running_val_loss / len(val_loader)

        # Only for readable printing, because VAE loss is naturally large
        print_train_loss = avg_train_loss / (3 * 64 * 64)
        print_val_loss = avg_val_loss / (3 * 64 * 64)

        if epoch % args.eval_every == 0 or epoch == args.epochs:
            real_tensor = torch.cat(real_imgs_list, dim=0)
            recon_tensor = torch.cat(recon_imgs_list, dim=0)

            with torch.no_grad():
                z = torch.randn(real_tensor.size(0), args.latent_dim, device=device)
                random_labels = torch.randint(
                    0,
                    num_classes,
                    (real_tensor.size(0),),
                    device=device
                )

                generated_imgs = model.decode(z, random_labels).cpu()

            ssim_val = evaluator.compute_ssim(
                recon_tensor,
                real_tensor,
                is_tanh=False
            )

            grid_path = os.path.join(args.save_dir, f"grid_epoch_{epoch}.png")
            save_image(generated_imgs[:32], grid_path, nrow=8)
            
            fid_val = evaluator.compute_fid(real_tensor, generated_imgs)
            is_mean, _ = evaluator.compute_is(generated_imgs)

            print(
                f"Epoch [{epoch}/{args.epochs}] "
                f"Train Loss: {print_train_loss:.4f} | "
                f"Val Loss: {print_val_loss:.4f} | "
                f"FID: {fid_val:.4f} | "
                f"IS: {is_mean:.4f} | "
                f"SSIM: {ssim_val:.4f}"
            )

            if fid_val < best_fid:
                best_fid = fid_val
                best_is = is_mean
                best_ssim = ssim_val
                best_train_loss = print_train_loss
                best_val_loss = print_val_loss
                epochs_no_improve = 0

                torch.save(
                    model.state_dict(),
                    os.path.join(args.save_dir, "best_vae.pth")
                )

                print(" -> New best model saved.")
            else:
                epochs_no_improve += 1
                print(f" -> No improvement for {epochs_no_improve} eval(s).")
                
                if epochs_no_improve >= patience:
                    print(f"Early stopping triggered! Model stopped improving for {patience * args.eval_every} epochs.")
                    break

        else:
            print(
                f"Epoch [{epoch}/{args.epochs}] "
                f"Train Loss: {print_train_loss:.4f} | "
                f"Val Loss: {print_val_loss:.4f}"
            )

    append_generative_metrics_to_csv(
        experiment_name=args.experiment_name,
        epochs=args.epochs,
        fid=best_fid,
        inception_score=best_is,
        ssim=best_ssim,
        train_loss=best_train_loss,
        val_loss=best_val_loss,
        save_path="results_gen/generative_experiments.csv",
    )


if __name__ == "__main__":
    train_vae()
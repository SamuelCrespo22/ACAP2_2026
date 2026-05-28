import argparse
import os

import torch
import torch.optim as optim

from torchvision.utils import save_image
from dataset import get_dataloaders
from models.diffusion import UNet64, DDPM
from utils import GenerativeEvaluator, append_generative_metrics_to_csv


def parse_args():
    parser = argparse.ArgumentParser(description="Train Conditional Diffusion Model")

    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--eval_every", type=int, default=10)

    parser.add_argument("--T", type=int, default=1000)
    parser.add_argument("--schedule", choices=["linear", "cosine"], default="linear")

    parser.add_argument("--save_dir", default="results_gen/diffusion")
    parser.add_argument("--experiment_name", default="Diffusion linear")

    return parser.parse_args()


def train_diffusion():
    args = parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs("results_gen", exist_ok=True)

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
    print(f"Number of classes: {num_classes}")

    model = UNet64(
        time_dim=128,
        num_classes=num_classes
    ).to(device)

    ddpm = DDPM(
        model=model,
        T=args.T,
        device=device,
        schedule=args.schedule
    )

    optimizer = optim.Adam(
        ddpm.model.parameters(),
        lr=args.learning_rate
    )

    evaluator = GenerativeEvaluator(device=device)

    best_fid = float("inf")
    best_is = ""
    best_train_loss = ""

    epochs_no_improve = 0
    patience = 4

    for epoch in range(1, args.epochs + 1):
        ddpm.model.train()
        running_train_loss = 0.0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            images = images * 2.0 - 1.0

            optimizer.zero_grad()

            loss = ddpm.loss(images, labels)

            loss.backward()
            optimizer.step()

            running_train_loss += loss.item()

        avg_train_loss = running_train_loss / len(train_loader)

        if epoch % args.eval_every == 0 or epoch == args.epochs:
            ddpm.model.eval()

            real_imgs_list = []
            fake_imgs_list = []

            with torch.no_grad():
                print(f"A iniciar amostragem (inferência) de avaliação. Isto pode demorar alguns minutos...")
                
                for val_imgs, val_labels in val_loader:
                    real_imgs_list.append(val_imgs)

                    labels = val_labels.to(device)

                    fakes = ddpm.sample(
                        n=val_imgs.size(0),
                        labels=labels
                    )

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
                f"Train Loss: {avg_train_loss:.4f} | "
                f"FID: {fid_val:.4f} | "
                f"IS: {is_mean:.4f}"
            )

            if fid_val < best_fid:
                best_fid = fid_val
                best_is = is_mean
                best_train_loss = avg_train_loss
                epochs_no_improve = 0

                torch.save(
                    ddpm.model.state_dict(),
                    os.path.join(args.save_dir, "best_diffusion.pth")
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
                f"Train Loss: {avg_train_loss:.4f}"
            )

    append_generative_metrics_to_csv(
        experiment_name=args.experiment_name,
        epochs=args.epochs,
        fid=best_fid,
        inception_score=best_is,
        train_loss=best_train_loss,
        save_path="results_gen/generative_experiments.csv",
    )


if __name__ == "__main__":
    train_diffusion()
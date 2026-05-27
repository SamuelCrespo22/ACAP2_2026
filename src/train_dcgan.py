import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim

from dataset import get_dataloaders
from models.dcgan import Generator, Discriminator, weights_init
from utils import GenerativeEvaluator, append_generative_metrics_to_csv


def parse_args():
    parser = argparse.ArgumentParser(description="Train Conditional DCGAN")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr_g", type=float, default=2e-4)
    parser.add_argument("--lr_d", type=float, default=2e-4)
    parser.add_argument("--beta1", type=float, default=0.5)
    parser.add_argument("--z_dim", type=int, default=100)
    parser.add_argument("--eval_every", type=int, default=1)
    parser.add_argument("--save_dir", default="results_gen/dcgan")
    parser.add_argument("--experiment_name", default="DCGAN lr 2e-4")
    return parser.parse_args()


def train_dcgan():
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
        augment=False,
        normalize=False,
        num_workers=0,
    )

    num_classes = len(train_loader.dataset.classes)

    netG = Generator(
        inputDim=args.z_dim,
        num_classes=num_classes
    ).to(device)

    netD = Discriminator(
        num_classes=num_classes
    ).to(device)

    netG.apply(weights_init)
    netD.apply(weights_init)

    criterion = nn.BCELoss()

    optimizerG = optim.Adam(
        netG.parameters(),
        lr=args.lr_g,
        betas=(args.beta1, 0.999)
    )

    optimizerD = optim.Adam(
        netD.parameters(),
        lr=args.lr_d,
        betas=(args.beta1, 0.999)
    )

    evaluator = GenerativeEvaluator(device=device)

    best_fid = float("inf")
    best_is = ""
    best_loss_g = ""
    best_loss_d = ""

    epochs_no_improve = 0

    real_label = 1.0
    fake_label = 0.0

    for epoch in range(1, args.epochs + 1):
        netG.train()
        netD.train()

        running_lossD = 0.0
        running_lossG = 0.0

        for real_imgs, labels in train_loader:
            real_imgs = real_imgs.to(device)
            labels = labels.to(device)

            real_imgs_scaled = real_imgs * 2.0 - 1.0
            cur_batch_size = real_imgs.size(0)

            # Train Discriminator
            netD.zero_grad()

            label_real = torch.full(
                (cur_batch_size,),
                real_label,
                device=device
            )

            output_real = netD(real_imgs_scaled, labels).view(-1)
            lossD_real = criterion(output_real, label_real)

            z = torch.randn(
                cur_batch_size,
                args.z_dim,
                1,
                1,
                device=device
            )

            fake_imgs = netG(z, labels)

            label_fake = torch.full(
                (cur_batch_size,),
                fake_label,
                device=device
            )

            output_fake = netD(fake_imgs.detach(), labels).view(-1)
            lossD_fake = criterion(output_fake, label_fake)

            lossD = lossD_real + lossD_fake
            lossD.backward()
            optimizerD.step()

            running_lossD += lossD.item()

            # Train Generator
            netG.zero_grad()

            label_gen = torch.full(
                (cur_batch_size,),
                real_label,
                device=device
            )

            output = netD(fake_imgs, labels).view(-1)
            lossG = criterion(output, label_gen)

            lossG.backward()
            optimizerG.step()

            running_lossG += lossG.item()

        avg_lossD = running_lossD / len(train_loader)
        avg_lossG = running_lossG / len(train_loader)

        if epoch % args.eval_every == 0 or epoch == args.epochs:
            netG.eval()

            real_imgs_list = []
            fake_imgs_list = []

            with torch.no_grad():
                for val_imgs, val_labels in val_loader:
                    if len(real_imgs_list) * args.batch_size >= 500:
                        break

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

            fid_val = evaluator.compute_fid(real_tensor, fake_tensor)
            is_mean, _ = evaluator.compute_is(fake_tensor)

            print(
                f"Epoch [{epoch}/{args.epochs}] "
                f"Loss D: {avg_lossD:.4f} | "
                f"Loss G: {avg_lossG:.4f} | "
                f"FID: {fid_val:.4f} | "
                f"IS: {is_mean:.4f}"
            )

            if fid_val < best_fid:
                best_fid = fid_val
                best_is = is_mean
                best_loss_d = avg_lossD
                best_loss_g = avg_lossG
                epochs_no_improve = 0

                torch.save(
                    netG.state_dict(),
                    os.path.join(args.save_dir, "best_dcgan_g.pth")
                )

                torch.save(
                    netD.state_dict(),
                    os.path.join(args.save_dir, "best_dcgan_d.pth")
                )

                print(" -> New best model saved.")
            else:
                epochs_no_improve += 1
                print(f" -> No improvement for {epochs_no_improve} eval(s).")

        else:
            print(
                f"Epoch [{epoch}/{args.epochs}] "
                f"Loss D: {avg_lossD:.4f} | "
                f"Loss G: {avg_lossG:.4f}"
            )

    append_generative_metrics_to_csv(
        experiment_name=args.experiment_name,
        epochs=args.epochs,
        fid=best_fid,
        inception_score=best_is,
        loss_g=best_loss_g,
        loss_d=best_loss_d,
        save_path="results_gen/generative_experiments.csv",
    )


if __name__ == "__main__":
    train_dcgan()
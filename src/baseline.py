import argparse
import os
import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score

from dataset import get_dataloaders
from utils import (
    plot_training_curves,
    evaluate_model,
    append_metrics_to_csv,
    save_confusion_matrix,
)

from models.base_cnn import BaselineCNN


def train_one_epoch(model, train_loader, criterion, optimizer, device):
    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

        preds = torch.argmax(outputs, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return running_loss / total, correct / total


def validate(model, val_loader, criterion, device):
    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0
    
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)

            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    val_loss = running_loss / total
    val_acc = correct / total
    
    val_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)

    return val_loss, val_acc, val_f1


def parse_args():
    parser = argparse.ArgumentParser(description="Train Baseline CNN (Fixed Data, Multiple Weight Seeds).")

    parser.add_argument("--csv_path", default="data/train.csv")
    parser.add_argument("--img_dir", default="data/train")
    parser.add_argument("--aug_csv_path", default=None, help="Path to generated images CSV")
    parser.add_argument("--aug_img_dir", default=None, help="Path to generated images directory")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--img_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--learning_rate", type=float, default=0.001)

    parser.add_argument("--val_size", type=float, default=0.15)
    parser.add_argument("--test_size", type=float, default=0.15)

    parser.add_argument("--save_dir", default="results/baseline_original")
    parser.add_argument("--experiment_name", default="Baseline Original")
    parser.add_argument("--num_workers", type=int, default=0)

    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def main():
    args = parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs("results", exist_ok=True)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        try:
            import torch_directml
            if torch_directml.is_available():
                device = torch_directml.device()
            else:
                device = torch.device("cpu")
        except ImportError:
            device = torch.device("cpu")
    print(f"Using device: {device}")

    seeds = [42, 43, 44, 45, 46]

    for seed in seeds:
        print(f"\n{'='*50}\nStarting Training - CNN Initialization Seed: {seed}\n{'='*50}")
        set_seed(seed)

        train_loader, val_loader, test_loader, classes = get_dataloaders(
            csv_path=args.csv_path,
            img_dir=args.img_dir,
            batch_size=args.batch_size,
            img_size=args.img_size,
            val_size=args.val_size,
            test_size=args.test_size,
            augment=False,
            normalize=True,
            aug_csv_path=args.aug_csv_path,
            aug_img_dir=args.aug_img_dir,
            num_workers=args.num_workers,
        )

        num_classes = len(classes)
        
        current_experiment_name = f"{args.experiment_name}_FixedData_NetSeed_{seed}"
        current_save_dir = os.path.join(args.save_dir, f"NetSeed_{seed}")
        os.makedirs(current_save_dir, exist_ok=True)

        model = BaselineCNN(num_classes=num_classes).to(device)
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, foreach=False)
        criterion = nn.CrossEntropyLoss()

        train_losses = []
        val_losses = []
        train_accs = []
        val_accs = []

        best_val_f1 = 0.0
        best_model_path = os.path.join(current_save_dir, "baseline_best.pth")

        patience = 15
        epochs_no_improve = 0

        for epoch in range(args.epochs):
            train_loss, train_acc = train_one_epoch(
                model=model,
                train_loader=train_loader,
                criterion=criterion,
                optimizer=optimizer,
                device=device,
            )

            val_loss, val_acc, val_f1 = validate(
                model=model,
                val_loader=val_loader,
                criterion=criterion,
                device=device,
            )

            train_losses.append(train_loss)
            val_losses.append(val_loss)
            train_accs.append(train_acc)
            val_accs.append(val_acc)

            print(
                f"Epoch [{epoch + 1}/{args.epochs}] "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Val Acc: {val_acc:.4f} | "
                f"Val F1: {val_f1:.4f}"
            )

            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                epochs_no_improve = 0

                torch.save(
                    {
                        "epoch": epoch + 1,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "val_loss": val_loss,
                        "val_acc": val_acc,
                        "val_f1": val_f1,
                        "classes": classes,
                        "experiment_name": current_experiment_name,
                    },
                    best_model_path,
                )
                print(f" -> New best model saved (F1: {val_f1:.4f}).")
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print(f"\nEarly stopping triggered at epoch {epoch + 1}")
                    break

        print(f"\nTraining finished for Network Seed {seed}.")

        plot_training_curves(
            train_losses=train_losses,
            val_losses=val_losses,
            train_accs=train_accs,
            val_accs=val_accs,
            save_path=os.path.join(current_save_dir, "training_curves.png"),
        )

        checkpoint = torch.load(best_model_path, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])

        val_metrics, val_report, y_true_val, y_pred_val = evaluate_model(
            model=model,
            dataloader=val_loader,
            device=device,
            class_names=classes,
        )

        append_metrics_to_csv(
            metrics=val_metrics,
            epochs=len(train_losses),
            experiment_name=current_experiment_name,
            best_val_acc=checkpoint.get('val_acc', 0.0),
            save_path="results/all_experiments_val.csv",
        )

        save_confusion_matrix(
            y_true=y_true_val,
            y_pred=y_pred_val,
            class_names=classes,
            save_path=os.path.join(current_save_dir, "confusion_matrix_validation.png"),
        )

if __name__ == "__main__":
    main()
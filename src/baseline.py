import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim

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

        optimizer.zero_grad()

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

    return running_loss / total, correct / total


def parse_args():
    parser = argparse.ArgumentParser(description="Treina a baseline CNN com validação e teste internos.")
    parser.add_argument("--csv_path", default="data/train.csv")
    parser.add_argument("--img_dir", default="data/train")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--img_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--val_size", type=float, default=0.2)
    parser.add_argument("--test_size", type=float, default=0.15)
    parser.add_argument("--save_dir", default="results")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, val_loader, test_loader, classes = get_dataloaders(
        csv_path=args.csv_path,
        img_dir=args.img_dir,
        batch_size=args.batch_size,
        img_size=args.img_size,
        val_size=args.val_size,
        test_size=args.test_size,
        augment=False,
        normalize=True,
    )

    if val_loader is None:
        raise ValueError("val_size must be > 0 to perform baseline validation.")

    num_classes = len(classes)
    print(f"Number of classes: {num_classes}")

    model = BaselineCNN(num_classes=num_classes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.CrossEntropyLoss()

    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    best_val_loss = float('inf')
    best_val_acc_at_early_stop = 0.0
    best_model_path = os.path.join(args.save_dir, "baseline_best.pth")

    patience = 7
    epochs_no_improve = 0

    for epoch in range(args.epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        print(
            f"Epoch [{epoch + 1}/{args.epochs}] "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc_at_early_stop = val_acc
            epochs_no_improve = 0

            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "classes": classes,
                },
                best_model_path,
            )
            print(" -> Nova melhor Val Loss! Modelo guardado.")
        else:
            epochs_no_improve += 1
            print(f" -> Sem melhoria há {epochs_no_improve} época(s).")
            if epochs_no_improve >= patience:
                print(f"\n[!] Early Stopping ativado na época {epoch + 1}!")
                break

    print("\nTreino terminado.")

    plot_training_curves(
        train_losses,
        val_losses,
        train_accs,
        val_accs,
        save_path=os.path.join(args.save_dir, "baseline_training_curves.png"),
    )

    checkpoint = torch.load(best_model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    val_metrics, val_report, y_true_val, y_pred_val = evaluate_model(
        model=model,
        dataloader=val_loader,
        device=device,
        class_names=classes,
    )
    print("\nValidation report:\n", val_report)

    append_metrics_to_csv(
        metrics=val_metrics,
        epochs=len(train_losses),
        split="validation",
        best_val_acc=best_val_acc_at_early_stop,
        save_path=os.path.join(args.save_dir, "metrics_history.csv"),
    )
    save_confusion_matrix(
        y_true=y_true_val,
        y_pred=y_pred_val,
        class_names=classes,
        save_path=os.path.join(args.save_dir, "baseline_confusion_matrix_validation.png"),
    )

    if test_loader is not None:
        test_metrics, test_report, y_true_test, y_pred_test = evaluate_model(
            model=model,
            dataloader=test_loader,
            device=device,
            class_names=classes,
        )
        print("\nTest report:\n", test_report)

        append_metrics_to_csv(
            metrics=test_metrics,
            epochs=len(train_losses),
            split="test",
            best_val_acc=best_val_acc_at_early_stop,
            save_path=os.path.join(args.save_dir, "metrics_history.csv"),
        )
        save_confusion_matrix(
            y_true=y_true_test,
            y_pred=y_pred_test,
            class_names=classes,
            save_path=os.path.join(args.save_dir, "baseline_confusion_matrix_test.png"),
        )

    print("Baseline evaluation completed.")


if __name__ == "__main__":
    main()
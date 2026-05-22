import os
import torch
import torch.nn as nn
import torch.optim as optim

from dataset import get_dataloaders
from utils import (
    plot_training_curves,
    evaluate_model,
    save_metrics_txt,
    append_metrics_to_csv,
    save_confusion_matrix,
)


class BaselineCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        def conv_block(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2)
            )

        self.features = nn.Sequential(
            conv_block(3, 64),
            conv_block(64, 128),
            conv_block(128, 256),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.classifier = nn.Sequential(
            *[
                layer
                for size in [256, 512, 1024]
                for layer in (
                    nn.Linear(size, size * 2),
                    nn.ReLU(inplace=True),
                    nn.Dropout(p=0.5)
                )
            ],
            nn.Linear(2048, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


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


def main():
    csv_path = "data/train.csv"
    img_dir = "data/train"

    batch_size = 32
    img_size = 224
    epochs = 50 #no notebook está com 20
    learning_rate = 0.001

    os.makedirs("results", exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, val_loader, classes = get_dataloaders(
        csv_path=csv_path,
        img_dir=img_dir,
        batch_size=batch_size,
        img_size=img_size,
        test_size=0.2
    )

    num_classes = len(classes)
    print(f"Number of classes: {num_classes}")

    model = BaselineCNN(num_classes=num_classes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    best_val_acc = 0.0
    best_model_path = "baseline_best.pth"

    for epoch in range(epochs):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )

        val_loss, val_acc = validate(
            model, val_loader, criterion, device
        )

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        print(
            f"Epoch [{epoch + 1}/{epochs}] "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc

            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_acc": val_acc,
                    "classes": classes,
                    "train_losses": train_losses,
                    "val_losses": val_losses,
                    "train_accs": train_accs,
                    "val_accs": val_accs,
                },
                best_model_path
            )

            print(f"Saved best model with Val Acc: {best_val_acc:.4f}")

    print("Training finished.")
    print(f"Best validation accuracy: {best_val_acc:.4f}")

    plot_training_curves(
        train_losses,
        val_losses,
        train_accs,
        val_accs,
        save_path="results/baseline_training_curves.png"
    )

    checkpoint = torch.load(best_model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    metrics, report, y_true, y_pred = evaluate_model(
        model=model,
        dataloader=val_loader,
        device=device,
        class_names=classes
    )

    append_metrics_to_csv(
        metrics=metrics,
        epochs=epochs,
        best_val_acc=best_val_acc,
        save_path="results/metrics_history.csv"
    )

    save_confusion_matrix(
        y_true=y_true,
        y_pred=y_pred,
        class_names=classes,
        save_path="results/baseline_confusion_matrix.png"
    )

    print("Baseline evaluation completed.")


if __name__ == "__main__":
    main()
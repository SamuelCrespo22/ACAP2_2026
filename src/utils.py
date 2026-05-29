import os
import csv

import numpy as np
import torch
import matplotlib.pyplot as plt
import scipy.linalg
import torch.nn.functional as F
from torchvision.models import inception_v3, Inception_V3_Weights
from skimage.metrics import structural_similarity

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)


def plot_training_curves(train_losses, val_losses, train_accs, val_accs, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label="Train Loss")
    plt.plot(epochs, val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Over Epochs")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accs, label="Train Accuracy")
    plt.plot(epochs, val_accs, label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy Over Epochs")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"Saved training curves to {save_path}")


def top_k_accuracy(y_true, y_prob, k=5):
    top_k_preds = np.argsort(y_prob, axis=1)[:, -k:]
    correct = np.any(top_k_preds == y_true[:, np.newaxis], axis=1).sum()
    return correct / len(y_true)


def evaluate_model(model, dataloader, device, class_names):
    model.eval()

    y_true = []
    y_pred = []
    y_prob = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)

            y_true.append(labels.cpu().numpy())
            y_pred.append(preds.cpu().numpy())
            y_prob.append(probs.cpu().numpy())

    y_true = np.concatenate(y_true, axis=0)
    y_pred = np.concatenate(y_pred, axis=0)
    y_prob = np.concatenate(y_prob, axis=0)

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "weighted_precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "weighted_recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "top5_accuracy": top_k_accuracy(y_true, y_prob, k=5),
    }

    report = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        zero_division=0
    )

    return metrics, report, y_true, y_pred


def append_metrics_to_csv(
    metrics,
    epochs,
    experiment_name,
    best_val_acc=None,
    save_path="results/all_experiments.csv"
):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    file_exists = os.path.isfile(save_path)

    row = {
        "experiment_name": experiment_name,
        "epochs": epochs,
        "best_val_acc": best_val_acc if best_val_acc is not None else "",
        "accuracy": metrics["accuracy"],
        "macro_precision": metrics["macro_precision"],
        "macro_recall": metrics["macro_recall"],
        "macro_f1": metrics["macro_f1"],
        "weighted_precision": metrics["weighted_precision"],
        "weighted_recall": metrics["weighted_recall"],
        "weighted_f1": metrics["weighted_f1"],
        "top5_accuracy": metrics["top5_accuracy"],
    }

    with open(save_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())

        if not file_exists:
            writer.writeheader()

        writer.writerow(row)

    print(f"Appended metrics to {save_path}")

    
def append_generative_metrics_to_csv(
    experiment_name,
    epochs,
    save_path="results/generative_experiments.csv",
    fid="",
    inception_score="",
    ssim="",
    train_loss="",
    val_loss="",
    loss_g="",
    loss_d="",
    loss_c="",
):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    file_exists = os.path.isfile(save_path)

    row = {
        "experiment_name": experiment_name,
        "epochs": epochs,
        "fid": fid,
        "inception_score": inception_score,
        "ssim": ssim,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "loss_g": loss_g,
        "loss_d": loss_d,
        "loss_c": loss_c,
    }

    with open(save_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())

        if not file_exists:
            writer.writeheader()

        writer.writerow(row)

    print(f"Appended generative metrics to {save_path}")


def save_confusion_matrix(y_true, y_pred, class_names, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(24, 20))
    plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion Matrix")
    plt.colorbar()

    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=90, fontsize=6)
    plt.yticks(tick_marks, class_names, fontsize=6)

    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()

    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"Saved confusion matrix to {save_path}")


class GenerativeEvaluator:
    def __init__(self, device="cpu"):
        self.device = device
        # Loads InceptionV3 for FID and IS metrics
        self.inception = inception_v3(weights=Inception_V3_Weights.DEFAULT, transform_input=False).to(device)
        self.inception.eval()

    def _get_features_and_probs(self, images, batch_size=32):
        self.inception.eval()
        features_list = []
        probs_list = []
        
        # Normalized with ImageNet statistics - PyTorch official documentation
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(self.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(self.device)
        
        with torch.no_grad():
            for i in range(0, len(images), batch_size):
                batch = images[i:i+batch_size].to(self.device)
                
                # Resize to 299x299 (required by InceptionV3)
                batch = F.interpolate(batch, size=(299, 299), mode='bilinear', align_corners=False)
                batch = (batch - mean) / std
                
                # Save the original fc layer
                fc_backup = self.inception.fc
                
                # Extract Features (temporary removal of the classification layer)
                self.inception.fc = torch.nn.Identity()
                features = self.inception(batch)
                
                # Restore the fc layer and get Probabilities
                self.inception.fc = fc_backup
                logits = self.inception.fc(features)
                probs = F.softmax(logits, dim=1)
                
                features_list.append(features.cpu().numpy())
                probs_list.append(probs.cpu().numpy())
                
        return np.concatenate(features_list, axis=0), np.concatenate(probs_list, axis=0)

    def calculate_frechet_distance(self, mu1, sigma1, mu2, sigma2, eps=1e-6):
        diff = mu1 - mu2
        covmean, _ = scipy.linalg.sqrtm(sigma1.dot(sigma2), disp=False)
        
        if not np.isfinite(covmean).all():
            offset = np.eye(sigma1.shape[0]) * eps
            covmean = scipy.linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
            
        if np.iscomplexobj(covmean):
            covmean = covmean.real
            
        return diff.dot(diff) + np.trace(sigma1 + sigma2 - 2 * covmean)

    def compute_fid(self, real_images, fake_images, batch_size=32):
        """
        Calculates the Frechet Inception Distance (FID).
        Expects two PyTorch tensors [N, C, H, W] in the [0, 1] range.
        """
        real_features, _ = self._get_features_and_probs(real_images, batch_size)
        fake_features, _ = self._get_features_and_probs(fake_images, batch_size)
        
        mu1, sigma1 = np.mean(real_features, axis=0), np.cov(real_features, rowvar=False)
        mu2, sigma2 = np.mean(fake_features, axis=0), np.cov(fake_features, rowvar=False)
        
        return self.calculate_frechet_distance(mu1, sigma1, mu2, sigma2)

    def compute_is(self, fake_images, splits=10, batch_size=32):
        """
        Calculates the Inception Score (IS).
        Expects a PyTorch tensor [N, C, H, W] in the [0, 1] range.
        """
        _, probs = self._get_features_and_probs(fake_images, batch_size)
        
        scores = []
        n_samples = probs.shape[0]
        split_size = max(1, n_samples // splits)
        
        for i in range(splits):
            part = probs[i * split_size : (i + 1) * split_size, :]
            if len(part) == 0:
                continue
            kl = part * (np.log(part + 1e-10) - np.log(np.mean(part, axis=0, keepdims=True) + 1e-10))
            kl = np.mean(np.sum(kl, axis=1))
            scores.append(np.exp(kl))
            
        return np.mean(scores), np.std(scores)

    def compute_ssim(self, recon_x, x, is_tanh=False):
        """
        Calculates the Structural Similarity Index Measure (SSIM) in batches (mean).
        Expects PyTorch tensors [N, C, H, W].
        """
        if is_tanh:
            recon_x = (recon_x + 1) / 2
            x = (x + 1) / 2
            
        recon_x_np = recon_x.detach().cpu().numpy().transpose(0, 2, 3, 1)
        x_np = x.detach().cpu().numpy().transpose(0, 2, 3, 1)
        
        total_ssim = 0.0
        for i in range(x.size(0)):
            val = structural_similarity(x_np[i], recon_x_np[i], data_range=1.0, channel_axis=-1)
            total_ssim += val
            
        return total_ssim / x.size(0)

import argparse
import os
import random
import numpy as np

import torch

from dataset import get_dataloaders
from utils import (
    evaluate_model,
    append_metrics_to_csv,
    save_confusion_matrix,
)

from models.base_cnn import BaselineCNN


def parse_args():
    parser = argparse.ArgumentParser(description="Test Baseline CNNs (Fixed Data, Multiple Weight Seeds).")

    parser.add_argument("--csv_path", default="data/train.csv")
    parser.add_argument("--img_dir", default="data/train")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--img_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4, help="Número de processos a carregar imagens")

    parser.add_argument("--seed", type=int, default=None, help="Seed específica a avaliar (omitir para correr todas: 42-46)")

    parser.add_argument("--val_size", type=float, default=0.15)
    parser.add_argument("--test_size", type=float, default=0.15)

    parser.add_argument("--save_dir", default="results/baseline_original")
    # Se não for passado, é derivado automaticamente do save_dir
    parser.add_argument("--experiment_name", default=None,
                        help="Nome do experimento (default: basename do save_dir)")

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
    os.makedirs("results", exist_ok=True)

    # Deriva o nome do experimento a partir do save_dir se não for passado
    if args.experiment_name is None:
        args.experiment_name = os.path.basename(os.path.normpath(args.save_dir))

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    print(f"\nFamília: {args.experiment_name}")
    print(f"Modelos em: {args.save_dir}")
    print("Preparing fixed test dataloader...")

    _, _, test_loader, classes = get_dataloaders(
        csv_path=args.csv_path,
        img_dir=args.img_dir,
        batch_size=args.batch_size,
        img_size=args.img_size,
        val_size=args.val_size,
        test_size=args.test_size,
        augment=False,
        normalize=True,
        aug_csv_path=None,
        aug_img_dir=None,
        num_workers=args.num_workers,
    )

    num_classes = len(classes)
    seeds = [args.seed] if args.seed is not None else list(range(42, 47))

    for seed in seeds:
        print(f"\n{'='*50}\nEvaluating Test Set for CNN Initialization Seed: {seed}\n{'='*50}")
        set_seed(seed)

        current_experiment_name = f"{args.experiment_name}_FixedData_NetSeed_{seed}"
        current_save_dir = os.path.join(args.save_dir, f"NetSeed_{seed}")
        best_model_path = os.path.join(current_save_dir, "baseline_best.pth")

        if not os.path.exists(best_model_path):
            print(f"Model not found at {best_model_path}. Please train this seed first. Skipping...")
            continue

        model = BaselineCNN(num_classes=num_classes).to(device)
        checkpoint = torch.load(best_model_path, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])

        epoch_saved = checkpoint.get('epoch', 'N/A')
        f1_saved = checkpoint.get('val_f1', 0.0)
        print(f"Successfully loaded model from epoch {epoch_saved} (Val F1 Macro: {f1_saved:.4f}).")

        if test_loader is not None:
            test_metrics, test_report, y_true_test, y_pred_test = evaluate_model(
                model=model,
                dataloader=test_loader,
                device=device,
                class_names=classes,
            )

            print("\nTest report:\n")
            print(test_report)

            append_metrics_to_csv(
                metrics=test_metrics,
                epochs=checkpoint.get('epoch', 0),
                experiment_name=f"Test_{current_experiment_name}",
                best_val_acc=checkpoint.get('val_acc', 0.0),
                save_path="results/all_experiments_test.csv",
            )

            save_confusion_matrix(
                y_true=y_true_test,
                y_pred=y_pred_test,
                class_names=classes,
                save_path=os.path.join(current_save_dir, "confusion_matrix_test.png"),
            )

        print(f"Test evaluation completed for Network Seed {seed}.")

    print(f"\n{'='*50}")
    print(f"Família '{args.experiment_name}' concluída. Seeds avaliadas: {seeds}")
    print(f"Resultados em: results/all_experiments_test.csv")


if __name__ == "__main__":
    main()
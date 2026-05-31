import os
import torch
import torch_fidelity
from src.utils import append_generative_metrics_to_csv

def main():
    real_images_dir = 'data/train'
    # TO RUN THIS, YOU MUST HAVE A generated_butterflies FOLDER WITH ONLY AUGMENTED IMAGES
    generated_images_dir = 'code_for_lora/generated_butterflies'
    
    experiment_name = 'Generated Butterflies'
    csv_path = 'results_gen/generative_experiments.csv'

    print("Preparing to calculate FID...")
    print(f"Real Folder (Base): {real_images_dir}")
    print(f"Generated Folder (LoRA): {generated_images_dir}")

    if not os.path.exists(real_images_dir) or not os.path.exists(generated_images_dir):
        print("Error: One of the folders was not found. Make sure you ran the generation first.")
        return

    use_cuda = torch.cuda.is_available()
    print(f"Using GPU (CUDA/ROCm compatible): {use_cuda}")

    metrics_dict = torch_fidelity.calculate_metrics(
        input1=real_images_dir,
        input2=generated_images_dir,
        cuda=use_cuda,
        isc=False,
        fid=True,
        verbose=True,
    )

    fid_result = metrics_dict['frechet_inception_distance']

    print("\n" + "="*30)
    print("        METRIC RESULTS        ")
    print("="*30)
    print(f"Fréchet Inception Distance (FID): {fid_result:.4f}")

    append_generative_metrics_to_csv(
        experiment_name=experiment_name,
        epochs="N/A", 
        fid=fid_result,
        inception_score="N/A",
        save_path=csv_path
    )
    print(f"\nMetrics successfully saved to {csv_path}")

if __name__ == '__main__':
    main()
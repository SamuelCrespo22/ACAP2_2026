import os
import torch_fidelity
from src.utils import append_generative_metrics_to_csv

def main():
    # Caminhos para as pastas de imagens
    # Ajusta estes caminhos se quiseres comparar apenas uma classe específica
    real_images_dir = 'data/train'
    generated_images_dir = 'data/generated_butterflies'
    
    experiment_name = 'Generated Butterflies' # Podes alterar o nome desta experiência
    csv_path = 'results_gen/generative_experiments.csv'

    print("A preparar o cálculo de FID e Inception Score (IS)...")
    print(f"Pasta Real: {real_images_dir}")
    print(f"Pasta Gerada: {generated_images_dir}")

    if not os.path.exists(real_images_dir) or not os.path.exists(generated_images_dir):
        print("Erro: Uma das pastas não foi encontrada. Certifica-te de que executaste a geração primeiro.")
        return

    metrics_dict = torch_fidelity.calculate_metrics(
        input1=real_images_dir,
        input2=generated_images_dir,
        cuda=True,
        isc=True,
        fid=True,
        verbose=True, # Mostra barra de progresso
    )

    print("\n" + "="*30)
    print("      RESULTADOS DAS MÉTRICAS      ")
    print("="*30)
    print(f"Inception Score (IS): {metrics_dict['inception_score_mean']:.4f} ± {metrics_dict['inception_score_std']:.4f}")
    print(f"Fréchet Inception Distance (FID): {metrics_dict['frechet_inception_distance']:.4f}")

    # Guardar métricas no CSV
    append_generative_metrics_to_csv(
        experiment_name=experiment_name,
        epochs="N/A", 
        fid=metrics_dict['frechet_inception_distance'],
        inception_score=metrics_dict['inception_score_mean'],
        save_path=csv_path
    )

if __name__ == '__main__':
    main()
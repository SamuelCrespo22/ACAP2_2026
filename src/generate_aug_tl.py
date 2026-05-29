import os
import pandas as pd
import torch
from diffusers import AutoPipelineForText2Image

# --- CONFIGURAÇÕES ---
csv_path = 'data/train.csv'
lora_weights_dir = 'borboletas_lora'  # A pasta onde o vosso treino guardou os pesos
output_base_dir = 'data/generated_butterflies'  # Pasta temporária para triagem

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"A usar o dispositivo: {device}")

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Não encontrei o ficheiro {csv_path}. Garante que estás a correr o script na raiz.")

    # 1. Calcular o desbalanceamento das classes
    df = pd.read_csv(csv_path)
    counts = df['label'].value_counts()
    max_imagens = counts.max()
    print(f"-> A classe mais populosa tem {max_imagens} imagens.")
    print(f"-> Objetivo: Igualar todas as classes a {max_imagens} exemplares.\n")

    # 2. Carregar o Modelo e os Pesos LoRA treinados por vós
    print("A carregar o Stable Diffusion + os vossos pesos LoRA...")
    pipeline = AutoPipelineForText2Image.from_pretrained(
        "runwayml/stable-diffusion-v1-5", 
        torch_dtype=torch.float16
    ).to(device)
    
    pipeline.load_lora_weights(lora_weights_dir)

    # 3. Truque de poupança de memória para a vossa RTX 3070 Laptop (8GB VRAM)
    pipeline.enable_attention_slicing()

    # 4. Loop de Geração Automatizada
    for classe, total_atual in counts.items():
        imagens_a_gerar = max_imagens - total_atual
        
        if imagens_a_gerar <= 0:
            print(f"Classe [{classe}] já está no máximo ({total_atual} imagens). A saltar...")
            continue

        print(f"\n[Processando] Classe: {classe} | Atual: {total_atual} -> Gerar mais: {imagens_a_gerar}")
        
        # Criar pasta específica para esta classe
        pasta_classe = os.path.join(output_base_dir, str(classe).replace(" ", "_"))
        os.makedirs(pasta_classe, exist_ok=True)

        prompt = f"a macro photo of a {classe} butterfly, natural background"
        
        # Gerar em lotes (batches) pequenos para não estourar a VRAM de 8GB
        batch_size = 2  # Podem tentar aumentar para 4 se virem que a gráfica aguenta
        geradas = 0
        
        while geradas < imagens_a_gerar:
            # Garante que no último lote não gera imagens a mais
            lote_atual = min(batch_size, imagens_a_gerar - geradas)
            
            with torch.inference_mode():
                # num_inference_steps=30 é um bom equilíbrio entre velocidade e qualidade
                resultado = pipeline(
                    prompt, 
                    num_inference_steps=30, 
                    guidance_scale=7.5, 
                    num_images_per_prompt=lote_atual
                ).images

            # Guardar as imagens geradas no disco
            for idx, img in enumerate(resultado):
                nome_ficheiro = f"sintetica_{geradas + idx}.png"
                img.save(os.path.join(pasta_classe, nome_ficheiro))
            
            geradas += lote_atual
            print(f"   Progresso: {geradas}/{imagens_a_gerar} imagens criadas.")

    print("\n[SUCESSO] Processo de geração concluído!")
    print(f"As vossas imagens estão organizadas por classes em: {output_base_dir}")

if __name__ == "__main__":
    main()
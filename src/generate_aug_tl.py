import os
import shutil
import pandas as pd
import torch
from diffusers import AutoPipelineForText2Image
from dataset import get_dataloaders

# --- CONFIGURAÇÕES ---
csv_path = 'data/train.csv'
lora_weights_dir = 'borboletas_lora'  # A pasta onde o vosso treino guardou os pesos
output_dir = 'data/train_aug_tl'
output_csv = 'data/train_aug_tl.csv'

def main():
    if torch.cuda.is_available():
        device = "cuda"
        dtype = torch.float16
    elif torch.backends.mps.is_available():
        device = "mps"
        dtype = torch.float32
    else:
        device = "cpu"
        dtype = torch.float32
    print(f"A usar o dispositivo: {device}")

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Não encontrei o ficheiro {csv_path}. Garante que estás a correr o script na raiz.")

    train_loader, _, _, classes = get_dataloaders(
        csv_path=csv_path,
        img_dir="data/train",
        val_size=0.15,
        test_size=0.15,
        augment=False,
        normalize=False
    )
    train_df = train_loader.dataset.img_labels
    full_df = pd.read_csv(csv_path)
    
    os.makedirs(output_dir, exist_ok=True)
    print("A copiar as imagens originais...")
    for _, row in train_df.iterrows():
        orig_path = os.path.join("data/train", row['filename'])
        if os.path.exists(orig_path):
            shutil.copy2(orig_path, os.path.join(output_dir, row['filename']))

    # 2. Carregar o Modelo e os Pesos LoRA treinados por vós
    print("A carregar o Stable Diffusion + os vossos pesos LoRA...")
    pipeline = AutoPipelineForText2Image.from_pretrained(
        "runwayml/stable-diffusion-v1-5", 
        torch_dtype=dtype,
        safety_checker=None,
        requires_safety_checker=False
    ).to(device)
    
    pipeline.load_lora_weights(lora_weights_dir)

    # 3. Truque de poupança de memória para a vossa RTX 3070 Laptop (8GB VRAM)
    pipeline.enable_attention_slicing()

    new_rows = []

    # 4. Loop de Geração Automatizada
    for class_idx, classe in enumerate(classes):
        total_atual = len(full_df[full_df['label'] == classe])
        
        if 51 <= total_atual <= 60:
            imagens_a_gerar = int(round(total_atual * 0.20))
        elif 61 <= total_atual <= 70:
            imagens_a_gerar = int(round(total_atual * 0.15))
        elif 71 <= total_atual <= 80:
            imagens_a_gerar = int(round(total_atual * 0.10))
        elif 81 <= total_atual <= 90:
            imagens_a_gerar = int(round(total_atual * 0.05))
        else:
            imagens_a_gerar = 0
        
        if imagens_a_gerar <= 0:
            continue

        print(f"\n[Processando] Classe: {classe} | Atual: {total_atual} -> Gerar mais: {imagens_a_gerar}")
        
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
                nome_ficheiro = f"gen_tl_{class_idx}_{geradas + idx}.jpg"
                img.save(os.path.join(output_dir, nome_ficheiro))
                new_rows.append({"filename": nome_ficheiro, "label": classe})
            
            geradas += lote_atual
            print(f"   Progresso: {geradas}/{imagens_a_gerar} imagens criadas.")
            
    if new_rows:
        pd.concat([train_df, pd.DataFrame(new_rows)], ignore_index=True).to_csv(output_csv, index=False)
    else:
        train_df.to_csv(output_csv, index=False)

    print("\n[SUCESSO] Processo de geração concluído!")
    print(f"TL Dataset concluído: {output_csv}")

if __name__ == "__main__":
    main()
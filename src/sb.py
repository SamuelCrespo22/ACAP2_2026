import os
import csv
import torch
import torch_directml
from torchvision import transforms
from PIL import Image
from pathlib import Path
from models.base_cnn import BaselineCNN

# ==========================================
# 1. CONFIGURAÇÕES E CAMINHOS
# ==========================================
SCRIPT_DIR = Path(__file__).parent.resolve()
ROOT_DIR = SCRIPT_DIR.parent

MODEL_PATH = ROOT_DIR / "results/baseline_aug_vae/NetSeed_44/baseline_best.pth"
TEST_DIR = ROOT_DIR / "data/test"
OUTPUT_CSV = SCRIPT_DIR / "submission_gerada.csv"

# NOTA: Confirma se o img_size no teu treino era 224. Se era outro (ex: 256), altera aqui!
IMG_SIZE = 64

def main():
    print("A iniciar geração da submissão...")
    
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"ERRO: Ficheiro de pesos não encontrado em {MODEL_PATH}")
        
    # ==========================================
    # 2. CARREGAR CHECKPOINT E CLASSES REAIS
    # ==========================================
    print("A ler o checkpoint...")
    checkpoint = torch.load(MODEL_PATH, map_location='cpu', weights_only=False)
    
    classes_do_modelo = checkpoint.get('classes', [])
    if not classes_do_modelo:
        raise ValueError("ERRO: O checkpoint não contém a chave 'classes'.")
        
    idx_to_class = {i: str(nome) for i, nome in enumerate(classes_do_modelo)}
    numero_de_classes = len(idx_to_class)
    print(f"Lidas {numero_de_classes} classes perfeitamente alinhadas com o modelo.")

    # ==========================================
    # 3. CONFIGURAR DISPOSITIVO (AMD DirectML)
    # ==========================================
    if torch_directml.is_available():
        device = torch_directml.device()
        print(f"A usar o dispositivo AMD (DirectML): {device}")
    else:
        device = torch.device("cpu")
        print("Aviso: DirectML não encontrado. A usar CPU.")

    # ==========================================
    # 4. INICIALIZAR O MODELO
    # ==========================================
    model = BaselineCNN(num_classes=numero_de_classes)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval() # Modo de inferência

    # ==========================================
    # 5. PREPARAÇÃO DAS IMAGENS (CORRIGIDO)
    # ==========================================
    # Agora está EXATAMENTE igual ao que usaste no treino
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)), 
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    results = []
    
    # ==========================================
    # 6. INFERÊNCIA
    # ==========================================
    if not TEST_DIR.exists():
        raise FileNotFoundError(f"ERRO: Pasta de teste não encontrada em {TEST_DIR}")

    valid_extensions = ('.png', '.jpg', '.jpeg')
    image_files = [f for f in os.listdir(TEST_DIR) if f.lower().endswith(valid_extensions)]
    
    print(f"Encontradas {len(image_files)} imagens em {TEST_DIR}. A prever...")

    # Iterar pelos ficheiros
    for filename in image_files:
        img_path = TEST_DIR / filename
        
        try:
            # Abrir e transformar a imagem
            img = Image.open(img_path).convert('RGB')
            img_tensor = transform(img).unsqueeze(0).to(device) 
            
            # Fazer a previsão
            with torch.no_grad():
                outputs = model(img_tensor)
                _, predicted = torch.max(outputs, 1)
                idx = predicted.item()
            
            # Obter a string da label (já está em MAIÚSCULAS no dicionário do modelo, mas forçamos por segurança)
            label = idx_to_class.get(idx, f"UNKNOWN_{idx}").upper()
            
            # Guardar resultado
            results.append([filename, label])
            
        except Exception as e:
            print(f"Erro ao processar a imagem {filename}: {e}")

    # ==========================================
    # 7. GUARDAR EM CSV
    # ==========================================
    # Opcional: ordenar os resultados pelo nome do ficheiro antes de guardar, se o Kaggle exigir
    results.sort(key=lambda x: x[0])
    
    with open(OUTPUT_CSV, mode='w', newline='', encoding='utf-8') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['filename', 'label'])
        writer.writerows(results)

    print(f"Sucesso Total! Submissão gerada e guardada em: {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
import torch
from pathlib import Path

# Ajusta o caminho se necessário
MODEL_PATH = "results/baseline_aug_vae/NetSeed_44/baseline_best.pth"

# Carregar o checkpoint
checkpoint = torch.load(MODEL_PATH, map_location='cpu', weights_only=False)

print("=== DIAGNÓSTICO DO MODELO ===")
print(f"Época guardada: {checkpoint.get('epoch', 'Desconhecida')}")
print(f"Accuracy na Validação (Treino): {checkpoint.get('val_acc', 'Desconhecida')}")
print(f"F1-Score na Validação: {checkpoint.get('val_f1', 'Desconhecido')}")

# Vamos ver as primeiras 10 classes guardadas no modelo
classes_do_modelo = checkpoint.get('classes', [])
if classes_do_modelo:
    print(f"Total de classes guardadas: {len(classes_do_modelo)}")
    print(f"As 5 primeiras classes no dicionário do modelo: {classes_do_modelo[:5]}")
else:
    print("Aviso: A chave 'classes' não foi encontrada ou está vazia.")
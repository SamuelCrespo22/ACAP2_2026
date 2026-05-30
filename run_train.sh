#!/bin/bash

# Define as variáveis de ambiente
export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export DATASET_DIR="data/train"
export OUTPUT_DIR="borboletas_lora"

# Inicia o treino usando o accelerate
accelerate launch train_text_to_image_lora.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$DATASET_DIR \
  --resolution=512 \
  --center_crop \
  --random_flip \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --max_train_steps=2000 \
  --learning_rate=1e-04 \
  --max_grad_norm=1 \
  --lr_scheduler="cosine" \
  --lr_warmup_steps=0 \
  --output_dir=$OUTPUT_DIR \
  --checkpointing_steps=500 \
  --validation_prompt="a macro photo of a borboleta butterfly, natural background" \
  --seed=42 \
  --use_8bit_adam \
  --mixed_precision="fp16" \
  --gradient_checkpointing \
  --enable_xformers_memory_efficient_attention

@echo off

REM =========================================================
REM Stable Diffusion LoRA Training
REM =========================================================

set MODEL_NAME=runwayml/stable-diffusion-v1-5
set DATASET_DIR=data/train
set OUTPUT_DIR=borboletas_lora

echo Starting LoRA training...
echo.

accelerate launch ^
  --num_processes=1 ^
  --num_machines=1 ^
  --mixed_precision=fp16 ^
  --dynamo_backend=no ^
  train_text_to_image_lora.py ^
  --pretrained_model_name_or_path=%MODEL_NAME% ^
  --train_data_dir=%DATASET_DIR% ^
  --resolution=512 ^
  --center_crop ^
  --random_flip ^
  --train_batch_size=1 ^
  --gradient_accumulation_steps=4 ^
  --max_train_steps=2000 ^
  --learning_rate=1e-4 ^
  --max_grad_norm=1 ^
  --lr_scheduler=cosine ^
  --lr_warmup_steps=0 ^
  --output_dir=%OUTPUT_DIR% ^
  --checkpointing_steps=500 ^
  --validation_prompt="a macro photo of a borboleta butterfly, natural background" ^
  --seed=42 ^
  --mixed_precision=fp16 ^
  --gradient_checkpointing

echo.
echo Training finished.
pause
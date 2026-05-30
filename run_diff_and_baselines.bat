@echo off

echo Starting diffusion + augmented baseline experiments...

mkdir results_gen 2>nul
mkdir results 2>nul

REM =========================================================
REM BASELINE WITH AUGMENTED DATA
REM =========================================================

echo Running baseline with DCGAN augmentation...
python -u src/baseline.py --epochs 200  --aug_csv_path data/train_aug_dcgan.csv --aug_img_dir data/train_aug_dcgan --save_dir results/baseline_aug_dcgan --experiment_name "Baseline Aug DCGAN"

echo Running baseline with Transfer Learning augmentation...
python -u src/baseline.py --epochs 200  --aug_csv_path data/train_aug_tl.csv --aug_img_dir data/train_aug_tl --save_dir results/baseline_aug_tl --experiment_name "Baseline Aug TL"

echo Running baseline with VAE augmentation...
python -u src/baseline.py --epochs 200  --aug_csv_path data/train_aug_vae.csv --aug_img_dir data/train_aug_vae --save_dir results/baseline_aug_vae --experiment_name "Baseline Aug VAE" 

echo Running baseline with WGAN-GP augmentation...
python -u src/baseline.py --epochs 200 --aug_csv_path data/train_aug_wgangp.csv --aug_img_dir data/train_aug_wgangp --save_dir results/baseline_aug_wgangp --experiment_name "Baseline Aug WGAN-GP" 


REM =========================================================
REM DIFFUSION - 200 epochs, eval every 15
REM =========================================================

echo Running Diffusion cosine 200ep eval15...
python -u src/train_diffusion.py --schedule cosine --epochs 200 --eval_every 15 --batch_size 64 --save_dir results_gen/diff_cosine_200ep_eval15 --experiment_name "Diff cosine 200ep eval15" > results_gen\log_diff_cosine_200ep_eval15.txt 2>&1



echo All experiments finished!
pause
@echo off

REM =========================================================
REM LONG GENERATIVE EXPERIMENTS
REM =========================================================

echo Starting long generative experiments...

REM Criar a pasta base para os logs não darem erro no CMD
mkdir results_gen 2>nul

REM =========================================================
REM VAE
REM =========================================================

echo Running VAE latent 128...
python src/train_vae.py --latent_dim 128 --epochs 300 --batch_size 128 --save_dir results_gen/vae_128 --experiment_name "VAE latent 128" > results_gen\log_vae_128.txt 2>&1

echo Running VAE latent 256...
python src/train_vae.py --latent_dim 256 --epochs 300 --batch_size 128 --save_dir results_gen/vae_256 --experiment_name "VAE latent 256" > results_gen\log_vae_256.txt 2>&1

REM =========================================================
REM DCGAN
REM =========================================================

echo Running DCGAN lr 2e-4...
python src/train_dcgan.py --lr_g 2e-4 --lr_d 2e-4 --epochs 300 --batch_size 128 --save_dir results_gen/dcgan_lr2e4 --experiment_name "DCGAN lr 2e-4" > results_gen\log_dcgan_lr2e4.txt 2>&1

echo Running DCGAN split lr...
python src/train_dcgan.py --lr_g 1e-4 --lr_d 4e-4 --epochs 300 --batch_size 128 --save_dir results_gen/dcgan_splitlr --experiment_name "DCGAN lrG 1e-4 lrD 4e-4" > results_gen\log_dcgan_splitlr.txt 2>&1

REM =========================================================
REM WGAN-GP
REM =========================================================

echo Running WGAN-GP n_critic 5...
python src/train_wgangp.py --n_critic 5 --epochs 300 --batch_size 128 --save_dir results_gen/wgangp_ncritic5 --experiment_name "WGAN-GP n_critic 5" > results_gen\log_wgangp_ncritic5.txt 2>&1

echo Running WGAN-GP n_critic 3...
python src/train_wgangp.py --n_critic 3 --epochs 300 --batch_size 128 --save_dir results_gen/wgangp_ncritic3 --experiment_name "WGAN-GP n_critic 3" > results_gen\log_wgangp_ncritic3.txt 2>&1

REM =========================================================
REM OPTIONAL 500 EPOCH RUNS (Only if needed)
REM =========================================================

echo Running WGAN-GP n_critic 5 (500 epochs)...
python src/train_wgangp.py --n_critic 5 --epochs 500 --batch_size 128 --save_dir results_gen/wgangp_ncritic5_500ep --experiment_name "WGAN-GP n_critic 5 500 epochs" > results_gen\log_wgangp_ncritic5_500ep.txt 2>&1

echo Running DCGAN lr 2e-4 (500 epochs)...
python src/train_dcgan.py --lr_g 2e-4 --lr_d 2e-4 --epochs 500 --batch_size 128 --save_dir results_gen/dcgan_lr2e4_500ep --experiment_name "DCGAN lr 2e-4 500 epochs" > results_gen\log_dcgan_lr2e4_500ep.txt 2>&1

echo All experiments finished!
pause
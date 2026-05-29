@echo off

REM =========================================================
REM LONG GENERATIVE EXPERIMENTS - eval_every=2, patience=20
REM =========================================================

echo Starting long generative experiments with eval_every=2 and patience=20...

mkdir results_gen 2>nul

REM =========================================================
REM VAE
REM =========================================================

echo Running VAE latent 128 ev2 p20...
python -u src/train_vae.py --latent_dim 128 --epochs 300 --batch_size 128 --save_dir results_gen/vae128_ev2_p20 --experiment_name "VAE128 ev2 p20" > results_gen\log_vae128_ev2_p20.txt 2>&1

echo Running VAE latent 256 ev2 p20...
python -u src/train_vae.py --latent_dim 256 --epochs 300 --batch_size 128 --save_dir results_gen/vae256_ev2_p20 --experiment_name "VAE256 ev2 p20" > results_gen\log_vae256_ev2_p20.txt 2>&1

REM =========================================================
REM DCGAN
REM =========================================================

echo Running DCGAN lr 2e-4 ev2 p20...
python -u src/train_dcgan.py --lr_g 2e-4 --lr_d 2e-4 --epochs 300 --batch_size 128 --save_dir results_gen/dcgan2e4_ev2_p20 --experiment_name "DCGAN2e4 ev2 p20" > results_gen\log_dcgan2e4_ev2_p20.txt 2>&1

echo Running DCGAN split lr ev2 p20...
python -u src/train_dcgan.py --lr_g 1e-4 --lr_d 4e-4 --epochs 300 --batch_size 128 --save_dir results_gen/dcgan_split_ev2_p20 --experiment_name "DCGANsplit ev2 p20" > results_gen\log_dcgan_split_ev2_p20.txt 2>&1

REM =========================================================
REM WGAN-GP
REM =========================================================

echo Running WGAN-GP ncritic5 ev2 p20...
python -u src/train_wgangp.py --n_critic 5 --epochs 300 --batch_size 128 --save_dir results_gen/wgan_n5_ev2_p20 --experiment_name "WGAN n5 ev2 p20" > results_gen\log_wgan_n5_ev2_p20.txt 2>&1

echo Running WGAN-GP ncritic3 ev2 p20...
python -u src/train_wgangp.py --n_critic 3 --epochs 300 --batch_size 128 --save_dir results_gen/wgan_n3_ev2_p20 --experiment_name "WGAN n3 ev2 p20" > results_gen\log_wgan_n3_ev2_p20.txt 2>&1

REM =========================================================
REM OPTIONAL 500 EPOCH RUNS
REM Uncomment if needed
REM =========================================================

REM echo Running WGAN-GP ncritic5 500ep ev2 p20...
REM python -u src/train_wgangp.py --n_critic 5 --epochs 500 --batch_size 128 --save_dir results_gen/wgan_n5_500ep_ev2_p20 --experiment_name "WGAN n5 500ep ev2 p20" > results_gen\log_wgan_n5_500ep_ev2_p20.txt 2>&1

REM echo Running DCGAN lr 2e-4 500ep ev2 p20...
REM python -u src/train_dcgan.py --lr_g 2e-4 --lr_d 2e-4 --epochs 500 --batch_size 128 --save_dir results_gen/dcgan2e4_500ep_ev2_p20 --experiment_name "DCGAN2e4 500ep ev2 p20" > results_gen\log_dcgan2e4_500ep_ev2_p20.txt 2>&1
echo All experiments finished!
pause
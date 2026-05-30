@echo off

REM =========================================================
REM SELECTED LONG EXPERIMENTS
REM WGAN 500 + Diffusion cosine/linear 500
REM =========================================================

echo Starting selected long experiments...

mkdir results_gen 2>nul

REM =========================================================
REM VAE - COMMENTED
REM =========================================================

REM echo Running VAE latent 128 ev2 p20 padding...
REM python -u src/train_vae.py --latent_dim 128 --epochs 300 --batch_size 128 --save_dir results_gen/vae128_ev2_p20_padding --experiment_name "VAE128 ev2 p20 padding" > results_gen\log_vae128_ev2_p20_padding.txt 2>&1

REM echo Running VAE latent 256 ev2 p20 padding...
REM python -u src/train_vae.py --latent_dim 256 --epochs 300 --batch_size 128 --save_dir results_gen/vae256_ev2_p20_padding --experiment_name "VAE256 ev2 p20 padding" > results_gen\log_vae256_ev2_p20_padding.txt 2>&1

REM =========================================================
REM DCGAN - COMMENTED
REM =========================================================

REM echo Running DCGAN lr 2e-4 ev2 p20 padding...
REM python -u src/train_dcgan.py --lr_g 2e-4 --lr_d 2e-4 --epochs 300 --batch_size 128 --save_dir results_gen/dcgan2e4_ev2_p20_padding --experiment_name "DCGAN2e4 ev2 p20 padding" > results_gen\log_dcgan2e4_ev2_p20_padding.txt 2>&1

REM echo Running DCGAN split lr ev2 p20 padding...
REM python -u src/train_dcgan.py --lr_g 1e-4 --lr_d 4e-4 --epochs 300 --batch_size 128 --save_dir results_gen/dcgan_split_ev2_p20_padding --experiment_name "DCGANsplit ev2 p20 padding" > results_gen\log_dcgan_split_ev2_p20_padding.txt 2>&1


REM =========================================================
REM DIFFUSION - 100 EPOCHS
REM =========================================================

echo Running Diffusion cosine 100ep padding final...
python -u src/train_diffusion.py --schedule cosine --epochs 100 --batch_size 64 --save_dir results_gen/diff_cosine_100ep_padding_final --experiment_name "Diff cosine 100ep padding final" > results_gen\log_diff_cosine_100ep_padding_final.txt 2>&1

echo Running Diffusion linear 500ep padding final...
python -u src/train_diffusion.py --schedule linear --epochs 100 --batch_size 64 --save_dir results_gen/diff_linear_100ep_padding_final --experiment_name "Diff linear 100ep padding final" > results_gen\log_diff_linear_100ep_padding_final.txt 2>&1


REM =========================================================
REM WGAN-GP - RUNNING ONLY 500 EPOCHS
REM =========================================================

echo Running WGAN-GP ncritic5 500ep padding final...
python -u src/train_wgangp.py --n_critic 5 --epochs 500 --batch_size 128 --save_dir results_gen/wgan_n5_500ep_padding_final --experiment_name "WGAN n5 500ep padding final" > results_gen\log_wgan_n5_500ep_padding_final.txt 2>&1

REM echo Running WGAN-GP ncritic3 ev2 p20 padding...
REM python -u src/train_wgangp.py --n_critic 3 --epochs 300 --batch_size 128 --save_dir results_gen/wgan_n3_ev2_p20_padding --experiment_name "WGAN n3 ev2 p20 padding" > results_gen\log_wgan_n3_ev2_p20_padding.txt 2>&1


echo All selected experiments finished!
pause
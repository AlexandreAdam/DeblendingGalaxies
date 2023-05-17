#!/bin/bash
#SBATCH --tasks=1
#SBATCH --cpus-per-task=1 # maximum cpu per task is 3.5 per gpus
#SBATCH --gres=gpu:1
#SBATCH --mem=32G			 # memory per node
#SBATCH --time=00-01:00		# time (DD-HH:MM)
#SBATCH --account=rrg-lplevass
#SBATCH --job-name=Deconvolution
#SBATCH --output=%x-%j.out
source $HOME/environments/scope/bin/activate
python $DEBLENDER/scripts/psf_deconvolution.py\
  --experiment_name=skirt_g_hst_mock_psf_deconvolution\
  --psf_fits=$DEBLENDER/data/psf_target1.fits\
  --psf_key=PRIMARY\
  --real_data\
  --observation_fits=$DEBLENDER/data/target1.fits\
  --observation_key=PRIMARY\
  --model_pixels=512\
  --model_pixel_size=0.01\
  --noise_rms=1e-1\
  --super_sampling_factor=1\
  --diagonal_gaussian_likelihood\
  --result_dir=$DEBLENDER/results/\
  --checkpoints_dir=$DEBLENDER/ncsnpp_skirt_g_larger_230421030814/\
  -N=8000\
  -W=30\
  -B=10\

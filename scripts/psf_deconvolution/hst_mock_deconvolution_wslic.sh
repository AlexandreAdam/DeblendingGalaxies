#!/bin/bash
#SBATCH --array=1-10
#SBATCH --tasks=1
#SBATCH --cpus-per-task=1 # maximum cpu per task is 3.5 per gpus
#SBATCH --gres=gpu:1
#SBATCH --mem=32G			 # memory per node
#SBATCH --time=00-05:00		# time (DD-HH:MM)
#SBATCH --account=rrg-lplevass
#SBATCH --job-name=Deconvolution
#SBATCH --output=%x-%j.out
source $HOME/environments/scope/bin/activate
python $DEBLENDER/scripts/psf_deconvolution.py\
  --experiment_name=skirt_g_hst_mock_psf_deconvolution\
  --psf_fits=$DEBLENDER/data/F814w_WFC3UV_cropped_psf.fits\
  --psf_key=PRIMARY\
  --injection_test\
  --dataset_path=/home/aadam/scratch/skirt512_grizy.h5\
  --dataset_channels 0\
  --dataset_key=images\
  --dataset_id=42\
  --dataset_channels_last\
  --observation_pixels=128\
  --observation_pixel_size=0.04\
  --model_pixels=256\
  --model_pixel_size=0.02\
  --super_sampling_factor=4\
  --slic_likelihood\
  --slic_model=ncsnpp_hst_noise_psf_f814w_wfc3uv_230421041030\
  --result_dir=$DEBLENDER/results/\
  --checkpoints_dir=$DEBLENDER/ncsnpp_skirt_g_256_linear_230425023555\
  -N=4000\
  -W=30\
  -B=10\

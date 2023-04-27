#!/bin/bash
#SBATCH --array=1-10
#SBATCH --tasks=1
#SBATCH --cpus-per-task=1 # maximum cpu per task is 3.5 per gpus
#SBATCH --gres=gpu:1
#SBATCH --mem=32G			 # memory per node
#SBATCH --time=00-02:00		# time (DD-HH:MM)
#SBATCH --account=rrg-lplevass
#SBATCH --job-name=Deconvolution
#SBATCH --output=%x-%j.out
source $HOME/environments/scope/bin/activate
python $DEBLENDER/scripts/psf_deconvolution.py\
  --experiment_name=skirt_g_hst_probes_mock_psf_deconvolution_115\
  --psf_fits=$DEBLENDER/data/F814w_WFC3UV_cropped_psf.fits\
  --psf_key=PRIMARY\
  --injection_test\
  --dataset_path=/home/aadam/projects/rrg-lplevass/data/probes.h5\
  --dataset_channels 0\
  --probes\
  --dataset_key=galaxies\
  --dataset_id=115\
  --dataset_channels_last\
  --observation_pixels=64\
  --observation_pixel_size=0.04\
  --model_pixels=256\
  --model_pixel_size=0.01\
  --noise_rms=2e-2\
  --super_sampling_factor=4\
  --diagonal_gaussian_likelihood \
  --result_dir=$DEBLENDER/results/\
  --checkpoints_dir=$HOME/projects/rrg-lplevass/data/score_models/ncsnpp_ct_g_220912024942\
  -N=4000\
  -W=30\
  -B=10\

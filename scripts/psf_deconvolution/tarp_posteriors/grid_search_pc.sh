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

alpha_grid=(200 200 200 200 50 50 50 50)
N_grid=(500 1000 2000 4000 500 1000 2000 4000)
M_grid=(1 1 1 1 2 2 2 2)
snr=1e-1

alpha=${alpha_grid[$THIS_WORKER]}
N=${N_grid[$THIS_WORKER]}
M=${M[$THIS_WORKER]}

source $HOME/environments/milex/bin/activate
python $DEBLENDER/scripts/psf_deconvolution.py\
  --experiment_name="tarp_posterior_alpha$alpha_N$N"\
  --psf_fits=$DEBLENDER/data/F814w_WFC3UV_cropped_psf.fits\
  --psf_key=PRIMARY\
  --injection_test\
  --dataset_path=$HOME/projects/rrg-lplevass/data/probes.h5\
  --dataset_channels 0\
  --dataset_key=galaxies\
  --dataset_id=468\
  --dataset_channels_last\
  --observation_pixels=128\
  --observation_pixel_size=0.04\
  --model_pixels=256\
  --downsample=1\
  --model_pixel_size=0.02\
  --noise_rms=2e-2\
  --super_sampling_factor=4\
  --diagonal_gaussian_likelihood\
  --result_dir=$DEBLENDER/results/\
  --checkpoints_dir=$DEBLENDER/models/ncsnpp_ct_g_220912024942\
  -N=4000\
  -W=30\
  -B=10\

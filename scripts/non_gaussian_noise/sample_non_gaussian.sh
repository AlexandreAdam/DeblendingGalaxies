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
  --experiment_name=probes_g_hst_mock_psf_deconvolution_slic_101\
  --psf_fits=$DEBLENDER/data/F814w_WFC3UV_cropped_psf.fits\
  --psf_key=PRIMARY\
  --injection_test\
  --probes\
  --dataset_path=$HOME/projects/rrg-lplevass/data/probes.h5\
  --dataset_channels 0\
  --dataset_key=galaxies\
  --dataset_id=101\
  --dataset_channels_last\
  --observation_pixels=64\
  --observation_pixel_size=0.04\
  --super_sampling_factor=4\
  --model_pixels=256\
  --model_pixel_size=0.01\
  --downsample=0\
  --slic_model=$DEBLENDER/models/ncsnpp_hst_noise_psf_f814w_wfc3uv_230421041030\
  --slic_likelihood\
  --slic_likelihood_fudge_factor=10\
  --noise_map_multiplicative_factor=1\
  --noise_map=$HOME/projects/rrg-lplevass/data/hst_cutouts_noclip.npy\
  --result_dir=$DEBLENDER/results/\
  --checkpoints_dir=$HOME/projects/rrg-lplevass/data/score_models/ncsnpp_ct_g_220912024942\
  -N=4000\
  -W=30\
  -B=10\

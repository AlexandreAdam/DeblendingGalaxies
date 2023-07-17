#!/bin/bash
#SBATCH --tasks=1
#SBATCH --cpus-per-task=1 # maximum cpu per task is 3.5 per gpus
#SBATCH --gres=gpu:1
#SBATCH --mem=32G			 # memory per node
#SBATCH --time=02-23:00		# time (DD-HH:MM)
#SBATCH --account=rrg-lplevass
#SBATCH --job-name=Train_SLIC_hst_psf_deconvolution64
#SBATCH --output=%x-%j.out
source $HOME/environments/milex/bin/activate
python $DEBLENDER/scripts/train_slic_with_psf.py\
  --loss=dsm\
  --model_architecture=ncsnpp\
  --dataset_path=$HOME/projects/rrg-lplevass/data/hst_cutouts_noclip.npy\
  --model_parameters=$DEBLENDER/scripts/psf_deconvolution/ncsnpp_hst_noise.json\
  --epochs=10000\
  --learning_rate=5e-5\
  --max_time=70\
  --batch_size=8\
  --logdir=$DEBLENDER/logs/\
  --logname_prefixe=ncsnpp_hst_noise_ssf2_32_psf_f814w_wfc3uv\
  --model_dir=$DEBLENDER/models/\
  --checkpoints=3\
  --seed=42\
  --psf_fits=$DEBLENDER/data/F814w_WFC3UV_cropped_psf.fits\
  --psf_key="PRIMARY"\
  --prior_model=$DEBLENDER/models/ncsnpp_probes_g_64_230604024652\
  --observation_pixels=32\
  --observation_pixel_size=0.05\
  --model_pixels=64\
  --model_pixel_size=0.025\
  --zero_padding=0\
  --super_sampling_factor=4\

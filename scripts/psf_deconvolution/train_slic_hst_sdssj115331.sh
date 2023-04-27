#!/bin/bash
#SBATCH --tasks=1
#SBATCH --cpus-per-task=1 # maximum cpu per task is 3.5 per gpus
#SBATCH --gres=gpu:1
#SBATCH --mem=32G			 # memory per node
#SBATCH --time=02-23:00		# time (DD-HH:MM)
#SBATCH --account=rrg-lplevass
#SBATCH --job-name=Train_SLIC_hst_psf_deconvolution
#SBATCH --output=%x-%j.out
source $HOME/environments/scope/bin/activate
python $DEBLENDER/scripts/train_slic_with_psf.py\
  --loss=dsm\
  --model_architecture=ncsnpp\
  --dataset_path=$DEBLENDER/data/SDSSJ115331_noise_cutouts.npy\
  --model_parameters=$DEBLENDER/scripts/psf_deconvolution/ncsnpp_hst_noise.json\
  --epochs=10000\
  --learning_rate=5e-5\
  --max_time=70\
  --batch_size=8\
  --logdir=$DEBLENDER/logs/\
  --logname_prefixe=ncsnpp_hst_noise_psf_f814w_wfc3uv\
  --model_dir=$DEBLENDER/models/\
  --checkpoints=5\
  --seed=42\
  --psf_fits=$DEBLENDER/data/F606w_WFC3UV_cropped_psf.fits\
  --psf_key="PRIMARY"\
  --prior_model=$HOME/projects/rrg-lplevass/data/score_models/ncsnpp_ct_g_220912024942\
  --observation_pixels=128\
  --observation_pixel_size=0.04\
  --model_pixels=128\
  --model_pixel_size=0.04\
  --zero_padding=0\
  --super_sampling_factor=4\

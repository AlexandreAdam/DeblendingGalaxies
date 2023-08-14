#!/bin/bash
#SBATCH --tasks=1
#SBATCH --cpus-per-task=1 # maximum cpu per task is 3.5 per gpus
#SBATCH --gres=gpu:1
#SBATCH --mem=32G                        # memory per node
#SBATCH --time=02-23:00         # time (DD-HH:MM)
#SBATCH --account=rrg-lplevass
#SBATCH --job-name=Train_KernelSLIC_f814w_ssf2
#SBATCH --output=%x-%j.out

source $HOME/environments/milex/bin/activate
python $DEBLENDER/scripts/train_slic_with_psf_v2.py\
  --model_architecture=ncsnpp\
  --dataset_path=$DEBLENDER/data/connor_targets_flat_fields_noise_cutouts_flux_lt_0.02.npy\
  --model_parameters=$DEBLENDER/scripts/psf_deconvolution/kernel_slic/ncsnpp_vp_hst_noise.json\
  --epochs=10000\
  --learning_rate=1e-4\
  --max_time=70\
  --batch_size=32\
  --logname_prefix=ncsnpp_vp_hst_noise_flat_field_psf_f814w_ssf2\
  --model_dir=$DEBLENDER/models/\
  --checkpoints=3\
  --psf_fits=$DEBLENDER/data/F814w_WFC3UV_psf.fits\
  --psf_key="PRIMARY"\
  --observation_pixels=64\
  --observation_pixel_size=0.05\
  --model_pixels=128\
  --model_pixel_size=0.025\
  --zero_padding=0\
  --super_sampling_factor=4\

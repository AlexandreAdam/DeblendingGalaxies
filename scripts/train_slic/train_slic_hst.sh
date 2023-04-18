#!/bin/bash
#SBATCH --tasks=1
#SBATCH --cpus-per-task=1 # maximum cpu per task is 3.5 per gpus
#SBATCH --gres=gpu:1
#SBATCH --mem=32G			 # memory per node
#SBATCH --time=00-01:00		# time (DD-HH:MM)
#SBATCH --account=rrg-lplevass
#SBATCH --job-name=Train_SLIC_hst_psf_deconvolution
#SBATCH --output=%x-%j.out
source $HOME/environments/scope/bin/activate
python $DEBLENDER/scripts/train_slic_with_psf.py\
  --model_architecture=ncsnpp\
  --dataset_path=$HOME/projects/rrg-lplevass/data/hst_cutouts_noclip.npy\
  --model_parameters=$DEBLENDER/scripts/train_slic/ncsnpp_hst_noise.json\
  --epochs=10000\
  --learning_rate=5e-5\
  --max_time=70\
  --batch_size=32\
  --logdir=$DEBLENDER/logs/\
  --logname_prefixe=ncsnpp_hst_noise_for_psf_deconvolution\
  --model_dir=$DEBLENDER/models/\
  --checkpoints=5\
  --seed=42\
  --psf_fits=$HOME/scratch/SKIRT9_hsc_mocks/063/shalo_063-382649_v1_HSC_GRIZY.fits\
  --psf_key="SUBARU_HSC.G PSF"\
  --prior_model=$DEBLENDER/models/ncsnpp_skirt_g_230415205127\
  --observation_pixels=128\
  --observation_pixel_size=0.17\
  --model_pixels=512\
  --model_pixel_size=0.04\
  --zero_padding=0\
  --hutchinson_noise_type=rademacher

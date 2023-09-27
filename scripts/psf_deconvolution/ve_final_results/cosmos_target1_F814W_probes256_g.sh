#!/bin/bash
#SBATCH --array=1-15
#SBATCH --tasks=1
#SBATCH --cpus-per-task=1 # maximum cpu per task is 3.5 per gpus
#SBATCH --gres=gpu:1
#SBATCH --mem=32G			 # memory per node
#SBATCH --time=00-02:00		# time (DD-HH:MM)
#SBATCH --account=rrg-lplevass
#SBATCH --job-name=Deconvolution_cosmos_target1_probes
#SBATCH --output=%x-%j.out
source $HOME/environments/milex/bin/activate
python $DEBLENDER/scripts/psf_deconvolution.py\
  --experiment_name=cosmos_target1_F814W_probes256_g_N3000\
  --psf_fits=$DEBLENDER/data/F814w_WFC3UV_psf.fits\
  --psf_key=PRIMARY\
  --real_data\
  --observation_fits $DEBLENDER/data/connors_target1.fits\
  --observation_keys 2 3 4 5\
  --psf_super_sampling_factor=4\
  --model_super_sampling_factor=4\
  --model_pixels=256\
  --model_pixel_size=0.0125\
  --fiducial_ra="9:57:56.5294"\
  --fiducial_dec="2:27:37.963"\
  --slic_likelihood\
  --slic_model=$DEBLENDER/models/ncsnpp_ve_hst_noise_flat_field_psf_f814w_ssf4_230926113009\
  --prior_model=$DEBLENDER/models/ncsnpp_ve_probes_g_256_230926020331\
  --result_dir=$DEBLENDER/results/ve_final_posteriors\
  -N=3000\
  -W=30\
  -B=10\

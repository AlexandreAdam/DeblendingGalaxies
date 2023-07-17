#!/bin/bash
#SBATCH --array=1-10
#SBATCH --tasks=1
#SBATCH --cpus-per-task=1 # maximum cpu per task is 3.5 per gpus
#SBATCH --gres=gpu:1
#SBATCH --mem=32G			 # memory per node
#SBATCH --time=00-05:00		# time (DD-HH:MM)
#SBATCH --account=rrg-lplevass
#SBATCH --job-name=Deconvolution_target3_probes_alpha100
#SBATCH --output=%x-%j.out
source $HOME/environments/caustic/bin/activate
python $DEBLENDER/scripts/psf_deconvolution.py\
  --experiment_name=posterior_connor_target3_probes_g_prior_guidance100_N2000\
  --psf_fits=$DEBLENDER/data/F814w_WFC3UV_cropped_psf.fits\
  --psf_key=PRIMARY\
  --real_data\
  --observation_fits=$DEBLENDER/data/connors_target3.fits\
  --observation_keys 2 3 4 5\
  --super_sampling_factor=4\
  --model_pixels=256\
  --model_pixel_size=0.0125\
  --fidcuial_ra="9:57:49.1102"\
  --fidcual_dec="2:28:19.430"\
  --super_sampling_factor=4\
  --slic_likelihood\
  --slic_model=$DEBLENDER/models/ncsnpp_hst_noise_psf_f814w_wfc3uv_230603210517\
  --slic_guidance_factor=100\
  --result_dir=$DEBLENDER/results/\
  --prior_model=$HOME/projects/rrg-lplevass/data/score_models/ncsnpp_ct_g_220912024942\
  -N=2000\
  -W=100\
  -B=10\

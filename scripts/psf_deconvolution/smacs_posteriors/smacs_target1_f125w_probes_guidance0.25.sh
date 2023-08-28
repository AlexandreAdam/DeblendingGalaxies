#!/bin/bash
#SBATCH --array=1-10
#SBATCH --tasks=1
#SBATCH --cpus-per-task=1 # maximum cpu per task is 3.5 per gpus
#SBATCH --gres=gpu:1
#SBATCH --mem=32G			 # memory per node
#SBATCH --time=00-05:00		# time (DD-HH:MM)
#SBATCH --account=rrg-lplevass
#SBATCH --job-name=Deconvolution_smacs_target0_F125w_probes
#SBATCH --output=%x-%j.out
source $HOME/environments/milex/bin/activate
python $DEBLENDER/scripts/psf_deconvolution.py\
  --experiment_name=posterior_smacs_target0_F125W_probes_g_prior_guidance1_N2000\
  --psf_fits=$DEBLENDER/data/F105w_WFC3IR_psf.fits\
  --psf_key=PRIMARY\
  --real_data\
  --observation_fits $DEBLENDER/data/smacs_target0_F125W_0.fits \
  $DEBLENDER/data/smacs_target0_F125W_1.fits \
  $DEBLENDER/data/smacs_target0_F125W_2.fits \
  $DEBLENDER/data/smacs_target0_F125W_3.fits \
  --observation_keys SCI\
  --super_sampling_factor=4\
  --model_pixels=256\
  --model_pixel_size=0.0320\
  --probes\
  --fiducial_ra="7:23:24.5462"\
  --fiducial_dec="-73:27:20.172"\
  --slic_likelihood\
  --slic_model=$DEBLENDER/models/ncsnpp_hst_noise_psf_f814w_wfc3uv_230603210517\
  --slic_guidance_factor=0.25\
  --result_dir=$DEBLENDER/results/\
  --prior_model=$DEBLENDER/models/ncsnpp_ct_g_220912024942\
  -N=2000\
  -W=100\
  -B=10\

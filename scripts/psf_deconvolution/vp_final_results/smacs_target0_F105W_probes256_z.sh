#!/bin/bash
#SBATCH --array=1-15
#SBATCH --tasks=1
#SBATCH --cpus-per-task=1 # maximum cpu per task is 3.5 per gpus
#SBATCH --gres=gpu:1
#SBATCH --mem=32G			 # memory per node
#SBATCH --time=00-02:00		# time (DD-HH:MM)
#SBATCH --account=rrg-lplevass
#SBATCH --job-name=Deconvolution_smacs_target0_probes
#SBATCH --output=%x-%j.out
source $HOME/environments/milex/bin/activate
python $DEBLENDER/scripts/psf_deconvolution.py\
  --experiment_name=smacs_target0_F105W_probes256_z_N3000\
  --psf_fits=$DEBLENDER/data/F105w_WFC3IR_psf.fits\
  --psf_key=PRIMARY\
  --real_data\
  --observation_fits \
  $DEBLENDER/data/smacs_target0_F105WO1_0.fits \
  $DEBLENDER/data/smacs_target0_F105WO1_1.fits \
  $DEBLENDER/data/smacs_target0_F105WO1_2.fits \
  $DEBLENDER/data/smacs_target0_F105WO1_3.fits \
  --observation_keys SCI\
  --psf_super_sampling_factor=4\
  --model_super_sampling_factor=8\
  --model_pixels=256\
  --model_pixel_size=0.01625\
  --fiducial_ra="7:23:24.5462"\
  --fiducial_dec="-73:27:20.172"\
  --slic_likelihood\
  --slic_model=$DEBLENDER/models/ncsnpp_vp_hst_noise_flat_field_psf_f105w_ssf8_230821133228\
  --prior_model=$DEBLENDER/models/ncsnpp_vp_probes_z_256_230824141341\
  --result_dir=$DEBLENDER/results/vp_final_posteriors\
  -N=3000\
  -W=20\
  -B=10\

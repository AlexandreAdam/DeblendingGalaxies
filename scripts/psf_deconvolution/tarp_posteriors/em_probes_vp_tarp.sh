#!/bin/bash
#SBATCH --array=1-10
#SBATCH --tasks=1
#SBATCH --cpus-per-task=1 # maximum cpu per task is 3.5 per gpus
#SBATCH --gres=gpu:1
#SBATCH --mem=32G			 # memory per node
#SBATCH --time=00-10:00		# time (DD-HH:MM)
#SBATCH --account=rrg-lplevass
#SBATCH --job-name=Deconvolution_tarp_em
#SBATCH --output=%x-%j.out

source $HOME/environments/milex/bin/activate


n_posteriors=300 
walkers=50
batch_size=10
n_obs=1
N=2000
result_dir=$DEBLENDER/results/tarp_rm_202309

index_start=0

image_index=$index_start
noise_index=$index_start
echo ${mass_grid[$grid_id]}
for ((i=1;i<=n_posteriors;i++))
    do
    end=$(($noise_index + $n_obs - 1))
    noise_indices=$(seq $noise_index $end)
    python $DEBLENDER/scripts/psf_deconvolution.py\
      --experiment_name=em_probes_vp_prior\
      --result_dir=$result_dir\
      --prior_model=$DEBLENDER/models/\
      --psf_fits=$DEBLENDER/data/F814w_WFC3UV_cropped_psf.fits\
      --psf_key=PRIMARY\
      --injection_test\
      --sample_reference_from_prior\
      --sample_noise_from_likelihood\
      --model_pixels=64\
      --model_pixel_size=0.025\
      --observation_pixels=32\
      --observation_pixel_size=0.05\
      --n_obs=$n_obs\
      --super_sampling_factor=4\
      --slic_likelihood\
      --slic_model=$DEBLENDER/models/\
      --slic_guidance_factor=1\
      --em_iterations=$N\
      --walkers=$walkers\
      --batch_size=$batch_size\
    image_index=$(($image_index+1))
    noise_index=$(($noise_index+$n_obs))
done


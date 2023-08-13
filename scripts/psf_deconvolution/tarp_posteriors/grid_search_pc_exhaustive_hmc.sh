#!/bin/bash
#SBATCH --array=1-20
#SBATCH --tasks=1
#SBATCH --cpus-per-task=1 # maximum cpu per task is 3.5 per gpus
#SBATCH --gres=gpu:1
#SBATCH --mem=32G			 # memory per node
#SBATCH --time=02-00:00		# time (DD-HH:MM)
#SBATCH --account=rrg-lplevass
#SBATCH --job-name=Deconvolution_tarp_hmc
#SBATCH --output=%x-%j.out

source $HOME/environments/milex/bin/activate

# Main setup
mass_grid=(0.1 0.1 0.1 1 1 1)
M_grid=(1 1 3 1 1 3)
leapfrog_steps_grid=(2 5 2 2 5 2)

total=${#mass_grid[@]}

n_posteriors=100 
walkers=10
batch_size=10
alpha=100
n_obs=2
snr_grid=1e-1
corrector_tmin_grid=0.5
N=500
result_dir=$DEBLENDER/results/tarp_hmc_20230723

index_start=0

echo "Grid search over $total TARP experiments"

for ((grid_id=0;grid_id<total;grid_id++))
do
    mass=${mass_grid[$grid_id]}
    leapfrog_steps=${leapfrog_steps_grid[$grid_id]}
    M=${M_grid[$grid_id]}
    echo "Posteriors for mass: $mass | leapfrog_steps: $leapfrog_steps | M: $M | corrector_tmin: $corrector_tmin | snr: $snr"
    image_index=$index_start
    noise_index=$index_start
    echo ${mass_grid[$grid_id]}
    for ((i=1;i<=n_posteriors;i++))
    do
        end=$(($noise_index + $n_obs - 1))
        noise_indices=$(seq $noise_index $end)
        python $DEBLENDER/scripts/psf_deconvolution.py\
          --experiment_name="tarp_hmc_gridid"$grid_id"_posterior"$i"_mass"$mass"_leapfrog"$leapfrog_steps"_M"$M"_tmin"$corrector_tmin"_snr"$snr\
          --result_dir=$result_dir\
          --prior_model=$DEBLENDER/models/ncsnpp_probes_g_64_230604024652\
          --psf_fits=$DEBLENDER/data/F814w_WFC3UV_cropped_psf.fits\
          --psf_key=PRIMARY\
          --injection_test\
          --probes\
          --downsample=2\
          --dataset_path=$HOME/projects/rrg-lplevass/data/probes.h5\
          --dataset_channels 0\
          --dataset_key=galaxies\
          --dataset_id=$image_index\
          --dataset_channels_last\
          --observation_pixels=32\
          --observation_pixel_size=0.05\
          --n_obs=$n_obs\
          --model_pixels=64\
          --model_pixel_size=0.025\
          --super_sampling_factor=4\
          --slic_likelihood\
          --slic_model=$DEBLENDER/models/ncsnpp_hst_noise_flat_field_psf_f814w_wfc3uv_ssf2_32_flux_lt_0.013_230718211513\
          --slic_guidance_factor=$alpha\
          --noise_map=$DEBLENDER/data/connor_targets_flat_fields_noise_cutouts_flux_lt_0.013.npy\
          --em_iterations=$N\
          --corrector_iterations=$M\
          --walkers=$walkers\
          --batch_size=$batch_size\
          --corrector_tmin=$corrector_tmin\
          --corrector=HMC\
          --snr=$snr\
          --mass=$mass\
          --leapfrog_steps=$leapfrog_steps\
          --delta_logp_steps=2\
          --noise_indices ${noise_indices[@]}\ 
        image_index=$(($image_index+1))
        noise_index=$(($noise_index+$n_obs))
    done
done


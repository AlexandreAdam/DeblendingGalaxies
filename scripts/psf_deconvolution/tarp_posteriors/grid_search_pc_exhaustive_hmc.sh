#!/bin/bash
#SBATCH --array=1-2
#SBATCH --tasks=1
#SBATCH --cpus-per-task=1 # maximum cpu per task is 3.5 per gpus
#SBATCH --gres=gpu:1
#SBATCH --mem=32G			 # memory per node
#SBATCH --time=00-02:00		# time (DD-HH:MM)
#SBATCH --account=rrg-lplevass
#SBATCH --job-name=Deconvolution_tarp_hmc
#SBATCH --output=%x-%j.out

source $HOME/environments/milex/bin/activate

# Main setup
#mass_grid=(0.1 1 5)
#M_grid=(1 2 3)
#leapfrog_steps_grid=(2 3 5)
#snr_grid=(1e-2 5e-2 1e-1)
#corrector_tmin_grid=(0.1 0.5)

# test
mass_grid=(1)
M_grid=(1 3)
leapfrog_steps_grid=(2 5)
snr_grid=(1e-2)
corrector_tmin_grid=(0.1 0.5)

len_mass=${#mass_grid[@]}
len_leapfrog=${#leapfrog_steps_grid[@]}
len_M=${#M_grid[@]}
len_snr=${#snr_grid[@]}
len_corrector_tmin=${#corrector_tmin_grid[@]}

n_posteriors=10 
walkers=10
batch_size=10
alpha=100
n_obs=2
N=1000

total=$((len_mass*len_leapfrog*len_M*len_corrector_tmin*len_snr))
echo "Grid search over $total TARP experiments"

for mass in ${mass_grid[@]}
do
    for leapfrog_steps in ${leapfrog_steps_grid[@]}
    do
        for M in ${M_grid[@]}
        do
            for corrector_tmin in ${corrector_tmin_grid[@]}
            do 
                for snr in ${snr_grid[@]}
                do
                    echo "Posteriors for | mass: $mass | leapfrog_steps: $leapfrog_steps | M: $M | corrector_tmin: $corrector_tmin | snr: $snr"
                    image_index=0
                    noise_index=0
                    for ((i=1;i<=n_posteriors;i++))
                    do
                        end=$(($noise_index + $n_obs - 1))
                        noise_indices=$(seq $noise_index $end)
                        python $DEBLENDER/scripts/psf_deconvolution.py\
                          --experiment_name="tarp_posterior_hmc_exhaustive_"$i"_mass"$mass"_leapfrog"$leapfrog_steps"_M"$M"_tmin"$corrector_tmin"_snr"$snr\
                          --result_dir=$DEBLENDER/results/\
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
                          --noise_indexes ${noise_indices[@]}\ 
                        image_index=$(($image_index+1))
                        noise_index=$(($noise_index+$n_obs))
                    done
                done
            done
        done
    done
done


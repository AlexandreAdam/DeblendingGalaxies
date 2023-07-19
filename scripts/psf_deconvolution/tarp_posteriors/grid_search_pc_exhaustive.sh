#!/bin/bash
#SBATCH --array=1-10
#SBATCH --tasks=1
#SBATCH --cpus-per-task=1 # maximum cpu per task is 3.5 per gpus
#SBATCH --gres=gpu:1
#SBATCH --mem=32G			 # memory per node
#SBATCH --time=00-02:00		# time (DD-HH:MM)
#SBATCH --account=rrg-lplevass
#SBATCH --job-name=Deconvolution_tarp
#SBATCH --output=%x-%j.out

source $HOME/environments/milex/bin/activate

# Main setup
#alpha_grid=(50 100 200)
#N_grid=(1000 2000)
#M_grid=($(seq 0 4 12))
#n_observations_grid=(1 2)
#snr_grid=(1e-2 1e-1)
#len_posteriors=10 
#walkers=10
#batch_size=10

# test setup
alpha_grid=(50)
N_grid=(2000)
M_grid=(0)
n_observations_grid=(2)
snr_grid=(1e-2)
len_posteriors=10 
walkers=10
batch_size=10

len_alpha=${#alpha_grid[@]}
len_N=${#N_grid[@]}
len_M=${#M_grid[@]}
len_snr=${#snr_grid[@]}
len_obs=${#n_observations_grid[@]}
total=$((len_alpha*len_N*len_M*len_obs*len_snr))
echo "Grid search over $total TARP experiments"

#source $HOME/environments/milex/bin/activate
for alpha in ${alpha_grid[@]}
do
    for N in ${N_grid[@]}
    do
        for M in ${M_grid[@]}
        do
            for n_obs in ${n_observations_grid[@]}
            do
                for snr in ${snr_grid[@]}
                do
                    image_index=0
                    noise_index=0
                    for ((i=1;i<=len_posteriors;i++))
                    do
                        echo "alpha: $alpha | N: $N | M: $M | n_obs: $n_obs | SNR: $snr"
                        end=$(($noise_index + $n_obs - 1))
                        noise_indices=$(seq $noise_index $end)
                        python $DEBLENDER/scripts/psf_deconvolution.py\
                          --experiment_name="tarp_posterior_exhaustive_"$i"_ref"$image_index"_alpha"$alpha"_N"$N"_M"$M"_snr"$snr"_nobs"$n_obs\
                          --result_dir=$DEBLENDER/results/\
                          --prior_model=$DEBLENDER/models/ncsnpp_ct_g_220912024942\
                          --psf_fits=$DEBLENDER/data/F814w_WFC3UV_cropped_psf.fits\
                          --psf_key=PRIMARY\
                          --injection_test\
                          --probes\
                          --dataset_path=$HOME/projects/rrg-lplevass/data/probes.h5\
                          --dataset_channels 0\
                          --dataset_key=galaxies\
                          --dataset_id=$image_index\
                          --dataset_channels_last\
                          --observation_pixels=64\
                          --observation_pixel_size=0.05\
                          --model_pixels=128\
                          --model_pixel_size=0.025\
                          --super_sampling_factor=2\
                          --slic_likelihood\
                          --slic_model=$DEBLENDER/models/ncsnpp_hst_noise_flat_field_psf_f814w_wfc3uv_ssf2_flux_lt_0.013_230718121853\
                          --slic_guidance_factor=$alpha\
                          --noise_map=$DEBLENDER/data/connor_targets_flat_fields_noise_cutouts_flux_lt_0.013.npy\
                          --em_iterations=$N\
                          --walkers=$walkers\
                          --batch_size=$batch_size\
                          --noise_indexes ${noise_indices[@]}\ 
                        image_index=$(($image_index+1))
                        noise_index=$(($noise_index+$n_obs))
                    done
                done
            done
        done
    done
done


#!/bin/bash
#SBATCH --array=1-10
#SBATCH --tasks=1
#SBATCH --cpus-per-task=1 # maximum cpu per task is 3.5 per gpus
#SBATCH --gres=gpu:1
#SBATCH --mem=32G			 # memory per node
#SBATCH --time=00-02:00		# time (DD-HH:MM)
#SBATCH --account=rrg-lplevass
#SBATCH --job-name=Deconvolution
#SBATCH --output=%x-%j.out

alpha_grid=(50 100 200)
N_grid=(1000 2000)
M_grid=($(seq 0 4 12))
n_observations_grid=(1 2)
snr_grid=(1e-2 1e-1)

# Maximum index in dataset used for TARP (to choose reference)
max_index=10
# number of posteriors to run for a given index in the grid
len_posteriors=100

len_alpha=${#alpha_grid[@]}
len_N=${#N_grid[@]}
len_M=${#M_grid[@]}
len_snr=${#snr_grid[@]}
len_obs=${#n_observations_grid[@]}
total=$((len_alpha*len_N*len_M*len_obs*len_snr))
echo "Grid search over $total TARP experiments"

#source $HOME/environments/milex/bin/activate
j=0
image_index=0
noise_index=0
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
                    for ((i=1;i<=len_posteriors;i++))
                    do
                        echo "alpha: $alpha | N: $N | M: $M | n_obs: $n_obs | SNR: $snr"
                        echo $j
                        j=$(($j+1))
                        image_index=$(($image_index+1))
                        #python $DEBLENDER/scripts/psf_deconvolution.py\
                          #--experiment_name=tarp_posterior_ref{$image_index}_alpha{$alpha}_N{$N}_M{$}_snr{$snr}_nobs{$n_obs}\
                          #--psf_fits=$DEBLENDER/data/F814w_WFC3UV_cropped_psf.fits\
                          #--psf_key=PRIMARY\
                          #--injection_test\
                          #--dataset_path=$HOME/projects/rrg-lplevass/data/probes.h5\
                          #--dataset_channels 0\
                          #--dataset_key=galaxies\
                          #--dataset_id=$image_index\
                          #--dataset_channels_last\
                          #--observation_pixels=128\
                          #--observation_pixel_size=0.04\
                          #--noise_map ${args[@]}\ #TODO give a number of those based on n_obs make sure args will match accross jobs for same reference (obs must stay the same for a given posterior)
                          #--model_pixels=256\
                          #--downsample=1\
                          #--model_pixel_size=0.02\
                          #--noise_rms=2e-2\
                          #--super_sampling_factor=4\
                          #--diagonal_gaussian_likelihood\
                          #--result_dir=$DEBLENDER/results/\
                          #--checkpoints_dir=$DEBLENDER/models/ncsnpp_ct_g_220912024942\
                          #-N=4000\
                          #-W=100\
                          #-B=50\
                    done
                done
            done
        done
    done
done


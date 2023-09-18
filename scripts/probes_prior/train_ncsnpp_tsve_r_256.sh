#!/bin/bash
#SBATCH --tasks=1
#SBATCH --cpus-per-task=1 # maximum cpu per task is 3.5 per gpus
#SBATCH --gres=gpu:1
#SBATCH --mem=32G			     # memory per node
#SBATCH --time=02-23:00		# time (DD-HH:MM)
#SBATCH --account=rrg-lplevass
#SBATCH --job-name=Train_Probes256_r
#SBATCH --output=%x-%j.out
cp $HOME/scratch/probes.h5 $SLURM_TMPDIR/probes.h5
source $HOME/environments/milex/bin/activate
python $DEBLENDER/scripts/train_score_modelv2.py\
  --model_architecture=ncsnpp\
  --dataset_path=$SLURM_TMPDIR/probes.h5\
  --channels_last\
  --dataset_channels 1\
  --dataset_key=galaxies\
  --model_parameters=$DEBLENDER/scripts/probes_prior/ncsnpp_tsve_skirt512_256.json\
  --epochs=10000\
  --learning_rate=2e-5\
  --max_time=70\
  --batch_size=4\
  --logname_prefix=ncsnpp_tsve_probes_r_256\
  --model_dir=$DEBLENDER/models/\
  --checkpoints=10\
  --seed=42\
  --epoch_iterations=1000\
  --probes_preprocessing

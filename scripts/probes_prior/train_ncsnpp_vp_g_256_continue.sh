#!/bin/bash
#SBATCH --tasks=1
#SBATCH --cpus-per-task=1 # maximum cpu per task is 3.5 per gpus
#SBATCH --gres=gpu:1
#SBATCH --mem=32G			     # memory per node
#SBATCH --time=02-23:00		# time (DD-HH:MM)
#SBATCH --account=rrg-lplevass
#SBATCH --job-name=Train_Probes256_g
#SBATCH --output=%x-%j.out
cp $HOME/scratch/probes.h5 $SLURM_TMPDIR/probes.h5
source $HOME/environments/milex/bin/activate
python $DEBLENDER/scripts/train_score_modelv2.py\
  --model_architecture=ncsnpp\
  --dataset_path=$SLURM_TMPDIR/probes.h5\
  --channels_last\
  --dataset_channels 0\
  --dataset_key=galaxies\
  --checkpoints_directory=$DEBLENDER/models/ncsnpp_vp_probes_g_256_230824011809\
  --epochs=10000\
  --learning_rate=2e-5\
  --max_time=70\
  --batch_size=4\
  --checkpoints=10\
  --seed=42\
  --epoch_iterations=1000\


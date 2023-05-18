#!/bin/bash
#SBATCH --tasks=1
#SBATCH --cpus-per-task=1 # maximum cpu per task is 3.5 per gpus
#SBATCH --gres=gpu:1
#SBATCH --mem=32G			     # memory per node
#SBATCH --time=02-23:00		# time (DD-HH:MM)
#SBATCH --account=rrg-lplevass
#SBATCH --job-name=Train_SKIRT128_log_i
#SBATCH --output=%x-%j.out
source $HOME/environments/scope/bin/activate
python $DEBLENDER/scripts/train_score_nonlinear_sde.py\
  --model_architecture=ncsnpplog\
  --dataset_path=/home/aadam/scratch/skirt512_grizy.h5\
  --dataset_channels 2\
  --dataset_key=images\
  --model_parameters=$DEBLENDER/scripts/training_on_512_cutouts/ncsnpplog_skirt512_single_channel_128.json\
  --epochs=10000\
  --learning_rate=1e-4\
  --max_time=70\
  --batch_size=16\
  --logdir=$DEBLENDER/logs/\
  --logname_prefixe=ncsnpplog_skirt_i_128\
  --model_dir=$DEBLENDER/models/\
  --checkpoints=2\
  --minimum_flux=1e-3\
  --epoch_iterations=1000\
  --downsample=2

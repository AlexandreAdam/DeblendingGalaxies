#!/bin/bash
#SBATCH --tasks=1
#SBATCH --cpus-per-task=1 # maximum cpu per task is 3.5 per gpus
#SBATCH --gres=gpu:4
#SBATCH --mem=32G			     # memory per node
#SBATCH --time=02-23:00		# time (DD-HH:MM)
#SBATCH --account=rrg-lplevass
#SBATCH --job-name=Train_SKIRT512_r
#SBATCH --output=%x-%j.out
source $HOME/environments/scope/bin/activate
python $DEBLENDER/scripts/train_score_model.py\
  --model_architecture=ncsnpp\
  --dataset_path=/home/aadam/scratch/skirt512_grizy.h5\
  --dataset_channels 1\
  --dataset_key=images\
  --model_parameters=$DEBLENDER/scripts/training_on_512_cutouts/ncsnpp_skirt512_single_channel_256_2.json\
  --epochs=10000\
  --learning_rate=2e-5\
  --max_time=70\
  --batch_size=16\
  --logdir=$DEBLENDER/logs/\
  --logname_prefixe=ncsnpp_skirt_r_256_linear_2\
  --model_dir=$DEBLENDER/models/\
  --checkpoints=3\
  --seed=42\
  --epoch_iterations=1000\
  --downsample=1\
  --linear_preprocessing


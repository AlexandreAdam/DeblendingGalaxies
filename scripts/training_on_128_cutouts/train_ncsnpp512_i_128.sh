#!/bin/bash
#SBATCH --tasks=1
#SBATCH --cpus-per-task=1 # maximum cpu per task is 3.5 per gpus
#SBATCH --gres=gpu:1
#SBATCH --mem=32G			     # memory per node
#SBATCH --time=02-23:00		# time (DD-HH:MM)
#SBATCH --account=rrg-lplevass
#SBATCH --job-name=Train_SKIRT128_i
#SBATCH --output=%x-%j.out
cp $HOME/scratch/skirt128_grizy.h5 $SLURM_TMPDIR/skirt128_grizy.h5
source $HOME/environments/scope/bin/activate
python $DEBLENDER/scripts/train_score_model.py\
  --model_architecture=ncsnpp\
  --dataset_path=$SLURM_TMPDIR/skirt128_grizy.h5\
  --dataset_channels 2\
  --dataset_key=images\
  --model_parameters=$DEBLENDER/scripts/training_on_128_cutouts/ncsnpp_skirt512_128.json\
  --epochs=10000\
  --learning_rate=2e-5\
  --max_time=70\
  --batch_size=4\
  --logname_prefixe=ncsnpp_skirt_i_128\
  --model_dir=$DEBLENDER/models/\
  --checkpoints=10\
  --seed=42\
  --dynamic_range=1e5\
  --epoch_iterations=1000\

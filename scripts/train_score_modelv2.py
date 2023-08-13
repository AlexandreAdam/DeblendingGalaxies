from score_models import ScoreModel
from definitions import microjy_preprocessing
import json
import numpy as np
import torch
import h5py

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
LOG10 = np.log(10.)


class Dataset(torch.utils.data.Dataset):
    def __init__(self, path_to_h5, key, channels, channels_last=False, device=DEVICE):
        self.filepath = path_to_h5
        self.device = device
        self.key = key
        self.channels = channels
        self.channels_last = channels_last
        self.hf = h5py.File(self.filepath, "r")
        self.size = self.hf[self.key].shape[0]

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        if self.channels_last:
            im = torch.tensor(self.hf[self.key][index, :, :, self.channels]).to(self.device)
            # put channels first for Conv2D score model
            return torch.permute(im, (2, 0, 1))
        else:
            return torch.tensor(self.hf[self.key][index, self.channels]).to(self.device)


def main(args):
    with open(args.model_parameters, "r") as f:
        hyperparameters = json.load(f)
    model = ScoreModel(args.model_architecture, **hyperparameters)
    dataset = Dataset(args.dataset_path, args.dataset_key, args.dataset_channels, channels_last=args.channels_last, device=DEVICE)
    preprocessing = microjy_preprocessing
    
    model.fit(
            dataset, 
            preprocessing_fn=preprocessing,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            ema_decay=args.ema_decay,
            batch_size=args.batch_size,
            max_time=args.max_time ,
            warmup=args.warmup,
            clip=args.clip,
            checkpoints_directory=args.checkpoints_directory,
            model_checkpoint=args.model_checkpoint,
            checkpoints=args.checkpoints,
            models_to_keep=args.models_to_keep,
            seed=args.seed,
            logname=args.logname,
            logdir=args.logdir,
            n_iterations_in_epoch=args.epoch_iterations,
            logname_prefix=args.logname_prefix
            )


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--model_architecture",     required=True,                  help="Either 'ddpm' or 'ncsnpp'")
    parser.add_argument("--dataset_path",       required=True)
    parser.add_argument("--dataset_key",        required=True)
    parser.add_argument("--dataset_channels",   nargs="+", required=True, type=int, help="Channels of the dataset to use. ")
    parser.add_argument("--model_id",           default="none",                     help="The script will search in provided model_dir argument for model_id and load checkpoint if it exists.")
    parser.add_argument("--model_checkpoint",   default=None,       type=int,       help="Index of the checkpoint to load.")

    # Model parameters
    parser.add_argument("--model_parameters",               required=True,                  help="Path to model parameter json file.")

    # Optimization params
    parser.add_argument("--epochs",                         default=10,     type=int,       help="Number of epochs for training.")
    parser.add_argument("--epoch_iterations",               default=None,   type=int,       help="Number of iterations to do in an epoch")
    parser.add_argument("--learning_rate",                  default=2e-5,   type=float,     help="Initial learning rate.")
    parser.add_argument("--max_time",                       default=np.inf, type=float,     help="Time allowed for the training, in hours.")
    parser.add_argument("--ema_decay",                      default=0.9999, type=float)
    parser.add_argument("--warmup",                         default=5000,   type=int,       help="Warmup the learning up to the target learning rate over this amount of iterations")
    parser.add_argument("--clip",                           default=0.,     type=float,     help="Gradient clipping")

    # Training set params
    parser.add_argument("--batch_size",             default=1,      type=int,       help="Number of images in a batch.")
    parser.add_argument("--downsample",             default=0,   type=int,       help="Average pooling, if zero, no downsampling, if 1, then downsample by a factor of 2, etc. ")
    parser.add_argument("--shuffle",                action="store_true",            help="Shuffle the dataset, not recommended for large hdf5 datasets, will slow down training tremedously")
    parser.add_argument("--channels_last",          action="store_true",            help="Wether the data was saved in channels_last format. For backward compatibility mainly. ")

    # logs
    parser.add_argument("--logdir",             default="None",                     help="Path of logs directory. Default if None, no logs recorded.")
    parser.add_argument("--logname",            default=None,                       help="Overwrite name of the log with this argument")
    parser.add_argument("--logname_prefixe",    default="score_model",                  help="If name of the log is not provided, this prefix is prepended to the date")
    parser.add_argument("--model_dir",          default="None",                     help="Path to the directory where to save models checkpoints.")
    parser.add_argument("--checkpoints",        default=10, type=int,               help="Save a checkpoint of the models each {%} epoch.")
    parser.add_argument("--models_to_keep",     default=2,  type=int,               help="Only keep 3 best model, on top of the last checkpoint")

    # Reproducibility params
    parser.add_argument("--seed",                   default=None,   type=int,       help="Random seed for numpy and tensorflow.")

    args = parser.parse_args()
    main(args)

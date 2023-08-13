from kernel_slic import KernelSLIC, effective_kernel
import json
import numpy as np
import torch
import os
from forward_model_old import make_forward_model
from torch.func import jacrev
from astropy.io import fits

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



class Dataset(torch.utils.data.Dataset):
    """
    Those should be dark images from the target instrument
    """
    def __init__(self, path_to_npy, device=DEVICE):
        self.filepath = path_to_npy
        self.dataset = np.load(path_to_npy)
        self.size = self.dataset.shape[0]
        self.device = device

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        return torch.tensor(self.dataset[index]).float().to(self.device)[None]


def random_crop(images, new_shape):
    """
    Performs a random crop on a batch of images.

    Parameters:
    images (torch.Tensor): a 4D tensor with shape [B, C, H, W]
    new_shape (int): the desired height and width of the cropped images

    Returns:
    torch.Tensor: a 4D tensor with shape [B, C, new_shape, new_shape]
    """
    batch, channels, height, width = images.shape
    start_x = torch.randint(0, width - new_shape, (batch, ), dtype=torch.long)
    start_y = torch.randint(0, height - new_shape, (batch, ), dtype=torch.long)
    
    cropped_images = torch.empty((batch, channels, new_shape, new_shape), dtype=images.dtype, device=images.device)
    for i, (img, x, y) in enumerate(zip(images, start_x, start_y)):
        cropped_images[i] = img[:, y:y+new_shape, x:x+new_shape]

    return cropped_images

def main(args):
    with open(args.model_parameters, "r") as f:
        hyperparameters = json.load(f)
    """
    sigma_min and sigma_max are not free to be defined.
    Instead, those should be inferred from the prior model used during inference, since we are trying to match 
    the time variable in the prior SDE. 
    """
    model_name = os.path.split(args.prior_model)[-1]
    with open(os.path.join(args.prior_model, "model_hparams.json"), "r") as f:
        prior_hyperparameters = json.load(f)
    # Get SDE parameters from prior
    for key in ["sigma_min", "sigma_max", "beta_min", "beta_max"]:
        v = prior_hyperparameters.get(key, None)
        if v is not None:
            hyperparameters.update({key: v})
    T = prior_hyperparameters.get("T", 1.0)
    epsilon = prior_hyperparameters.get("epsilon", 1e-5)
    hyperparameters["T"] = T
    hyperparameters["epsilon"] = epsilon
    hyperparameters["prior_model"] = model_name

    with fits.open(args.psf_fits) as data:
        psf = data[args.psf_key].data.astype(np.float32)[None] # add the channel dimension, a single channel for now.
    hyperparameters["psf_file"] = args.psf_fits
    hyperparameters["psf_key"] = args.psf_key
    
    forward_model = make_forward_model(args, psf)
    # Get the effective kernel for this forward model
    idim = [1, args.model_pixels, args.model_pixels]
    odim = [1, args.observation_pixels, args.observation_pixels]
    kernel = effective_kernel(forward_model, idim, odim, 0, args.observation_pixels//2, args.observation_pixels//2) 

    model = KernelSLIC(kernel, idim, forward_model, model=args.model_architecture.lower(), **hyperparameters)
    dataset = Dataset(args.dataset_path, device=DEVICE)
    model.fit(
            dataset, 
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
    parser.add_argument("--model_architecture", required=True,                      help="Either 'ddpm' or 'ncsnpp'")
    parser.add_argument("--dataset_path",       required=True,                      help="Path to .npy dataset")
    parser.add_argument("--model_id",           default="none",                     help="The script will search in provided model_dir argument for model_id and load checkpoint if it exists.")
    parser.add_argument("--model_checkpoint",   default=None, type=int,             help="Index of the checkpoint to load.")
    parser.add_argument("--psf_fits",           required=True,                      help="Path to PSF fits file")
    parser.add_argument("--psf_key",            required=True,                      help="Key to the PSF in the fits file")

    parser.add_argument("--prior_model",        required=True,                      help="Path to prior model, we mainly need it's sigma_min and sigma_max. The name"
                                                                                         "of the prior model is encoded in the the SLIC model params for future reference, as well "
                                                                                         "as the forward model params.")
    parser.add_argument("--observation_pixels", required=True,    type=int,          help="Has to correspond to the size of the noise images, otherwise the script will break. ")
    parser.add_argument("--observation_pixel_size", required=True, type=float,       help="Pixel size for the fake observation, in arcseconds. Should correspond "
                                                                                         "to the pixel size of the noise dataset used (e.g. for HST this should be roughly 0.04 arcseconds.")
    parser.add_argument("--model_pixels",       required=True,     type=int,         help="Number of pixels on a side for the (prior) model ")
    parser.add_argument("--model_pixel_size",   required=True,  type=float,          help="Size of a pixel for the (prior) model, in arcseconds")
    parser.add_argument("--super_sampling_factor", default=2,   type=int,           help="Factor by which the PSF is super sampled. ")
    parser.add_argument("--zero_padding",       default=0,      type=int,            help="Zero padding in the forward model. Default is no zero-padding")

    # Model parameters
    parser.add_argument("--model_parameters",   required=True,                      help="Path to model hyperparameter json file.")

    # Optimization params
    parser.add_argument("--epochs",             default=10, type=int,               help="Number of epochs for training.")
    parser.add_argument("--epoch_iterations",   default=None, type=int,             help="Number of iterations to do in an epoch")
    parser.add_argument("--learning_rate",      default=2e-5, type=float,           help="Initial learning rate.")
    parser.add_argument("--max_time",           default=np.inf, type=float,         help="Time allowed for the training, in hours.")
    parser.add_argument("--ema_decay",          default=0.9999, type=float)
    parser.add_argument("--warmup",             default=5000, type=int,             help="Warmup the learning up to the target learning rate over this amount of iterations")
    parser.add_argument("--clip",               default=0., type=float,             help="Gradient clipping")

    # Training set params
    parser.add_argument("--batch_size",         default=1, type=int,                help="Number of images in a batch.")

    # logs
    parser.add_argument("--logdir",             default=None,                     help="Path of logs directory. Default if None, no logs recorded.")
    parser.add_argument("--logname",            default=None,                       help="Overwrite name of the log with this argument")
    parser.add_argument("--logname_prefix",    default="score_model",              help="If name of the log is not provided, this prefix is prepended to the date")
    parser.add_argument("--checkpoints_directory", default=None)
    parser.add_argument("--model_dir",          default="None",                     help="Path to the directory where to save models checkpoints.")
    parser.add_argument("--checkpoints",        default=2, type=int,               help="Save a checkpoint of the models each {%} epoch.")
    parser.add_argument("--models_to_keep",     default=3, type=int,               help="Only keep 3 best model, on top of the last checkpoint")

    # Reproducibility params
    parser.add_argument("--seed",               default=None, type=int,             help="Random seed for numpy and torch")

    args = parser.parse_args()
    main(args)

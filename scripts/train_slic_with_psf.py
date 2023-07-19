from score_models import ScoreModel, NCSNpp, DDPM
from torch.utils.data import DataLoader
from torch.func import vjp
from datetime import datetime
from tqdm import tqdm
import time
import json
import numpy as np
import torch
import os
from glob import glob
import re
from torch_ema import ExponentialMovingAverage
from forward_model_old import make_forward_model
from astropy.io import fits

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def sliced_score_matching_loss(score_fn, samples, n_cotangent_vectors=1, device=DEVICE, noise_type="gaussian"):
    """
    Score matching loss with the Hutchinson trace estimator trick. See Theorem 1 of
    Hyvärinen (2005). Estimation of Non-Normalized Statistical Models by Score Matching,
    (https://www.jmlr.org/papers/volume6/hyvarinen05a/hyvarinen05a.pdf).

    We implement an unbiased estimator of this loss with reduced variance reported in
    Y. Song et al. (2019). A Scalable Approach to Density and Score Estimation
    (https://arxiv.org/abs/1905.07088).

    Inspired from the official implementation of Sliced Score Matching at https://github.com/ermongroup/sliced_score_matching
    We also implement the weighting scheme for NCSN (Song & Ermon 2019 https://arxiv.org/abs/1907.05600)
    """
    if noise_type not in ["gaussian", "rademacher"]:
        raise ValueError("noise_type has to be either 'gaussian' or 'rademacher'")
    B, *D = samples.shape
    # duplicate noisy samples across the number of particle for the Hutchinson trace estimator
    samples = torch.tile(samples, [n_cotangent_vectors, *[1]*len(D)])
    samples.requires_grad_(True)

    # sample cotangent vectors
    vectors = torch.randn_like(samples)
    if noise_type == 'rademacher':
        vectors = vectors.sign()

    score, vjp_func = vjp(score_fn, samples)
    trace_estimate = vectors * vjp_func(vectors)
    loss = torch.sum(trace_estimate + score**2) / B / n_cotangent_vectors
    return loss


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
    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    with open(args.model_parameters, "r") as f:
        hyperparameters = json.load(f)

    """
    sigma_min and sigma_max are not free to be defined by the user.
    Instead, those should be inferred from the prior model used during inference, since we are trying to match 
    the time variable in the prior SDE. 
    """
    model_name = os.path.split(args.prior_model)[-1]
    with open(os.path.join(args.prior_model, "model_hparams.json"), "r") as f:
        prior_hyperparameters = json.load(f)
    sigma_min = prior_hyperparameters["sigma_min"]
    sigma_max = prior_hyperparameters["sigma_max"]
    hyperparameters["sigma_min"] = sigma_min
    hyperparameters["sigma_max"] = sigma_max
    hyperparameters["prior_model"] = model_name

    # TODO support multiple channels
    # Todo possibly convert pixel size from pc in Connor B. fits file to arcsec using a user specified Hubble constant and redshift
    # TODO, would it make sense to sample different PSF and condition the SLIC model on it?????
    with fits.open(args.psf_fits) as data:
        psf = data[args.psf_key].data.astype(np.float32)[None] # add the channel dimension, a single channel for now.
    hyperparameters["psf_file"] = args.psf_fits
    hyperparameters["psf_key"] = args.psf_key
    forward_model = make_forward_model(args, psf)

    # Define the architecture of the SLIC model
    if args.model_architecture.lower() == "ddpm":
        net = DDPM(**hyperparameters).to(DEVICE)
    elif args.model_architecture.lower() == "ncsnpp":
        net = NCSNpp(**hyperparameters).to(DEVICE)
    else:
        raise ValueError

    model = ScoreModel(net, sigma_min=sigma_min, sigma_max=sigma_max)
    dataset = Dataset(args.dataset_path, device=DEVICE)
    dataset = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    ema = ExponentialMovingAverage(model.parameters(), decay=args.ema_decay)

    def loss_fn(x):
        """
        Denoising or sliced score matching loss on the distribution implicitly defined by
        taking a noise sample from a telescope dark image (t=0)
        and adding noise (at temperature t) through a user specified forward model.

        This model will be specialized to this specific (linear) forward model, and won't necessarily be a
        good approximation in general
        """
        B = x.shape[0]
        t = torch.rand(B).to(DEVICE)
        mu, sigma = model.sde.marginal_prob_scalars(t)
        mu, sigma = mu.view(B, 1, 1, 1), sigma.view(B, 1, 1, 1)
        z = forward_model(torch.randn([B, 1, args.model_pixels, args.model_pixels]).to(DEVICE)) # noise propagated through forward model
        perturbed_x = mu * x + sigma * z
        if args.loss.lower() == "ssm":
            return sliced_score_matching_loss(score_fn=lambda x: model(t, x), samples=perturbed_x, noise_type=args.hutchinson_noise_type)
        elif args.loss.lower() == "dsm":
            return torch.sum((z + sigma * model(t, perturbed_x))**2) / B

    # ==== Take care of where to write logs and stuff =================================================================
    if args.model_id.lower() != "none":
        logname = args.model_id
    elif args.logname is not None:
        logname = args.logname
    else:
        logname = args.logname_prefixe + "_" + datetime.now().strftime("%y%m%d%H%M%S")
    if args.logdir.lower() != "none":
        logdir = os.path.join(args.logdir, logname)
        if not os.path.isdir(logdir):
            os.mkdir(logdir)
    # ===== Make sure directory and checkpoint manager are created to save model ===================================
    if args.model_dir.lower() != "none":
        checkpoints_dir = os.path.join(args.model_dir, logname)
        if not os.path.isdir(checkpoints_dir):
            os.mkdir(checkpoints_dir)
            with open(os.path.join(checkpoints_dir, "script_params.json"), "w") as f:
                json.dump(vars(args), f, indent=4)
            with open(os.path.join(checkpoints_dir, "model_hparams.json"), "w") as f:
                json.dump(hyperparameters, f, indent=4)
        save_checkpoint = True
        # ======= Load model if model_id is provided ===============================================================
        paths = glob(os.path.join(checkpoints_dir, "checkpoint*.pt"))
        opt_paths = glob(os.path.join(checkpoints_dir, "optimizer*.pt"))
        checkpoints = [int(re.findall('[0-9]+', os.path.split(path)[-1])[-1]) for path in paths]
        scores = [float(re.findall('([0-9]{1}.[0-9]+e[+-][0-9]{2})', os.path.split(path)[-1])[-1]) for path in paths]
        if args.model_id.lower() != "none" and checkpoints != []:
            if args.model_checkpoint is not None:
                model.load_state_dict(torch.load(paths[checkpoints == args.model_checkpoint], map_location=DEVICE))
                optimizer.load_state_dict(torch.load(opt_paths[checkpoints == args.model_checkpoint], map_location=DEVICE))
                print(f"Loaded checkpoint {args.model_checkpoint} of {args.model_id}")
                lastest_checkpoint = args.model_checkpoint
            else:
                model.load_state_dict(torch.load(paths[np.argmax(checkpoints)], map_location=DEVICE))
                optimizer.load_state_dict(torch.load(opt_paths[np.argmax(checkpoints)], map_location=DEVICE))
                print(f"Loaded checkpoint {max(checkpoints)} of {args.model_id}")
                lastest_checkpoint = max(checkpoints)
        else:
            lastest_checkpoint = 0
    else:
        save_checkpoint = False

    # ====== Training loop ============================================================================================
    best_loss = np.inf
    patience = args.patience
    step = 0
    global_start = time.time()
    estimated_time_for_epoch = 0
    out_of_time = False
    for epoch in tqdm(range(args.epochs)):
        if (time.time() - global_start) > args.max_time * 3600 - estimated_time_for_epoch:
            break
        epoch_start = time.time()
        time_per_step_epoch_mean = 0
        cost = 0
        for batch, x in enumerate(dataset):
            # Make sure noise vectors match observation space of the forward model
            if x.shape[2] > args.observation_pixels:
                x = random_crop(x, args.observation_pixels)
            elif x.shape[2] < args.observation_pixels:
                raise ValueError("Data is too small for the forward model considered")
            start = time.time()
            optimizer.zero_grad()
            loss = loss_fn(x)
            loss.backward()
            # warmup learning rate
            if step <= args.warmup:
                for g in optimizer.param_groups:
                    g['lr'] = args.learning_rate * np.minimum(step / args.warmup, 1.0)
            # gradient clipping
            if args.clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.clip)
            optimizer.step()
            ema.update()
            # ========== Summary and logs ==============================================================================
            _time = time.time() - start
            time_per_step_epoch_mean += _time
            cost += float(loss)
            step += 1
            if args.epoch_iterations is not None:
                if batch >= args.epoch_iterations:
                    break

        time_per_step_epoch_mean /= len(dataset)
        cost /= len(dataset)

        if np.isnan(cost):
            print("Training broke the Universe")
            break
        if cost < (1 - args.tolerance) * best_loss:
            best_loss = cost
            patience = args.patience
        else:
            patience -= 1
        if (time.time() - global_start) > args.max_time * 3600:
            out_of_time = True
        if save_checkpoint:
            if epoch % args.checkpoints == 0 or patience == 0 or epoch == args.epochs - 1 or out_of_time:
                lastest_checkpoint += 1
                with open(os.path.join(checkpoints_dir, "score_sheet.txt"), mode="a") as f:
                    f.write(f"{lastest_checkpoint} {cost}\n")
                with ema.average_parameters():  # save EMA parameters
                    torch.save(model.state_dict(), os.path.join(checkpoints_dir, f"checkpoint_{cost:.4e}_{lastest_checkpoint:03d}.pt"))
                torch.save(optimizer.state_dict(), os.path.join(checkpoints_dir, f"optimizer_{cost:.4e}_{lastest_checkpoint:03d}.pt"))
                checkpoints.append(lastest_checkpoint)
                scores.append(cost)
                print("Saved checkpoint for step {}".format(step))
            if len(checkpoints) >= args.models_to_keep + 1:
                # remove the worst score checkpoint (excluding the one we just saved)
                index_to_delete = np.argmax(scores[:-1])
                os.remove(os.path.join(checkpoints_dir, f"checkpoint_{scores[index_to_delete]:.4e}_{checkpoints[index_to_delete]:03d}.pt"))
                os.remove(os.path.join(checkpoints_dir, f"optimizer_{scores[index_to_delete]:.4e}_{checkpoints[index_to_delete]:03d}.pt"))
                del scores[index_to_delete]
                del checkpoints[index_to_delete]
        if patience == 0:
            print("Reached patience")
            break
        if out_of_time:
            break
        if epoch > 0:  # First epoch is always very slow and not a good estimate of an epoch time.
            estimated_time_for_epoch = time.time() - epoch_start
    print(f"Finished training after {(time.time() - global_start) / 3600:.3f} hours.")


if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--model_architecture", required=True,                      help="Either 'ddpm' or 'ncsnpp'")
    parser.add_argument("--dataset_path",       required=True,                      help="Path to .npy dataset")
    parser.add_argument("--model_id",           default="none",                     help="The script will search in provided model_dir argument for model_id and load checkpoint if it exists.")
    parser.add_argument("--model_checkpoint",   default=None, type=int,             help="Index of the checkpoint to load.")
    parser.add_argument("--psf_fits",           required=True,                      help="Path to PSF fits file")
    parser.add_argument("--psf_key",            required=True,                      help="Key to the PSF in the fits file")
    parser.add_argument("--loss",               default="dsm",                      help="Either dsm for Denoising Score Matching or ssm for sliced score matching")

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
    parser.add_argument("--patience",           default=np.inf, type=int,           help="Number of step at which training is stopped if no improvement is recorder.")
    parser.add_argument("--tolerance",          default=0, type=float,              help="Current score <= (1 - tolerance) * best score => reset patience, else reduce patience.")
    parser.add_argument("--max_time",           default=np.inf, type=float,         help="Time allowed for the training, in hours.")
    parser.add_argument("--ema_decay",          default=0.9999, type=float)
    parser.add_argument("--warmup",             default=5000, type=int,             help="Warmup the learning up to the target learning rate over this amount of iterations")
    parser.add_argument("--clip",               default=0., type=float,             help="Gradient clipping")
    parser.add_argument("--hutchinson_noise_type", default="rademacher",            help="Distribution of noise for the Hutchinson trace estimator. Default is Rademacher")

    # Training set params
    parser.add_argument("--batch_size",         default=1, type=int,                help="Number of images in a batch.")

    # logs
    parser.add_argument("--logdir",             default="None",                     help="Path of logs directory. Default if None, no logs recorded.")
    parser.add_argument("--logname",            default=None,                       help="Overwrite name of the log with this argument")
    parser.add_argument("--logname_prefixe",    default="score_model",              help="If name of the log is not provided, this prefix is prepended to the date")
    parser.add_argument("--model_dir",          default="None",                     help="Path to the directory where to save models checkpoints.")
    parser.add_argument("--checkpoints",        default=2, type=int,               help="Save a checkpoint of the models each {%} epoch.")
    parser.add_argument("--models_to_keep",     default=3, type=int,               help="Only keep 3 best model, on top of the last checkpoint")

    # Reproducibility params
    parser.add_argument("--seed",               default=None, type=int,             help="Random seed for numpy and torch")

    args = parser.parse_args()
    main(args)

from score_models import NCSNpp
from functorch import grad, vmap
from torch.nn import functional as F
from definitions import interpolate, inverse_proprocessing
import json
import numpy as np
import torch
import os
from glob import glob
from astropy.io import fits
import h5py
import re
from tqdm import tqdm

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")

# total number of slurm workers detected
# defaults to 1 if not running under SLURM
N_WORKERS = int(os.getenv('SLURM_ARRAY_TASK_COUNT', 1))

# this worker's array index. Assumes slurm array job is zero-indexed
# defaults to one if not running under SLURM
THIS_WORKER = int(os.getenv('SLURM_ARRAY_TASK_ID', 1))


def make_forward_model(args, psf):
    """
    Takes in argument specifying the size of the model (number of pixels, size of the pixels)
    and the target size (number of pixels in the image, size of the pixels). It also takes in a psf
    specified by the user for the deconvolution.

    The operator returned is a functional form of the matrix A that appears in the linear inverse problem
        y = Ax + "noise"

    A is composed of
        0) Possible zero-padding. Default is no padding in the script.
        1) Interpolation of the model to twice the size of the observed image
        2) PSF convolution
        3) Pixelization (2d Average pooling with stride=2)

    Note that A does not include any preprocessing of the generated image. This must be done explicitly in the likelihood
        function. It is important to keep track of such a transformation (which might be non-linear) in the gradient of
        the likelihood function, thus this function is not the right place for it.

    Assumes psf is a 3D numpy array, with channels first. I assume PSF has the same number of channels as the observation.
        #TODO: support more than one channels in the script. Requires either a prior trained on all channels or separate priors for each

    args must have the following elements in its namespace:
        dynamic_range: The dynamic range of the prior, which specifies how we recover micro Jansky units. This parameter
            is written in the model hyperparameter json file.
        model_pixels: An integer that specifies the number of pixels on a side. This
            is user defined, although there is some restrictions: the value should be at least a multiple of 2^{levels},
            where levels is the number of downsampling layers in the score model. levels is equal to the length of
            the list "ch_mult" in the model hyperparameter json file.
        model_pixel_size: The size of a model pixel, in arcseconds.
        observation_pixels: An integer that specifies the number of pixel of our target observation
        observation_pixel_size: The size of a pixel in our observation, in arcseconds.
        zero_padding: Number of zeros padded to each side of the model.

    # TODO support a more sophisticated coordinate systems with astropy WCS.
    """
    C, H, W = psf.shape
    psf = torch.tensor(psf).to(DEVICE).view(C, 1, H, W) # reshape to a convolution kernel [channel_out, channels_in/groups, H, W]
    batched_interpolation = vmap(interpolate, in_dims=(0, None))  # only batch over the images

    # TODO support a shift of the ccordinates
    # define target coordinates at the super sampling resolution of the psf
    fov = args.observation_pixel_size * args.observation_pixels
    x = torch.linspace(-1, 1, args.super_sampling_factor*args.observation_pixels).float() * fov / 2
    x, y = torch.meshgrid(x, x, indexing="ij")  # TODO make this coherent with WCS
    # Transform these coordinates into model pixel indices
    _min = - args.model_pixel_size * (args.model_pixels + args.zero_padding) / 2
    i_coord = (x - _min) / args.model_pixel_size
    j_coord = (y - _min) / args.model_pixel_size
    coordinates = torch.stack([i_coord, j_coord], dim=0).to(DEVICE)
    def A(x):
        x = F.pad(x, pad=[args.zero_padding]*4, mode="constant", value=0.)
        x = batched_interpolation(x, coordinates)
        x = F.conv2d(x, psf, groups=C)
        x = F.avg_pool2d(x, kernel_size=args.super_sampling_factor, stride=args.super_sampling_factor)
        return x
    return A


def main(args):
    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    # Load model
    model_name = os.path.split(args.checkpoints_dir)[-1]
    with open(os.path.join(args.checkpoints_dir, "model_hparams.json"), "r") as f:
        hyperparameters = json.load(f)
    model = NCSNpp(**hyperparameters).to(DEVICE)
    paths = glob(os.path.join(args.checkpoints_dir, "*.pt"))
    checkpoints = [int(re.findall('[0-9]+', os.path.split(path)[-1])[-1]) for path in paths]
    model.eval()
    for p in model.parameters(): p.requires_grad = False  # being extra careful for some reasons
    if args.model_checkpoint is not None:
        model.load_state_dict(torch.load(paths[checkpoints == args.model_checkpoint], map_location=DEVICE))
        print(f"Loaded checkpoint {args.model_checkpoint} of {model_name}")
    else:
        model.load_state_dict(torch.load(paths[np.argmax(checkpoints)], map_location=DEVICE))
        print(f"Loaded checkpoint {max(checkpoints)} of {model_name}")
    model = torch.nn.DataParallel(model, device_ids=list(range(torch.cuda.device_count())))

    # Hack the VESDE in the model for readability
    sde = model.module.sde
    sigma_min = sde.sigma_min
    sigma_max = sde.sigma_max
    def sigma(t): # scale of the marginal prob. distiribution
        return sigma_min * (sigma_max / sigma_min)**t
    def g(t): # diffusion coefficient of the VESDE
        return sigma(t) * np.sqrt(2 * (np.log(sigma_max) - np.log(sigma_min)))

    if args.real_data:
        # Would load the data and overwrite the argument name space with fits header values
        raise NotImplementedError("Real data mode not yet supported")

    if args.slic_likelihood:
        # Would load the model
        raise NotImplementedError("SLIC mode not yet supported")

    # TODO support multiple channels
    # Todo possibly convert pixel size from pc in Connor B. fits file to arcsec using a user specified Hubble constant and redshift
    with fits.open(args.psf_fits) as data:
        psf = data[args.psf_key].data[None] # add the channel dimension, a single channel for now.

    forward_model = make_forward_model(args, psf)

    if args.injection_test:
        with h5py.File(args.dataset_path, "r") as hf:
            reference_profile = torch.tensor(hf[args.dataset_key][args.dataset_id]).to(DEVICE)[None, None] # single channel for now
        observation = forward_model(reference_profile)
        # if args.slic_likelihood:
        #     print("Sampling a noise realisation from the SLIC model")
        #     OBSERVATION += slic_model.sample(OBSERVATION.shape, N=args.N)
        # else:
        observation += torch.randn_like(observation) * args.noise_rms


    if args.diagonal_gaussian_likelihood:
        def convolved_likelihood(x, t, sigma_n=args.noise_rms):
            var = (sigma_n**2 + sigma(t)**2).view(*[1]*len(observation.shape))
            # TODO include invert_preprocessing here to make sure gradient picks it up
            y_hat = forward_model(x[None])
            ll = torch.sum(-0.5 * torch.square(observation - y_hat) / var)
            return ll
        convolved_likelihood_gradient = vmap(grad(convolved_likelihood, argnums=0))  # now take in batched inputs and return score
    elif args.pseudo_inverse_gaussian_likelihood:
        raise NotImplementedError("pseudo inverse likelihood not yet supported")
    elif args.slic_likelihood:
        raise NotImplementedError("SLIC not yet supported")

    if args.from_prior:
        def score_fn(x, t):
            B, *D = x.shape
            prior_score = model(x, t) / sigma(t).view(B, *[1]*len(D))
            return prior_score
    else:
        # Sample from the posterior
        def score_fn(x, t):
            B, *D = x.shape
            prior_score = model(x, t) / sigma(t).view(B, *[1]*len(D))
            likelihood_score = convolved_likelihood_gradient(x, t)
            return prior_score + likelihood_score

    def euler_maruyama_step(x, t, dt):
        t += dt
        x_mean = x - g(t) ** 2 * score_fn(x, t) * dt
        z = torch.randn_like(x)
        x = x_mean + g(t) * z * np.sqrt(-dt)
        return x_mean, x, t

    # TODO add a corrector step function like unadjusted Langevin, MALA, HMC, etc.

    # Now we do the hard work
    filename = os.path.join(args.result_dir, args.experiment_name + f"_{THIS_WORKER}" + ".h5")
    print("Solving the posterior...")
    with h5py.File(filename, "w") as hf:
        hf["observation"] = observation.cpu().numpy().astype(np.float32).squeeze()
        # TODO add a bunch of relevant info here for reproducibility
        # hf["observation"].attrs["units"] = 'micro Jy'
        # hf["observation"].attrs["pixel_size"] = 'micro Jy'
        hf["psf"] = psf.astype(np.float32).squeeze()
        # TODO support multiple channels
        hf.create_dataset("model", [args.W, 1, args.model_pixels, args.model_pixels], dtype=np.float32)
        hf["model"].attrs["posterior_sample"] = not args.from_prior # make sure we write somewhere if this is a posterior sample or not
        # TODO renormalize noise to make sure 10^x does not explode, add corresponding drift from Ito's lemma in the SDE
        for n in range(args.W // args.B):
            with torch.no_grad(): # important to add this context, otherwise Pytorch construct a graph through the sampling procedure.
                # TODO add the possibly of conditioning on a user defined guess, and a user specified "high temperature regime"
                #  x = guess + torch.randn(args.B, 1, args.model_pixels, args.model_pixels).to(DEVICE) * sigma(args.T)
                x = torch.randn(args.B, 1, args.model_pixels, args.model_pixels).to(DEVICE) * sigma(1.) # TODO add channels
                dt = -1. / args.N
                t = torch.ones(args.B).to(DEVICE)
                for _ in tqdm(range(args.N)):
                    x_mean, x, t = euler_maruyama_step(x, t, dt)
            hf["model"][n * args.B: (n+1) * args.B] = x_mean.cpu().numpy().astype(np.float32)
        # Do the last batch if there is one
        if args.W % args.B > 0:
            with torch.no_grad():
                x = torch.randn(args.W % args.B, 1, args.model_pixels, args.model_pixels).to(DEVICE) * sigma(1.)  # TODO add channels
                t = torch.ones(args.B).to(DEVICE)
                for _ in tqdm(range(args.N)):
                    x_mean, x, t = euler_maruyama_step(x, t, dt)
            hf["model"][(n+1) * args.B:] = x_mean.cpu().numpy().astype(np.float32)


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--experiment_name",    default="",                         help="Name of the output files")
    parser.add_argument("--psf_fits",           required=True,                       help="Path to PSF fits file")
    parser.add_argument("--psf_key",            required=True,                       help="Key to the PSF in the fits file")

    # REAL DATA MODEL TODO write the code for this mode
    # With real data, we only have access to the observation itself and the PSF
    parser.add_argument("--real_data",          action="store_true",                help="Real data mode. This mode requires a "
                                                                                         "fits file for the observation and a fits file for the PSF. "
                                                                                         "Note that this mode overwrite arguments related to a fake "
                                                                                         "observation, used in the injection test mode, with the content of the fits header.")
    parser.add_argument("--observation_fits",  default=None,                        help="Path to observation fits file")

    parser.add_argument("--injection_test",    action="store_true",                 help="Injection test mode will require an hdf5 file for the reference "
                                                                                         "profile to be recovered, the key in the hdf5 and the index of the profile. "
                                                                                         "Also requires a fits file for the PSF")
    parser.add_argument("--dataset_path",      default=None,                        help="Path to the h5 files with reference profiles for the injection test")
    parser.add_argument("--dataset_key",       default="images",                    help="Key to the reference profile in the dataset")
    parser.add_argument("--dataset_id",        default=None,    type=int,           help="Index for the reference profile to recover")
    parser.add_argument("--observation_pixels", default=128,    type=int,           help="Make a fake observation with this number of pixels on a side")
    parser.add_argument("--observation_pixel_size", default=0.05, type=float,       help="Pixel size for the fake observation, in arcseconds")
    parser.add_argument("--model_pixels",       default=512,     type=int,          help="Number of pixels on a side for the model")
    parser.add_argument("--model_pixel_size",   default=0.025,  type=float,         help="Size of a pixel for the model, in arcseconds")
    parser.add_argument("--zero_padding",       default=0,      type=int,           help="Zero padding in the forward model. Default is no zero-padding")
    parser.add_argument("--noise_rms",          default=0.01,   type=float,         help="White noise standard deviation added to the fake observation. If SLIC is provided, "
                                                                                         "a noise realisation from the SLIC model is used instead. ")
    parser.add_argument("--super_sampling_factor", default=2,   type=int,           help="Factor by which the PSF is super sampled. ")

    # Which likelihood approximation to use?
    parser.add_argument("--diagonal_gaussian_likelihood", action="store_true",      help="Use the diagonal gaussian likelihood approximation")
    # TODO implement this
    parser.add_argument("--pseudo_inverse_gaussian_likelihood", action="store_true", help="Use the pseudo-inverse of the forward model to get a "
                                                                                          "better approximation of the likelihood term. This is more accurate, but "
                                                                                          "also more costly to evaluate.")
    # TODO implement this
    parser.add_argument("--slic_likelihood",    action="store_true",                help="Use a trained SLIC model as an approximation for the likelihood")
    parser.add_argument("--slic_model",         default=None,                       help="Path to the slic model")

    # Prior sampling mode, this will ignore everything about the data. Used for testing or generating training sets.
    parser.add_argument("--from_prior",         action="store_true",               help="Ignore the observation and sample from the prior")

    parser.add_argument("--result_dir",         required=True)
    parser.add_argument("--checkpoints_dir",    required=True,                     help="The script will search in provided model_dir argument for model_id and load checkpoint if it exists.")
    parser.add_argument("--model_checkpoint",   default=None, type=int,            help="Index of the checkpoint to load.")

    # Samplers params
    parser.add_argument("-N", "--em_iterations", required=True,  type=int,           help="Total number of Euler-Maruyama steps to perform")
    parser.add_argument("-W", "--walkers",       default=1,      type=int,           help="Number of independent samples to produce")
    parser.add_argument("-W", "--batch_size",    default=1,      type=int,           help="Batch size, number of samples to produce at a given moment")

    # Reproducibility params
    parser.add_argument("--seed",                default=None,   type=int,       help="Seed for the random number generators.")

    args = parser.parse_args()
    main(args)

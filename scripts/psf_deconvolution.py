from score_models import NCSNpp
from functorch import grad, vmap, vjp
from torch.nn import functional as F
from astropy.cosmology import FlatLambdaCDM
import astropy.units as u
from torchvision.transforms import CenterCrop
from definitions import , DEVICE, load_model, linear_preprocessing
from forward_model import make_forward_model
import numpy as np
import torch
import os
from astropy.io import fits
import h5py
from tqdm import tqdm


# total number of slurm workers detected
# defaults to 1 if not running under SLURM
N_WORKERS = int(os.getenv('SLURM_ARRAY_TASK_COUNT', 1))

# this worker's array index. Assumes slurm array job is zero-indexed
# defaults to one if not running under SLURM
THIS_WORKER = int(os.getenv('SLURM_ARRAY_TASK_ID', 1))


def probes_link_function(x):
    return (x + 1) / 2.


def skirt_link_function(x):
    return x


def preprocess_probes_g_channel(img):  # channel 0
    img = torch.clamp(img, 0, 1.48)
    img = 2 * img / 1.48 - 1.
    return img


def rad_to_arcsec(theta):
    return theta * 180 / np.pi * 3600


def main(args):
    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
    if len(args.dataset_channels) > 1:
        raise ValueError("Only single channel for now, until the script is tested for more")

    if args.redshift is not None:
        cosmo = FlatLambdaCDM(H0=args.h0, Om0=args.Om0, Tcmb0=2.725)
        Ds = cosmo.angular_diameter_distance(args.redshift)
        vars(args)["model_pixel_size"] = rad_to_arcsec((0.1 * u.kpc / Ds).decompose().value)
        print(f"Model has pixel size {args.model_pixel_size} as and field of view {args.model_pixel_size * args.model_pixels} as")

    # Load model
    prior_model = load_model(args.checkpoints_dir, architecture=NCSNpp, data_parallel=False, model_checkpoint=args.model_checkpoint)
    # Hack the VESDE in the model for readability
    sde = prior_model.sde # .module is a hack to
    sigma_min = sde.sigma_min
    sigma_max = sde.sigma_max
    def sigma(t): # scale of the marginal prob. distiribution
        return sigma_min * (sigma_max / sigma_min)**t.view(-1, 1, 1, 1)
    def g(t): # diffusion coefficient of the VESDE
        return sigma(t) * np.sqrt(2 * (np.log(sigma_max) - np.log(sigma_min)))

    if args.real_data:
        data = fits.open(args.observation_fits)
        assert data[args.observation_key].header["CUNIT1"] == "deg"
        assert data[args.observation_key].header["CUNIT2"] == "deg"
        pixel_size = np.sqrt(data[args.observation_key].header["CD1_1"] ** 2 + data[args.observation_key].header["CD1_2"] ** 2) * 3600
        assert len(data[args.observation_key].data.shape) == 2
        assert data[args.observation_key].data.shape[0] == data[args.observation_key].data.shape[1]
        pixels = data[args.observation_key].data.shape[0]
        observation = torch.tensor(data[args.observation_key].data[None].astype(np.float32)).to(DEVICE)
        vars(args)["observation_pixel_size"] = pixel_size
        vars(args)["observation_pixels"] = pixels

    if args.slic_likelihood:
        slic_model = load_model(args.slic_model, architecture=NCSNpp, data_parallel=False, model_checkpoint=args.model_checkpoint)

    # TODO support multiple channels
    # Todo possibly convert pixel size from pc in Connor B. fits file to arcsec using a user specified Hubble constant and redshift
    with fits.open(args.psf_fits) as data:
        psf = data[args.psf_key].data[None].astype(np.float32) # add the channel dimension, a single channel for now.

    forward_model = make_forward_model(args, psf)

    if args.injection_test:
        if args.dataset_channels_last:
            with h5py.File(args.dataset_path, "r") as hf:
                reference_profile = torch.tensor(hf[args.dataset_key][args.dataset_id, ..., args.dataset_channels]).to(DEVICE)[None]
            reference_profile = torch.permute(reference_profile, (0, 3, 1, 2))  # put channels first
        else:
            with h5py.File(args.dataset_path, "r") as hf:
                reference_profile = torch.tensor(hf[args.dataset_key][args.dataset_id, args.dataset_channels]).to(DEVICE)[None]
        if args.downsample > 0:
            reference_profile = torch.nn.functional.avg_pool2d(reference_profile, kernel_size=2 * args.downsample, stride=2 * args.downsample)
        if args.probes:
            reference_profile = preprocess_probes_g_channel(reference_profile)
            link_function = probes_link_function
        else:
            reference_profile = linear_preprocessing(reference_profile)
            link_function = skirt_link_function
        observation = forward_model(link_function(reference_profile))
        if args.slic_likelihood:
            print(f"Using noise map {args.noise_map} | id = {args.noise_index}")
            noise = np.load(args.noise_map)[args.noise_index].astype(np.float32)
            # with h5py.File(args.noise_map, "r") as hf:
            #     noise = hf[args.noise_key][args.noise_index] # TODO support multiple channels
            #     *_, H, W = noise.shape
            noise = torch.tensor(noise).view(1, 1, *noise.shape).to(DEVICE)
            noise = CenterCrop(args.observation_pixels)(noise)
            observation += args.noise_map_multiplicative_factor * noise
        else:
            print(f"Using Gaussian noise with rms = {args.noise_rms}")
            observation += torch.randn_like(observation) * args.noise_rms

    if args.diagonal_gaussian_likelihood:
        def convolved_likelihood(x, t, sigma_n=args.noise_rms):
            var = sigma_n**2 + sigma(t)**2
            y_hat = forward_model(x[None])
            ll = torch.sum(-0.5 * torch.square(observation - y_hat) / var)
            return ll
        convolved_likelihood_gradient = vmap(grad(convolved_likelihood, argnums=0))  # now take in batched inputs and return score
    elif args.pseudo_inverse_gaussian_likelihood:
        raise NotImplementedError("pseudo inverse likelihood not yet supported")
    elif args.slic_likelihood:
        def convolved_likelihood_gradient(x, t):
            y_hat, vjpfunc = vjp(lambda x: forward_model(link_function(x)), x)
            score = slic_model.score(observation - y_hat, t)
            grad = vjpfunc(score)[0]
            return -grad

    if args.from_prior:
        def score_fn(x, t):
            B, *D = x.shape
            prior_score = prior_model(x, t) / sigma(t)
            return prior_score
    else:
        # Sample from the posterior
        def score_fn(x, t):
            B, *D = x.shape
            prior_score = prior_model(x, t) / sigma(t)
            likelihood_score = convolved_likelihood_gradient(x, t)
            return prior_score + args.slic_likelihood_fudge_factor * likelihood_score

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
        hf.create_dataset("model", [args.walkers, 1, args.model_pixels, args.model_pixels], dtype=np.float32)
        hf.create_dataset("reconstruction", [args.walkers, 1, args.observation_pixels, args.observation_pixels], dtype=np.float32)
        hf["model"].attrs["posterior_sample"] = not args.from_prior # make sure we write somewhere if this is a posterior sample or not
        for n in range(args.walkers // args.batch_size):
            with torch.no_grad(): # important to add this context, otherwise Pytorch construct a graph through the sampling procedure.
                # TODO add the possibly of conditioning on a user defined guess, and a user specified "high temperature regime"
                #  x = guess + torch.randn(args.batch_size, 1, args.model_pixels, args.model_pixels).to(DEVICE) * sigma(args.T)
                dt = -1. / args.em_iterations
                t = torch.ones(args.batch_size).to(DEVICE)
                x = torch.randn(args.batch_size, 1, args.model_pixels, args.model_pixels).to(DEVICE) * sigma(t) # TODO add channels
                for _ in tqdm(range(args.em_iterations)):
                    x_mean, x, t = euler_maruyama_step(x, t, dt)
            hf["model"][n * args.batch_size: (n+1) * args.batch_size] = x_mean.cpu().numpy().astype(np.float32)
            hf["reconstruction"][n * args.batch_size: (n+1) * args.batch_size] = forward_model(x_mean).cpu().numpy().astype(np.float32)

        # Do the last batch if there is one
        if args.walkers % args.batch_size > 0:
            with torch.no_grad():
                t = torch.ones(args.batch_size).to(DEVICE)
                x = torch.randn(args.walkers % args.batch_size, 1, args.model_pixels, args.model_pixels).to(DEVICE) * sigma(t)  # TODO add channels
                for _ in tqdm(range(args.em_iterations)):
                    x_mean, x, t = euler_maruyama_step(x, t, dt)
            hf["model"][(n+1) * args.batch_size:] = x_mean.cpu().numpy().astype(np.float32)
            hf["reconstruction"][(n+1) * args.batch_size:] = forward_model(x_mean).cpu().numpy().astype(np.float32)


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--experiment_name",    default="",                         help="Name of the output files")
    parser.add_argument("--psf_fits",           required=True,                      help="Path to PSF fits file")
    parser.add_argument("--psf_key",            required=True,                      help="Key to the PSF in the fits file")
    parser.add_argument("-z", "--redshift",     default=None,                      help="Redshift at which to place the model, which has a resolution of 0.1 comoving kpc")
    parser.add_argument("--h0",                 default=0.70,                       help="Hubble constant")
    parser.add_argument("--Om0",                default=0.3,                        help="Matter density parameter")
    # parser.add_argument("--")

    # REAL DATA MODEL TODO write the code for this mode -> requires handling HST units conversion and possibly others
    # With real data, we only have access to the observation itself and the PSF
    parser.add_argument("--real_data",          action="store_true",                help="Real data mode. This mode requires a "
                                                                                         "fits file for the observation and a fits file for the PSF. "
                                                                                         "Note that this mode overwrite arguments related to a fake "
                                                                                         "observation, used in the injection test mode, with the content of the fits header.")
    parser.add_argument("--observation_fits",   default=None,                        help="Path to observation fits file")
    parser.add_argument("--observation_key",    default=None,                        help="Key to observation in fits file")

    parser.add_argument("--injection_test",    action="store_true",                 help="Injection test mode will require an hdf5 file for the reference "
                                                                                         "profile to be recovered, the key in the hdf5 and the index of the profile. "
                                                                                         "Also requires a fits file for the PSF")
    parser.add_argument("--probes",             action="store_true",                help="Whether to use probes, this is a bit of a hack")
    parser.add_argument("--dataset_path",      default=None,                        help="Path to the h5 files with reference profiles for the injection test")
    parser.add_argument("--dataset_key",       default="images",                    help="Key to the reference profile in the dataset")
    parser.add_argument("--dataset_id",        default=None,    type=int,           help="Index for the reference profile to recover")
    parser.add_argument("--dataset_channels",   nargs="+", default=0, type=int,     help="Channels of the dataset to use. ")
    parser.add_argument("--dataset_channels_last", action="store_true",             help="If provided, then the channels of the dataset are found in the last dimension.")
    parser.add_argument("--observation_pixels", default=128,    type=int,           help="Make a fake observation with this number of pixels on a side")
    parser.add_argument("--observation_pixel_size", default=0.05, type=float,       help="Pixel size for the fake observation, in arcseconds")
    parser.add_argument("--model_pixels",       default=512,     type=int,          help="Number of pixels on a side for the model")
    parser.add_argument("--model_pixel_size",   default=0.01,    type=float,        help="Pixel size for the model")
    parser.add_argument("--zero_padding",       default=0,      type=int,           help="Zero padding in the forward model. Default is no zero-padding")
    parser.add_argument("--noise_rms",          default=0.01,   type=float,         help="White noise standard deviation added to the fake observation. If SLIC is provided, "
                                                                                         "a noise realisation from the SLIC model is used instead. ")
    parser.add_argument("--super_sampling_factor", default=2,   type=int,           help="Factor by which the PSF is super sampled. ")
    parser.add_argument("--downsample",             default=0,      type=int,           help="An argument used to make sure reference profile size match prior")

    # Which likelihood approximation to use?
    parser.add_argument("--diagonal_gaussian_likelihood", action="store_true",      help="Use the diagonal gaussian likelihood approximation")
    # TODO implement this
    parser.add_argument("--pseudo_inverse_gaussian_likelihood", action="store_true", help="Use the pseudo-inverse of the forward model to get a "
                                                                                          "better approximation of the likelihood term. This is more accurate, but "
                                                                                          "also more costly to evaluate.")
    parser.add_argument("--slic_likelihood",    action="store_true",                help="Use a trained SLIC model as an approximation for the likelihood")
    parser.add_argument("--slic_model",         default=None,                       help="Path to the slic model")
    parser.add_argument("--slic_model_checkpoint",   default=None, type=int,        help="Index of the slic model checkpoint to load.")
    parser.add_argument("--slic_likelihood_fudge_factor", default=1., type=float,   help="Balance likelihood and prior with this fudge factor.")
    parser.add_argument("--noise_map",           default=None)
    parser.add_argument("--noise_index",         default=None, type=int)
    parser.add_argument("--noise_map_multiplicative_factor", default=1., type=float, help="Multiply noise map by this factor, modifies noise amplitude")

    # Prior sampling mode, this will ignore everything about the data. Used for testing or generating training sets.
    parser.add_argument("--from_prior",         action="store_true",               help="Ignore the observation and sample from the prior")

    parser.add_argument("--result_dir",         required=True)
    parser.add_argument("--checkpoints_dir",    required=True,                     help="The script will search in provided model_dir argument for model_id and load checkpoint if it exists.")
    parser.add_argument("--model_checkpoint",   default=None, type=int,            help="Index of the checkpoint to load.")

    # Samplers params
    parser.add_argument("-N", "--em_iterations", required=True,  type=int,           help="Total number of Euler-Maruyama steps to perform")
    parser.add_argument("-W", "--walkers",       default=1,      type=int,           help="Number of independent samples to produce")
    parser.add_argument("-B", "--batch_size",    default=1,      type=int,           help="Batch size, number of samples to produce at a given moment")

    # Reproducibility params
    parser.add_argument("--seed",                default=None,   type=int,       help="Seed for the random number generators.")

    args = parser.parse_args()
    main(args)

from score_models import ScoreModel
from torch.func import vmap, grad, vjp
from torchvision.transforms import CenterCrop
from definitions import DEVICE, linear_preprocessing
from forward_model import make_forward_model
from forward_model_old import make_forward_model_old
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astropy import units
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

def try_int(x):
    try:
        return int(x)
    except ValueError:
        return x


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
    
    # Load model
    prior_model = ScoreModel(checkpoints_directory=args.prior_model)
    # Hack the VESDE in the model for readability
    sde = prior_model.sde # .module is a hack to
    sigma_min = sde.sigma_min
    sigma_max = sde.sigma_max
    def sigma(t): # scale of the marginal prob. distiribution
        return sigma_min * (sigma_max / sigma_min)**t.view(-1, 1, 1, 1)
    def g(t): # diffusion coefficient of the VESDE
        return sigma(t) * np.sqrt(2 * (np.log(sigma_max) - np.log(sigma_min)))

    with fits.open(args.psf_fits) as data:
        psf = data[args.psf_key].data[None].astype(np.float32) # add the channel dimension, a single channel for now.

    if args.real_data:
        print("Real data mode...")
        data = fits.open(args.observation_fits)
        observation = np.stack([data[try_int(k)].data for k in args.observation_keys], axis=0)
        observation = torch.tensor(observation).float().to(DEVICE)[None] # [1, O, pix, pix]
        wcs_list = [WCS(data[try_int(k)].header, data) for k in args.observation_keys]
        if args.fiducial_ra is not None:
            assert args.fiducial_dec is not None, "If RA is provided, so must be DEC"
            print(f"Specifying fiducial center coordinate for the model at {args.fiducial_ra}, {args.fiducial_dec}")
            coord = SkyCoord(args.fiducial_ra, args.fiducial_dec, unit=(units.hourangle, units.deg))
        else:
            coord = None
        print(f"Building forward model with {observation.shape[1]:d} exposures")
        forward_model = make_forward_model(
                psf, 
                wcs_list, 
                super_sampling_factor=args.super_sampling_factor,
                model_pixels=args.model_pixels,
                model_pixel_size=args.model_pixel_size * units.arcsec,
                fiducial_center=coord,
                fiducial_orientation=args.fiducial_orientation
                )

    elif args.injection_test:
        print("Injection test ...")
        if len(args.dataset_channels) > 1:
            raise ValueError("Only single channel for now, until the script is tested for more")
        if args.dataset_channels_last:
            print("Using channels last format to read dataset")
            with h5py.File(args.dataset_path, "r") as hf:
                reference_profile = torch.tensor(hf[args.dataset_key][args.dataset_id, ..., args.dataset_channels]).to(DEVICE)[None]
            reference_profile = torch.permute(reference_profile, (0, 3, 1, 2))  # put channels first
            
        else:
            print("Using channels first format to read dataset")
            with h5py.File(args.dataset_path, "r") as hf:
                reference_profile = torch.tensor(hf[args.dataset_key][args.dataset_id, args.dataset_channels]).to(DEVICE)[None]
         
        if args.downsample > 0:
            print(f"Downsampling {args.downsample} times")
            reference_profile = torch.nn.functional.avg_pool2d(reference_profile, kernel_size=2 * args.downsample, stride=2 * args.downsample)

        if args.probes:
            print("Using Probes preprocessing of the g channel")
            reference_profile = preprocess_probes_g_channel(reference_profile)
            
        else:
            print("Using SKIRT linear preprocessing (basically no preprocessing)")
            reference_profile = linear_preprocessing(reference_profile)
            
        
        print("Building old forward model with specified argument for simulation")
        forward_model = make_forward_model_old(args, psf)
        print("Building fake observation (single exposure)")
        observation = forward_model(link_function(reference_profile))
        if args.slic_likelihood:
            print("Adding non-gaussian noise to the observation...")
            print(f"Loading noise map {args.noise_map} | id = {args.noise_index}")
            noise = np.load(args.noise_map)[args.noise_index].astype(np.float32)
            # with h5py.File(args.noise_map, "r") as hf:
            #     noise = hf[args.noise_key][args.noise_index] # TODO support multiple channels
            #     *_, H, W = noise.shape
            noise = torch.tensor(noise).view(1, 1, *noise.shape).to(DEVICE)
            noise = CenterCrop(args.observation_pixels)(noise)
            print(f"Using multiplicative factor {args.noise_map_multiplicative_factor} for adding noise map")
            observation += args.noise_map_multiplicative_factor * noise
        else:
            print(f"Adding Gaussian noise with rms = {args.noise_rms} to the observation")
            observation += torch.randn_like(observation) * args.noise_rms

    if args.probes:
        print("Using probes link function")
        link_function = probes_link_function
        
    else:
        print("Not using any link function")
        link_function = lambda x: x

    if args.diagonal_gaussian_likelihood:
        print("Using Gaussian Likelihood for inference")
        def convolved_likelihood(t, x, sigma_n=args.noise_rms):
            var = sigma_n**2 + sigma(t)**2
            y_hat = forward_model(link_function(x[None]))
            ll = torch.sum(-0.5 * torch.square(observation - y_hat) / var)
            return ll
        convolved_likelihood_gradient = vmap(grad(convolved_likelihood, argnums=0))  # now take in batched inputs and return score
        
    elif args.pseudo_inverse_gaussian_likelihood:
        raise NotImplementedError("pseudo inverse likelihood not yet supported")
    
    elif args.slic_likelihood:
        print("Using SLIC likelihood for inference")
        slic_model = ScoreModel(checkpoints_directory=args.slic_model)
        def convolved_likelihood_gradient(t, x):
            B, *_ = x.shape
            _, O, pix, _ = observation.shape
            y_hat, vjpfunc = vjp(lambda x: forward_model(link_function(x)), x)
            # Compute residuals for each observation and concatenate in batch dimension for SLIC
            residuals = (observation - y_hat).view(B*O, 1, pix, pix)
            tiled_t = torch.tile(t, [O])
            slic_score = slic_model.score(t=tiled_t, x=residuals)
            # reshape slic score to be isomorph to cotangent space of the forward model
            slic_score = slic_score.view(B, O, pix, pix) 
            score = -vjpfunc(slic_score)[0]  # don't forget the minus sign
            return score
    
    if args.from_prior:
        print("Prior sampling: ignoring the likelhood completely")
        def score_fn(t, x):
            B, *D = x.shape
            prior_score = prior_model.score(t, x)
            return prior_score
    else:
        print(f"Posterior sampling with guidance factor {args.slic_guidance_factor}")
        # Sample from the posterior
        def score_fn(t, x):
            B, *D = x.shape
            prior_score = prior_model.score(t, x)
            likelihood_score = convolved_likelihood_gradient(t, x)
            return prior_score + args.slic_guidance_factor * likelihood_score

    def euler_maruyama_step(x, t, dt):
        t += dt
        x_mean = x - g(t) ** 2 * score_fn(t, x) * dt
        z = torch.randn_like(x)
        x = x_mean + g(t) * z * np.sqrt(-dt)
        return x_mean, x, t
    
    # def langevin_corrector_step(x, t, snr):
        

    # Now we do the hard work
    filename = os.path.join(args.result_dir, args.experiment_name + f"_{THIS_WORKER}" + ".h5")
    print("Solving the posterior...")
    with h5py.File(filename, "w") as hf:
        hf["observation"] = observation.cpu().numpy().astype(np.float32).squeeze()
        # TODO add a bunch of relevant info here for reproducibility
        # hf["observation"].attrs["units"] = 'micro Jy'
        # hf["observation"].attrs["pixel_size"] = 'micro Jy'
        hf["psf"] = psf.astype(np.float32).squeeze()
        hf.create_dataset("model", [args.walkers, 1, args.model_pixels, args.model_pixels], dtype=np.float32)
        hf.create_dataset("reconstruction", [args.walkers, 1, args.observation_pixels, args.observation_pixels], dtype=np.float32)
        hf["model"].attrs["posterior_sample"] = not args.from_prior
        for n in range(args.walkers // args.batch_size):
            with torch.no_grad():
                dt = -1. / args.em_iterations
                t = torch.ones(args.batch_size).to(DEVICE)
                x = torch.randn(args.batch_size, 1, args.model_pixels, args.model_pixels).to(DEVICE) * sigma(t)
                for _ in tqdm(range(args.em_iterations)):
                    x_mean, x, t = euler_maruyama_step(x, t, dt)
            hf["model"][n * args.batch_size: (n+1) * args.batch_size] = link_function(x_mean).cpu().numpy().astype(np.float32)
            hf["reconstruction"][n * args.batch_size: (n+1) * args.batch_size] = forward_model(link_function(x_mean)).cpu().numpy().astype(np.float32)

        # Do the last batch if there is one
        if args.walkers % args.batch_size > 0:
            with torch.no_grad():
                t = torch.ones(args.batch_size).to(DEVICE)
                x = torch.randn(args.walkers % args.batch_size, 1, args.model_pixels, args.model_pixels).to(DEVICE) * sigma(t)
                for _ in tqdm(range(args.em_iterations)):
                    x_mean, x, t = euler_maruyama_step(x, t, dt)
            hf["model"][(n+1) * args.batch_size:] = link_function(x_mean).cpu().numpy().astype(np.float32)
            hf["reconstruction"][(n+1) * args.batch_size:] = forward_model(link_function(x_mean)).cpu().numpy().astype(np.float32)


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--experiment_name",    default="",                         help="Name of the output files")
    parser.add_argument("--psf_fits",           required=True,                      help="Path to PSF fits file")
    parser.add_argument("--psf_key",            required=True,                      help="Key to the PSF in the fits file")
    
    parser.add_argument("--model_pixels",       default=256,     type=int,          help="Number of pixels on a side for the model")
    parser.add_argument("--model_pixel_size",   default=0.0125, type=float,        help="Pixel size for the model")
    parser.add_argument("--super_sampling_factor", default=4,   type=int,           help="Factor by which the PSF is super sampled. ")

    parser.add_argument("--real_data",          action="store_true",                help="Real data mode. This mode requires a "
                                                                                         "fits file for the observation and a fits file for the PSF. "
                                                                                         "Note that this mode overwrite arguments related to a fake "
                                                                                         "observation, used in the injection test mode, with the content of the fits header.")
    parser.add_argument("--observation_fits",   default=None,                        help="Path to observation fits file")
    parser.add_argument("--observation_keys",   nargs="+", default=None,             help="Key to observations in the fits file")
    parser.add_argument("--fiducial_ra",        default=None,                        help="Right ascension, in hourangle, of the central pixel of the model")
    parser.add_argument("--fiducial_dec",       default=None,                        help="Declination, in degrees, of the central pixel of the model")
    parser.add_argument("--fiducial_orientation", default=None,                      help="Orienation of the model. Default to first WCS orientation.")
    

    parser.add_argument("--injection_test",    action="store_true",                 help="Injection test mode will require an hdf5 file for the reference "
                                                                                         "profile to be recovered, the key in the hdf5 and the index of the profile. "
                                                                                         "Also requires a fits file for the PSF")
    parser.add_argument("--probes",             action="store_true",                help="Whether to use probes, this is a bit of a hack")
    parser.add_argument("--dataset_path",      default=None,                        help="Path to the h5 files with reference profiles for the injection test")
    parser.add_argument("--dataset_key",       default="images",                    help="Key to the reference profile in the dataset")
    parser.add_argument("--dataset_id",        default=None,    type=int,           help="Index for the reference profile to recover")
    parser.add_argument("--dataset_channels",   nargs="+", default=[0,], type=int,     help="Channels of the dataset to use. ")
    parser.add_argument("--dataset_channels_last", action="store_true",             help="If provided, then the channels of the dataset are found in the last dimension.")
    parser.add_argument("--observation_pixels", default=128,    type=int,           help="Make a fake observation with this number of pixels on a side")
    parser.add_argument("--observation_pixel_size", default=0.05, type=float,       help="Pixel size for the fake observation, in arcseconds")
    parser.add_argument("--zero_padding",       default=0,      type=int,           help="Zero padding in the forward model. Default is no zero-padding")
    parser.add_argument("--noise_rms",          default=0.01,   type=float,         help="White noise standard deviation added to the fake observation. If SLIC is provided, "
                                                                                         "a noise realisation from the SLIC model is used instead. ")
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
    parser.add_argument("--slic_guidance_factor", default=1., type=float,   help="Balance likelihood and prior with this fudge factor.")
    parser.add_argument("--noise_map",           default=None)
    parser.add_argument("--noise_index",         default=None, type=int)
    parser.add_argument("--noise_map_multiplicative_factor", default=1., type=float, help="Multiply noise map by this factor, modifies noise amplitude")

    # Prior sampling mode, this will ignore everything about the data. Used for testing or generating training sets.
    parser.add_argument("--from_prior",         action="store_true",               help="Ignore the observation and sample from the prior")

    parser.add_argument("--result_dir",     required=True)
    parser.add_argument("--prior_model",    required=True,                     help="Prior model checkoint path")

    # Samplers params
    parser.add_argument("-N", "--em_iterations", required=True,  type=int,           help="Total number of Euler-Maruyama steps to perform")
    parser.add_argument("-W", "--walkers",       default=1,      type=int,           help="Number of independent samples to produce")
    parser.add_argument("-B", "--batch_size",    default=1,      type=int,           help="Batch size, number of samples to produce at a given moment")

    # Reproducibility params
    parser.add_argument("--seed",                default=None,   type=int,       help="Seed for the random number generators.")

    args = parser.parse_args()
    main(args)

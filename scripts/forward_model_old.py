import torch
import torch.nn.functional as F
from torch.func import vmap
from definitions import DEVICE
from definitions import interpolate

# Old, keep as reference (didn't account for sub pixel shift)
def make_forward_model_old(args, psf):
    """
    Useful for injection test!! Since we don't care for WCS in those.

    Takes in argument specifying the size of the model (number of pixels, size of the pixels)
    and the target size (number of pixels in the image, size of the pixels). It also takes in a psf
    specified by the user for the deconvolution.

    The operator returned is a functional form of the matrix A that appears in the linear inverse problem
        y = Ax + "noise"

    A is composed of
        0) Possible zero-padding. Default is no padding in the script.
        1) Interpolation of the model to twice the size of the observed image
        2) PSF convolution
        3) Pixelization (2d Average pooling)

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

    fov = args.observation_pixel_size * args.observation_pixels
    x = torch.linspace(-1, 1, args.super_sampling_factor*args.observation_pixels).float() * fov / 2
    x, y = torch.meshgrid(x, x, indexing="ij")
    # Transform these coordinates into model pixel indices
    _min = - args.model_pixel_size * (args.model_pixels + args.zero_padding) / 2
    i_coord = (x - _min) / args.model_pixel_size
    j_coord = (y - _min) / args.model_pixel_size
    coordinates = torch.stack([i_coord, j_coord], dim=0).to(DEVICE)
    def A(x):
        x = F.pad(x, pad=[args.zero_padding]*4, mode="constant", value=0.)
        x = batched_interpolation(x, coordinates)
        x = F.conv2d(x, psf, groups=C, padding="same")
        x = F.avg_pool2d(x, kernel_size=args.super_sampling_factor, stride=args.super_sampling_factor)
        return x
    return A

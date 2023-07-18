import torch
from torch import vmap
from astropy import units as u
import torch.nn.functional as F
from definitions import interpolate, DEVICE


def make_forward_model(args, psf, wcs_list, fiducial_center=None):
    """
    fiducial_center should be a SkyCoord object is passed. Fiducial pc is a (torch) rotation matrix.
    """
    C, H, W = psf.shape
    psf = torch.tensor(psf).to(DEVICE).view(C, 1, H, W) # reshape to a convolution kernel [channel_out, channels_in/groups, H, W]
    batched_interpolation = vmap(interpolate, in_dims=(0, None))  # only batch over the images
    matmul = vmap(torch.matmul, in_dims=(None, 0))

    # Collect center of each WCS to infer sub pixel shifts
    center = args.observation_pixels / 2 - 0.5 # Cutout2D always crop around this pixel
    if fiducial_center is None:
        # use first wcs for our fiducial center 
        fiducial_center = wcs_list[0].pixel_to_world(center, center)
    observation_centers = []
    for wcs in wcs_list:
        observation_centers.append(wcs.pixel_to_world(center, center))
    
    # Observation coordinate system 
    observation_fov = args.observation_pixel_size * args.observation_pixels
    theta = torch.linspace(-1, 1, args.super_sampling_factor*args.observation_pixels).float() * observation_fov / 2
    thx, thy = torch.meshgrid(theta, theta, indexing="xy")
    # Leftmost pixel coordinate in model grid (+0.5 to center coordinates on pixel centers)
    _min = - args.model_pixel_size * (args.model_pixels + args.zero_padding) / 2 + 0.5 * args.model_pixel_size
    def A(x):
        x = F.pad(x, pad=[args.zero_padding]*4, mode="constant", value=0.)
        ys = []
        for i in range(len(wcs_list)):
            # minus sign on y_shift since we move the observation window, not the model
            y_shift = -(observation_centers[i].dec - fiducial_center.dec).to(u.arcsec).value # South -> North
            x_shift = (observation_centers[i].ra - fiducial_center.ra).to(u.arcsec).value # East <- West (hence cancel the minus sign)
            i_coord = (thy - y_shift - _min) / args.model_pixel_size
            j_coord = (thx - x_shift - _min) / args.model_pixel_size
            coordinates = torch.stack([i_coord, j_coord], dim=0).to(DEVICE)
            y = batched_interpolation(x, coordinates)
            y = F.conv2d(y, psf, groups=C, padding="same")
            y = F.avg_pool2d(y, kernel_size=args.super_sampling_factor, stride=args.super_sampling_factor)
            ys.append(y)
        return torch.concat(ys, dim=1) # cat along channel dimension for now, will have to introduce an event dimension
    return A

if __name__ == "__main__":
    from argparse import ArgumentParser
    import matplotlib.pyplot as plt
    from astropy.wcs import WCS
    import numpy as np
    parser = ArgumentParser()
    parser.add_argument("--observation_pixels", default=4,    type=int,           help="Make a fake observation with this number of pixels on a side")
    parser.add_argument("--observation_pixel_size", default=0.05, type=float,       help="Pixel size for the fake observation, in arcseconds")
    parser.add_argument("--model_pixels",       default=8,     type=int,          help="Number of pixels on a side for the model")
    parser.add_argument("--model_pixel_size",   default=0.05,    type=float,        help="Pixel size for the model")
    parser.add_argument("--zero_padding",       default=0,      type=int,           help="Zero padding in the forward model. Default is no zero-padding")
    parser.add_argument("--super_sampling_factor", default=2,   type=int,           help="Factor by which the PSF is super sampled. ")
    args = parser.parse_args()
    
    # Test 1 (im is the model, or signal)
    im = torch.ones([8, 8])
    psf = torch.ones([1, 1, 1])
    
    # reference WCS
    w = WCS(naxis=2) 
    w.wcs.crpix = [4, 4]
    w.wcs.crval = [1., 1.]
    w.wcs.cdelt = np.array([0.05, 0.05])
    w.wcs.ctype = ['RA---TAN', 'DEC--TAN']
    print(w)
    # shifted wcs
    w1 = WCS(naxis=2) 
    w1.wcs.crpix = [4, 4]
    shift = 0.1 / 3600 # divides by 3600 since I convert units to arcsec in the forward model
    w1.wcs.crval = [1. + shift, 1. + shift]
    w1.wcs.cdelt = np.array([0.05, 0.05])
    w1.wcs.ctype = ['RA---TAN',  'DEC--TAN']
    print(w1)
    
    wcs_list = [w, w1]
    A = make_forward_model(args, psf, wcs_list)
    y_hat = A(im[None, None])
   
    fig = plt.figure()
    ax = plt.gca()
    im = y_hat[0, 0]
    ax.set_title("Test 1 Fiducial")
    ax.imshow(im, vmin=-1, vmax=1, origin="lower")
    for (i, j), z in np.ndenumerate(im):
        ax.text(j, i, '{:0.1f}'.format(z), ha='center', va='center') 
    
    fig =plt.figure()
    ax = plt.gca()
    im = y_hat[0, 1]
    ax.set_title("Test 1 Shifted")
    ax.imshow(im, vmin=-1, vmax=1, origin="lower")
    for (i, j), z in np.ndenumerate(im):
        ax.text(j, i, '{:0.1f}'.format(z), ha='center', va='center') 

    plt.show()
    

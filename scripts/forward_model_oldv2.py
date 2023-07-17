import torch
from torch import vmap
from astropy import units as u
import torch.nn.functional as F
from definitions import interpolate, DEVICE


def make_forward_model(args, psf, wcs_list, fiducial_center=None, fiducial_pc=None):
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
    
    # # Collect PC matrix of each WCS (hopefully we deal with square pixel, otherwise math breaks I think)
    # if fiducial_pc is None:
        # fiducial_pc = torch.tensor(wcs_list[0].pixel_scale_matrix).to(DEVICE)
        
    # observation_pcs = []
    # for wcs in wcs_list:
        # observation_pcs.append(torch.tensor(wcs.pixel_scale_matrix).to(DEVICE))
    
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
            # Potential code for hanlding rotation
            # R = fiducial_pc @ observation_pcs[i].T
            # Transform angular coordinates into model pixel coordinates (shift then rotate)
            # beta_y = matmul(R, thy - y_shift)
            # beta_x = matmul(R, thx - x_shift)
            # For now, only shift the coordinates
            i_coord = (thy - y_shift - _min) / args.model_pixel_size
            j_coord = (thx - x_shift - _min) / args.model_pixel_size
            coordinates = torch.stack([i_coord, j_coord], dim=0).to(DEVICE)
            y = batched_interpolation(x, coordinates)
            y = F.conv2d(y, psf, groups=C, padding="same")
            y = F.avg_pool2d(y, kernel_size=args.super_sampling_factor, stride=args.super_sampling_factor)
            ys.append(y)
        return torch.concat(ys, dim=1) # cat along channel dimension for now, will have to introduce an event dimension
    return A

# Better way to do it yet
# # Shift and rotate to model coordinate system
# ra = (world.ra - fiducial_center.ra).to(units.arcsec).value
# dec = (world.dec - fiducial_center.dec).to(units.arcsec).value
# shifted_world = np.stack([ra.ravel(), dec.ravel()], axis=1)
# rotated_world = np.einsum("ij, ...j -> ...i", fiducial_rotation_matrix, shifted_world)
# # Convert into model pixel coordinates
# model_coordinates_i = (rotated_world[:, 1] - model_min_coordinate) / model_pixel_size 
# model_coordinates_j = (rotated_world[:, 0] - model_min_coordinate) / model_pixel_size


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
    """
    Test Rational:
        Each observations is slightly shifted and rotated wrt to a fiducial coordinate system. 
        We thus want to make sure our forward model shifts and rotates our signal 
        in the right direction, similar to what drizzle does.

    Convention:
        x: East <- West 
        y: South -> North 
        angle: East of North

    Explanation:
        A shift is defined in term of an observtion wrt to the model fiducial center. In that sense, the model
        is fixed in the background, and we are moving a window according to the shift and rotation
        
        Moving our window East (positive shift) should shift our model in the opposite direction.
        
        Similarly, moving our window North should move the model south.
        
        Rotating our window East of North will turn the model in the opposite direction.

        We move the window to match the observation. The signal does not move. This behavior match
        our telescope taking snapshots at different point relative to the fixed signal in space. 
    
    Mathematical details:
        Suppose theta is a pixel coordinate in the observation grid, and beta a pixel in the model grid
        We can relate theta and beta knowing a shift value:
            beta' = theta - shift
        
        From our explanation above, the shift is defined in term of the observation window. Thus, we know 
        from the WCS
            shift_window = observation_center - fiducial_center
        The equation for beta is written in term of the model pixel grid, thus
            shift = -shift_window
        The minus is changing our perspective from observation to model space for the interpolation.
        
        Once the coordinates are shifted, we can apply the rotation. Again, we start from the observation
        perspective. The observation has a rotation matrix R_theta relative to the RA-DEC coordinate system. 
        Similarly, the fiducial coordinate system has rotation matrix R_beta. Our goal is to transform theta into 
        beta, so we take the shifted coordinates beta', derotate them from the observation coordinate system, 
        and rerotate them into the fiducial coordinate system
            beta = R_beta R_theta^T beta'
        
        The fact that we apply R_theta^T will indeed rotate our signal clockwise if R_beta is the identity.
        
        Caveat: We generally don't have a rotation matrix. We have a PC matrix. It might be possible to
        get something close to R using
        R = +/- PC / |PC|**(1/2)
        (not sure about the sign) but overall it's a risky proposition. Need to read more in details.

    Test 1:
        x_shift = 0.1/3600 # signal move 2 pixels right 
        y_shift = 0.1/3600 # signal moves 2 pixels down
        angle = 0

    Test 2:
        x_shift = 0/3600 
        y_shift = 0/3600 
        angle = 90 # signal will rotate 90 degrees West of North
        
    Test 3:
        x_shift = 0.1/3600 
        y_shift = 0.1/3600 
        angle = 90 
        
    """
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

    # we will need to deal with this case
    # PC = np.array([[ 2.80967349e-06,  1.38453711e-05], [ 1.35581156e-05, -1.88708320e-06]])
    # R = PC / np.abs(np.linalg.det(PC))**(1/2)

    # # Test 2
    # im = torch.linspace(-1, 1, 8)
    # im, _ = toch.meshgrid(im, im)
    # psf = torch.ones([1, 1, 1])
    
    # # rotated wcs
    # w1 = WCS(naxis=2) 
    # w1.wcs.crpix = [4, 4]
    # shift = 0 / 3600 # divides by 3600 since I convert units to arcsec in the forward model
    # w1.wcs.crval = [1. + shift, 1. + shift]
    # w1.wcs.cdelt = np.array([0.05, 0.05])
    # # Easy case, PC is a simple rotation of 90 degrees
    # PC = np.array([[1, 0],[0, 1]])
    # R = np.array([[0, -1], [1, 0]])
    # PC = R @ PC @ R.T
    # w1.wcs.pc = PC
    # w1.wcs.ctype = ['RA---TAN',  'DEC--TAN']
    # print(w1)
    
    # wcs_list = [w, w1]
    # A = make_forward_model(args, psf, wcs_list)
    # y_hat = A(im[None, None])
   
    # fig = plt.figure()
    # ax = plt.gca()
    # im = y_hat[0, 0]
    # ax.set_title("Test 1 Fiducial")
    # ax.imshow(im, vmin=-1, vmax=1, origin="lower")
    # for (i, j), z in np.ndenumerate(im):
        # ax.text(j, i, '{:0.1f}'.format(z), ha='center', va='center') 
    
    # fig =plt.figure()
    # ax = plt.gca()
    # im = y_hat[0, 1]
    # ax.set_title("Test 1 Shifted")
    # ax.imshow(im, vmin=-1, vmax=1, origin="lower")
    # for (i, j), z in np.ndenumerate(im):
        # ax.text(j, i, '{:0.1f}'.format(z), ha='center', va='center') 
    
    plt.show()
    

from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
from astropy.io import fits
from astropy.nddata import Cutout2D
from stsci.skypac import pamutils
import numpy as np
import torch
import os, json, re
from glob import glob

import matplotlib.pylab as pylab
plt.style.use("science")
params = {
         'axes.labelsize': 15,
         'axes.titlesize': 25,
         'ytick.labelsize' :12,
         'xtick.labelsize' :12,
         'xtick.major.size': 8,
         'xtick.minor.size': 4,
         'xtick.major.width': 1,
         'xtick.minor.width': 1,
         'ytick.color': "w",
         'xtick.color': "w",
         'axes.labelcolor': "k",
         'ytick.labelcolor' : "k",
         'xtick.labelcolor' : "k",
         }
pylab.rcParams.update(params)

LOG10 = np.log(10.)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def adu_to_electron_count(img, gain, exposure_time):
    """
    A small remainder on how to convert ADU to electron / sec units
    """
    return gain * img / exposure_time


# The following conversion are wrt to the 3631 Jy zero point
def ab_mag_to_jansky_per_arcsec_squared(img):
    return 10**(-(img - 8.9) / 2.5)


def electron_count_to_ab_mag(img, photflam, photplam, minimum_count):
    """
    Assumes img is a torch tensor, expressed in electron / sec units.

    photflam and photplam are usually found in HST fits file headers, and refer to the
    inverse sensitivity (in erg/cm^2/sec/Angstrom) and the pivot wavelength respectively.
    See https://hst-docs.stsci.edu/acsdhb/chapter-5-acs-data-analysis/5-1-photometry.

    """
    zero_point = -2.5 * np.log10(photflam) - 21.10 - 5 * np.log10(photplam) + 18.6921
    return -2.5 * torch.log(torch.maximum(torch.ones_like(img) * minimum_count, img)) / LOG10 + zero_point

def electron_count_to_jansky_per_arcsec_squared(img, photflam, photplam):
    zero_point = -2.5 * np.log10(photflam) - 21.10 - 5 * np.log10(photplam) + 18.6921
    return img * 10**(-(zero_point - 8.9)/2.5)

def hst_observation_preprocessing(img, photflam, photplam, minimum_count):
    img = electron_count_to_ab_mag(img, photflam, photplam, minimum_count)
    img = ab_mag_to_jansky_per_arcsec_squared(img)
    return img


def preprocessing(img, dynamic_range=1e5):
    """
    We want the diffusion to happen in log space so that generated images
    strictly have positive flux.

    We use log10(10 micro Jansky / arcsec^2) units instead of AB mag.

    dynamic_range: Sets the decimal value, in Jansky, up to which we hope to model the surface
        brightness. This preprocessing destroys the information below the dynamic range,
        or too faint by our criteria.
    In the end, most pixel values should roughly fall in the range
    [0, log10(dynamic_range)].
    """
    img = ab_mag_to_jansky_per_arcsec_squared(img)
    return torch.log(1e5 * img + 1/dynamic_range) / np.log(10.) + np.log10(dynamic_range)


def preprocessing_nonlinear_sde(img, minimum_flux=1e-3):
    """
    We want the diffusion to happen in log space so that generated images
    strictly have positive flux.

    We use 10 micro Jansky / arcsec^2 units instead of AB mag.

    minimum_flux: Sets the minimum flux value we consider, in 10 micor Jy / as^2 units.
    """
    img = ab_mag_to_jansky_per_arcsec_squared(img)
    return 10**(torch.log(1e5 * img + minimum_flux) / np.log(10.))

def linear_preprocessing(img):
    """
    For usage in inverse problem, we need the processing to be linear. For that reason,
    we only rescale the flux units with linear operations.

    We use 10 micro Jansky / arcsec^2 units instead of AB mag.

    With this approach, the dynamic range is set by the maximum intensity present in the data,
    which should not much bigger than 20 (which is the equivalent of AB mag 18 in our unit system).
    """
    img = ab_mag_to_jansky_per_arcsec_squared(img)
    return 1e5 * img


def inverse_proprocessing(img, dynamic_range=1e5):
    """
    Take a generated image and return it in micro Jansky / arcsec^2. Note that
    this is not a strict inverse. Only the signal in our dynamic range is recovered. 
    """
    return 10**(img - np.log10(dynamic_range))


def interpolate(image, coordinates, zero_fill=False):
    """
    Interpolation function, without a batch size. To make it batched, used vmap from functorch.
    """
    C, H, W = image.shape
    x, y = torch.tensor_split(coordinates, 2, dim=0)
    idxs_out_of_bounds = (y < 0) | (y > W-1) | (x < 0) | (x > W-1)
    x = x.view(-1)
    y = y.view(-1)
    x0 = torch.floor(x).long()
    x1 = x0 + 1
    y0 = torch.floor(y).long()
    y1 = y0 + 1

    x0 = torch.clip(x0, 0, W - 1)
    x1 = torch.clip(x1, 0, W - 1)
    y0 = torch.clip(y0, 0, H - 1)
    y1 = torch.clip(y1, 0, H - 1)
    x = torch.clip(x, 0, W - 1)
    y = torch.clip(y, 0, H - 1)

    Ia = image[..., x0, y0]
    Ib = image[..., x0, y1]
    Ic = image[..., x1, y0]
    Id = image[..., x1, y1]

    wa = (x1 - x) * (y1 - y)
    wb = (x1 - x) * (y - y0)
    wc = (x - x0) * (y1 - y)
    wd = (x - x0) * (y - y0)

    _, new_H, new_W = coordinates.shape
    result = (wa * Ia + wb * Ib + wc * Ic + wd * Id).view(C, new_H, new_W)
    if zero_fill:
        result = torch.where(idxs_out_of_bounds, torch.zeros_like(result), result)
    return result


# Phase out, this was pushed to score_model package
def load_model(checkpoint_dir, architecture, data_parallel=False, model_checkpoint=None):
    model_name = os.path.split(checkpoint_dir)[-1]
    with open(os.path.join(checkpoint_dir, "model_hparams.json"), "r") as f:
        hyperparameters = json.load(f)
    model = architecture(**hyperparameters).to(DEVICE)
    paths = glob(os.path.join(checkpoint_dir, "checkpoint*.pt"))
    checkpoints = [int(re.findall('[0-9]+', os.path.split(path)[-1])[-1]) for path in paths]
    model.eval()
    for p in model.parameters(): p.requires_grad = False  # being extra careful for some reasons
    model.load_state_dict(torch.load(paths[np.argmax(checkpoints)], map_location=DEVICE))
    print(f"Loaded checkpoint {max(checkpoints)} of {model_name}")
    if model_checkpoint is not None:
        model.load_state_dict(torch.load(paths[checkpoints == model_checkpoint], map_location=DEVICE))
        print(f"Loaded checkpoint {model_checkpoint} of {model_name}")
    else:
        model.load_state_dict(torch.load(paths[np.argmax(checkpoints)], map_location=DEVICE))
        print(f"Loaded checkpoint {max(checkpoints)} of {model_name}")
    if data_parallel:
        model = torch.nn.DataParallel(model, device_ids=list(range(torch.cuda.device_count())))
    return model


def create_random_crops(img, crop_height, crop_width, num_crops):
    """
    This function creates a list of random crops from an image.

    Args:
    img: numpy array
    crop_height: int. Height of the crops.
    crop_width: int. Width of the crops.
    num_crops: int. Number of random crops to create.

    Returns:
    List of crops.
    """
    # Get the height of the image
    image_height = img.shape[0]
    image_width = img.shape[1]

    # Define a list to hold the crops
    crops = []

    for _ in range(num_crops):
        # Randomly choose the starting x and y coordinates for the crop
        start_x = np.random.randint(0, image_width - crop_width)
        start_y = np.random.randint(0, image_height - crop_height)

        # Get the crop
        crop = img[start_y:start_y+crop_height, start_x:start_x+crop_width]

        # Append the crop to the list
        crops.append(crop)

    return crops


def save_cutout(
        pixels, 
        flc_or_flt_path: str, 
        coordinate: SkyCoord, 
        output_path: str,
        drz_path=None, 
        ver=1,
        flux_smaller_than=0.35, # Used to compute statistics
        show_sky_statistics=True,
        overwrite=False
        ):
    """
    ver: The version of the SCI path. For HST, ver=1 points to CCDCHIP 2 and ver=2 points to CCDCHIP 1
    
    For each flc or flt paths, make a cutout and save the relevant information (SIP distortion, HDRLET etc.) 
    in each fits file to reconstruct the WCS correctly in inference time. 
    
    Our convention will be to save a cutout per flc. The script will injest each 
    """
    data = fits.open(flc_or_flt_path)
    hdul = []
    hdul.append(fits.PrimaryHDU(header=data[0].header))
    
    # Make science cutout
    header = data["SCI", ver].header
    image = data["SCI", ver].data
    exptime = data[0].header["EXPTIME"]
    wcs = WCS(header, data)
    cutout = Cutout2D(image, coordinate, pixels, wcs=wcs)

    if drz_path is not None:
        drz_data = fits.open(drz_path)
        drz_header = drz_data["SCI"].header
        drz_cutout = Cutout2D(drz_data["SCI"].data, coordinate, pixels, wcs=WCS(drz_header))
        drz_header.update(drz_cutout.wcs.to_header())
        drz = fits.ImageHDU(drz_cutout.data, name="DRZ", header=header)
    hdul.append(drz)
   
    temp = image.ravel() / exptime
    dark_sky = temp[(temp > 0) & (temp < flux_smaller_than)]
    three_sigma = np.quantile(dark_sky, 0.997))
    two_sigma = np.quantile(dark_sky, 0.95)
    dark_sky_mode = np.quantile(dark_sky, 0.5)
    print(f"Measured dark sky flux = {dark_sky_mode:.2e} electrons/s")
   
    if show_sky_statistics:
        plt.figure()
        plt.title(flc_or_flt_pathh)
        plt.hist(dark_sky, bins=100)
        plt.annotate(r"Dark sky flux = %.3f $e^{-}/s$" % dark_sky_mode, xy=(0.01, 0.8), xycoords="axes fraction")
        plt.annotate(r"Dark sky $3\sigma$ = %.3f $e^{-}/s$" % three_sigma, xy=(0.01, 0.7), xycoords="axes fraction")
        plt.annotate(r"Dark sky $2\sigma$ = %.3f $e^{-}/s$" % two_sigma, xy=(0.01, 0.6), xycoords="axes fraction")

        plt.xlim(-0.1)
        plt.xlabel(r"$e^{-}/s$")
        plt.ylabel("Count")
        plt.axvline(np.mean(dark_sky_modes), color="k", ls="--");
        plt.show()
    
    pam = pamutils.pam_from_wcs(cutout.wcs) # pixel area map reads from distortion table coefficients in fits file
    wcs_hdr = cutout.wcs.to_header()
    # Make sure to tell WCS we are using SIP distortions, otherwise it complains endlessly
    wcs_hdr["CTYPE1"] = 'RA---TAN-SIP'
    wcs_hdr["CTYPE2"] = 'DEC--TAN-SIP'
    header.update(wcs_hdr)
    header["DARKSKY"] = dark_sky_mode
    print("Applying PAM, dividing by EXPTIME and subtracting dark sky mode")
    hdul.append(fits.ImageHDU(cutout.data * pam / exptime - dark_sky_mode, name="SCI", header=header))
    
    distortion_papers = []
    for entry in data:
        if entry.name in ["D2IMARR", "WCSDVARR"]:
            distortion_papers.append(entry)
    hdul.extend(distortion_papers)
    hdul_obj = fits.HDUList(hdul)
    hdul_obj.writeto(output_path, overwrite=overwrite)
    return hdul_obj


# def create_noise_dataset():
    # N = 5000
    # img_size = int(size.value)
    # good_crops = []
    # bad_crops = []

    # criteria = 1 # percentage of pixel with a flux above 3 sigma
    # flux_criteria = 0.013
    # for i, f in tqdm(enumerate(flc_paths)):
        # for j, detector in enumerate([1, 4]):
            # k = i * 2 + j
            # data = fits.open(f)
            # img = data[detector].data
            # wcs = WCS(data[detector].header, data)
            # pam = pamutils.pam_from_wcs(wcs) # pixel area map read from distortion table coefficients in fits file
            # exptime = data[0].header["EXPTIME"]
            # dark_sky = dark_sky_modes[k] # use the measured (detector specific) dark sky flux measurement 
            # three_sigma_rule = three_sigmas[k]
            # two_sigma_rule = two_sigmas[k]
            # crops = create_random_crops(img * pam / exptime - dark_sky, img_size , img_size, N) # correct flux with PAM here
            # # Ignore cosmic rays, we don't mind them too much. We want to avoid objects that look like signal
            # for c in crops:
                # above_zero = c > 0
                # # if (((c[~probably_cosmic_ray] > two_sigma_rule).sum() / img_size**2 * 100) < criteria) & (c[above_zero].sum()/img_size**2 < flux_criteria):
                # if (c[above_zero].sum()/img_size**2 < flux_criteria):
                    # good_crops.append(c)
                # else: 
                    # bad_crops.append(c)
    # print(f"{len(good_crops):d} good crops and {len(bad_crops):d} bad crops")



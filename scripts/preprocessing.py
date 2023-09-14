import numpy as np
import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
from astropy.io import fits
from astropy.nddata import Cutout2D
from stsci.skypac import pamutils

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


def align_wcs(wcs, shift_x, shift_y, theta):
    """
    This method takes in a WCS and applies a shift and rotation to its coordinates. This is
    useful when we align the cutouts and fit the shift and orientation so that exposures match
    when we drizzle. 
    """
    hdr = wcs.to_header(True) # True makes it so that we have the SIP coefficients

    pc = wcs.pixel_scale_matrix
    rotation = np.array([[np.cos(theta), -np.sin(theta)],
                         [np.sin(theta), np.cos(theta)]])
    
    new_pc = rotation @ pc
    hdr["PC1_1"] = new_pc[0, 0]
    hdr["PC1_2"] = new_pc[0, 1]
    hdr["PC2_1"] = new_pc[1, 0]
    hdr["PC2_2"] = new_pc[1, 1]
    hdr["CRVAL1"] = hdr["CRVAL1"] + shift_y / 3600 # arcsec to deg
    hdr["CRVAL2"] = hdr["CRVAL2"] + shift_x / 3600
    hdr["CDELT1"] = 1
    hdr["CDELT2"] = 1
    # make a new wcs based on shift and rotation of the model
    new_wcs = WCS(hdr)
    return new_wcs


def save_cutout(
        output_path: str,
        pixels, 
        flc_or_flt_path: str, 
        coordinate: SkyCoord, 
        drz_path=None, 
        ver=1,
        shift_x:float=0.,
        shift_y:float=0.,
        theta:float=0.,
        flux_smaller_than=0.35, # Used to compute statistics
        verbose=0,
        overwrite=False
        ):
    """
    ver: The version of the SCI path. For HST, ver=1 points to CCDCHIP 2 and ver=2 points to CCDCHIP 1
    
    For each flc or flt paths, make a cutout and save the relevant information (SIP distortion, HDRLET etc.) 
    in each fits file to reconstruct the WCS correctly in inference time. 
    
    Our convention will be to save a cutout per flc. The script will injest each cutouts separately.

    I also defined shift_x, shift_y and theta such that an alignement solution can be provided (see notebooks aligning_smacs_targets).
    """
    data = fits.open(flc_or_flt_path)
    hdul = []
    hdul.append(fits.PrimaryHDU(header=data[0].header))
    
    # Make science cutout
    header = data["SCI", ver].header
    image = data["SCI", ver].data
    if data["SCI", ver].header["BUNIT"] != "ELECTRONS":
        print("Data units not in electrons count, exposure time will not be used to normalize the data")
        exptime = 1
    else:
        try:
            exptime = data[0].header["EXPTIME"]
        except KeyError as e:
            # For JWST, they changed the key to effective exposure time in i2d fits files
            exptime = data[0].header["EFFEXPTM"]
    # Important to use the active WCS to get the cutout, then correct cutout wcs with alignment solution
    wcs = WCS(header, data) 
    cutout = Cutout2D(image, coordinate, pixels, wcs=wcs)
    aligned_wcs = align_wcs(cutout.wcs, shift_x, shift_y, theta)

    if drz_path is not None:
        drz_data = fits.open(drz_path)
        drz_header = drz_data["SCI"].header
        drz_cutout = Cutout2D(drz_data["SCI"].data, coordinate, pixels, wcs=WCS(drz_header))
        drz_header.update(drz_cutout.wcs.to_header())
        drz = fits.ImageHDU(drz_cutout.data, name="DRZ", header=header)
        hdul.append(drz)
   
    temp = image.ravel() / exptime
    dark_sky = temp[(temp > 0) & (temp < flux_smaller_than)]
    three_sigma = np.quantile(dark_sky, 0.997)
    two_sigma = np.quantile(dark_sky, 0.95)
    dark_sky_mode = np.quantile(dark_sky, 0.5)
    if verbose:
        print(f"Measured dark sky flux = {dark_sky_mode:.2e} electrons/s")
   
    if verbose:
        plt.figure()
        plt.title(flc_or_flt_path, fontsize=15)
        plt.hist(dark_sky, bins=100)
        plt.annotate(r"Dark sky flux = %.2e $e^{-}/s$" % dark_sky_mode, xy=(0.01, 0.8), xycoords="axes fraction")
        plt.annotate(r"Dark sky $3\sigma$ = %.2e $e^{-}/s$" % three_sigma, xy=(0.01, 0.7), xycoords="axes fraction")
        plt.annotate(r"Dark sky $2\sigma$ = %.2e $e^{-}/s$" % two_sigma, xy=(0.01, 0.6), xycoords="axes fraction")

        # plt.xlim(-0.1)
        plt.xlabel(r"$e^{-}/s$")
        plt.ylabel("Count")
        plt.axvline(np.mean(dark_sky_mode), color="k", ls="--");
        plt.show()
    
    pam = pamutils.pam_from_wcs(cutout.wcs) # pixel area map reads from distortion table coefficients in fits file
    wcs_hdr = aligned_wcs.to_header(True) # True saves the SIP distortions coefficients
    # Make sure to tell WCS we are using SIP distortions, otherwise it complains endlessly
    header.update(wcs_hdr)
    header["DARKSKY"] = dark_sky_mode
    if verbose:
        print("Applying PAM, dividing by EXPTIME and subtracting dark sky mode")
    hdul.append(fits.ImageHDU(cutout.data * pam / exptime - dark_sky_mode, name="SCI", header=header))
    
    distortion_papers = []
    for entry in data:
        if entry.name in ["D2IMARR", "WCSDVARR"]:
            distortion_papers.append(entry)
    hdul.extend(distortion_papers)
    hdul_obj = fits.HDUList(hdul)
    hdul_obj.writeto(output_path, overwrite=overwrite)
    return hdul_obj, data, cutout


def create_noise_dataset(size, N, path, flux_criteria, ver=1,  flux_smaller_than=0.35, return_bad_crops=False, verbose=0):
    data = fits.open(path)
    image = data["SCI", ver].data
    wcs = WCS(data["SCI", ver].header, data)
    pam = pamutils.pam_from_wcs(wcs)
    if data["SCI", ver].header["BUNIT"] != "ELECTRONS":
        if verbose:
            print(f"Data units {data['SCI', ver].header['BUNIT']} is not electrons count, thus exposure time will not be used to normalize the data")
        exptime = 1
    else:
        try:
            exptime = data[0].header["EXPTIME"]
        except KeyError as e:
            # For JWST, they changed the key to effective exposure time in i2d fits files
            exptime = data[0].header["EFFEXPTM"]

    temp = image.ravel() / exptime
    dark_sky = temp[(temp > 0) & (temp < flux_smaller_than)]
    dark_sky_mode = np.quantile(dark_sky, 0.5)
    if verbose:
        print(f"Measured dark sky flux = {dark_sky_mode:.2e} electrons/s")
    
    crops = create_random_crops(image * pam / exptime - dark_sky_mode, size, size, N)
    good_crops = []
    bad_crops = []
    for c in crops:
        if (c[c>0].sum()/size**2 < flux_criteria):
            good_crops.append(c)
        else: 
            bad_crops.append(c)
    if verbose:
        print(f"{len(good_crops):d} good crops and {len(bad_crops):d} bad crops")
    if return_bad_crops:
        return good_crops, bad_crops
    else:
        return good_crops

import numpy as np
import torch
import os, json, re
from glob import glob


LOG10 = np.log(10.)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def preprocess_probes_z_channel(img):  # channel 2
    img = torch.clamp(img, 0, 7.35)
    img = 2 * img / 7.35 - 1.
    return img


def preprocess_probes_r_channel(img):  # channel 1
    img = torch.clamp(img, 0, 3.47)
    img = 2 * img / 3.47 - 1.
    return img


def preprocess_probes_g_channel(img):  # channel 0
    img = torch.clamp(img, 0, 1.48)
    img = 2 * img / 1.48 - 1.
    return img


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




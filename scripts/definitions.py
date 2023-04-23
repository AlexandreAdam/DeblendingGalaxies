import numpy as np
import torch

LOG10 = np.log(10.)


# The following conversion are wrt to the 3631 Jy zero point
def ab_mag_to_jansky(img): 
    return 10**(-(img - 8.9) / 2.5)


def preprocessing(img, dynamic_range=1e5):
    """
    We want the diffusion to happen in log space so that generated images
    strictly have positive fluxHSC_SSP/pdr3_wide

    We use log10(microJy / arcsec^2) units instead of AB mag.

    dynamic_range: Sets the decimal value, in Jy, up to which we hope to model the surface
        brightness. This preprocessing destroys the information below the dynamic range,
        or too faint by our criteria.
    In the end, most pixel values should roughly fall in the range
    [0, log10(dynamic_range)].
    """
    img = ab_mag_to_jansky(img)
    return torch.log(1e6 * img + 1/dynamic_range) / np.log(10.) + np.log10(dynamic_range)


def inverse_proprocessing(img, dynamic_range=1e5):
    """
    Take a generated image and return it in micro Jy. Note that
    this is not a strict inverse. Only the signal in our dynamic range is recovered. 
    """
    return 10**(img - np.log10(dynamic_range))


def interpolate(image, coordinates):
    """
    Interpolation function, without a batch size. To make it batched, used vmap from functorch.
    """
    C, H, W = image.shape
    x, y = torch.tensor_split(coordinates, 2, dim=0)
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
    return (wa * Ia + wb * Ib + wc * Ic + wd * Id).view(C, new_H, new_W)
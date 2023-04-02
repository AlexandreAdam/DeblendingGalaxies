import numpy as np
import torch

LOG10 = np.log(10.)

# The following conversion are wrt to the 3631 Jy zero point
def ab_mag_to_jansky(img): 
    return 10**(-(img - 8.9) / 2.5)

def ab_mag_to_cgs(img):
    return 10**(-(img + 48.6) / 2.5)


def preprocessing(img, dynamic_range=1e8, factor=10**(3.6)):
    """
    We want the diffusion to happen in log space so that generated images 
    strictly whave positive flux

    We use log10(factor * Jy) units instead of AB mag. 

    dynamic_range: Sets the decimal value, in Jy, up to which we hope to model the surface 
        brightness. This preprocessing destroys the information below the dynamic range, 
        or too faint by our criteria.
    factor: Since most galaxies have AB mag around 18 in ther center, 
        we multiply the pixel values by 10^(3.6), which shift the average value to 
        approximately 1 (or, equivalently, shifting AB mag by 9.1). 
    
    In the end, most pixel values should fall in the rough range 
    [log10(dynamic_range), 0].
    """
    img = ab_mag_to_jansky(img)
    return torch.log(factor * dynamic_range * img + 1/dynamic_range) / LOG10

def inverse_proprocessing(img, dynamic_range=1e8, factor=10**(3.6)):
    """
    take a generated or processed image and return it in Jy
    """
    return 10**(img - np.log10(dynamic_range) - np.log10(factor))

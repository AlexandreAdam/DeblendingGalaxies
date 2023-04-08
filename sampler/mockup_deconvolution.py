"""deconvolution mockup

Mockup interface with a single galaxy image for deconvolution. User
instantiates a galaxy_prior object (diffusion model trained on HSC
data), User writes the likelihood for the deconvolution problem, which
is the forward model result compared to a reference image.

"""
from torch import convolve # whatever it is

from models import galaxy_prior
from data import (
    hsc_psf, # store some HSC psfs for the sake of testing
    example_image_deconvolve # an example image to deconvolve
)
from sampler import Sampler

# galaxy prior defined by diffusion model. Could also very easily throw in AutoProf gaussian model or sersic model to show the benefit of having a full prior in this case
GP = galaxy_prior(size = (512,512))


class deconv(Sampler):

    def log_likelihood(self, x, image, variance, psf):

        conv = convolve(x, psf)

        return torch.sum((image - conv)**2 / variance)

my_img = example_image_deconvolve # alternatively user could load their own image with np.load("image.npy")

# create the sampler object with a prior
D = deconv(prior = GP)

for _ in range(10):
    samp = D.sample_posterior(my_img, torch.ones_like(my_img), hsc_psf)

    fig, axarr = plt.subplots(1,2, figsize = (10,5))
    axarr[0].imshow(my_img, origin = "lower")
    axarr[0].set_title("original image")
    axarr[1].imshow(samp.detach().cpu().numpy(), origin = "lower")
    axarr[1].set_title("deconvolved image")
    plt.show()

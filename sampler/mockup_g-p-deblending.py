"""galaxy-psf deblending mockup

Mockup interface with an image of an overlapping PSF and galaxy. The sampler draws posterior images of the galaxy.

"""

from models import galaxy_prior, joint_prior, flat_prior
from sampler import Sampler
from data import hsc_psf, example_image_gpdeblending

# Galaxy prior defined by diffusion
GP = galaxy_prior(size = (512,512))

my_img = example_image_gpdeblending # or user could provide image with np.load("image.npy")

class deblend(Sampler):

    def __init__(self, image, psf, variance = 1., **kwargs):
        super().__init__(**kwargs)
        
        self.image = image
        self.psf = psf
        self.variance = variance

    def log_likelihood(self, x):
        blend = x[0] + self.psf * (10**x[1]) # this just rescales the PSF, could use AutoProf to easily make variable position
        return torch.sum((self.image - blend)**2 / self.variance)

P = joint_prior(GP, flat_prior(low = -5, high = 5))
D = deblend(prior = P, image = my_img, psf = hsc_psf)

for _ in range(10):
    samp = D.sample_posterior()

    fig, axarr = plt.subplots(1,2, figsize = (10,5))
    axarr[0].imshow(my_img, origin = "lower")
    axarr[0].set_title("original image")
    axarr[1].imshow(samp[0].detach().cpu().numpy(), origin = "lower")
    axarr[1].set_title("galaxy image")
    plt.show()

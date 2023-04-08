"""mask in-painting mockup

Mockup interface with an image which has been partially masked. The galaxy prior is used to fill in the masked area.

"""

from models import galaxy_prior
from sampler import Sampler
from data import example_image_unmask, example_mask_unmask

# Here is the diffusion model which defines the galaxy prior
GP = galaxy_prior(size = (512,512)) 

class unmasker(Sampler):

    def log_likelihood(self, x, image, variance, mask):
        not_mask = torch.logical_not(mask)
        return torch.sum((image - x)[not_mask]**2 / variance[not_mask]) # this seems like the best way to evaluate likelihood for in-painting, instead of just setting variance at the masked pixels to be high

my_img = example_image_unmask # or could use np.load("image.npy") # user can provide an image
my_mask = example_mask_unmask # or could use np.load("mask.npy") # user can provide a mask

D = unmasker(prior = GP)

for _ in range(10):
    samp = D.sample_posterior(my_img, torch.ones_like(my_img), my_mask)

    fig, axarr = plt.subplots(1,2, figsize = (10,5))
    axarr[0].imshow(my_img, origin = "lower")
    axarr[0].set_title("original image")
    axarr[1].imshow(samp.detach().cpu().numpy(), origin = "lower")
    axarr[1].set_title("galaxy image")
    plt.show()

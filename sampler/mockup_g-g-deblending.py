"""galaxy-galaxy deblending mockup

Mockup interface with an image of two galaxies to be deblended. User
describes the likelihood, which in this case is the sum of the two
prior samples compared to the reference image.

"""

from models import galaxy_prior, joint_prior
from sampler import Sampler
from data import example_image_ggdeblending

# Here we need two galaxy priors to represent the two objects that are blended. A possible extension of this would be to model many overlapping systems with AutoProf handling the secondary galaxies/stars/sky and diffusion handling the primary object or two. NGC0070 might be a good case, three overlapping galaxies (one is a nice spiral, and many little interlopers that could be modelled with AutoProf
GP1 = galaxy_prior(size = (512,512))
GP2 = galaxy_prior(size = (512,512))

my_img = example_image_ggdeblending # or user could load their own with np.load("image.npy")

class deblend(Sampler):

    def __init__(self, image, variance = 1., **kwargs):
        super().__init__(**kwargs)
        
        self.image = image
        self.variance = variance

    def log_likelihood(self, x):
        blend = x[0] + x[1] # in this case there are two objects, so x is a tuple with the information for both, for blending we just add them
        return torch.sum((self.image - blend)**2 / self.variance)

P = joint_prior(GP1, GP2) # joint prior just holds all the priors and will return everything about them as tuples
D = deblend(prior = P, image = my_img)

for _ in range(10):
    samp = D.sample_posterior()

    fig, axarr = plt.subplots(1,3, figsize = (10,5))
    axarr[0].imshow(my_img, origin = "lower")
    axarr[0].set_title("original image")
    axarr[1].imshow(samp[0].detach().cpu().numpy(), origin = "lower")
    axarr[1].set_title("first galaxy image")
    axarr[2].imshow(samp[1].detach().cpu().numpy(), origin = "lower")
    axarr[2].set_title("second galaxy image")
    plt.show()

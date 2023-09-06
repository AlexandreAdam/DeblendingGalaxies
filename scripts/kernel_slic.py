from typing import Callable, Union

import torch
from torch import Tensor
from torch.nn import Module
from torch.func import vjp, jvp 
from score_models.sde import SDE
from score_models import ScoreModel
from score_models.utils import DEVICE

class KernelSLIC(ScoreModel):
    def __init__(
            self, 
            input_dimensions:list[int,int,int],
            output_dimensions:list[int,int,int],
            forward_model:Callable,
            model: Union[str, Module] = None, 
            sde: SDE=None, 
            checkpoints_directory=None, 
            low_pass_factor:float=1.,
            **hyperparameters
            ):
        """
        Parameters:
        -----------
        input_dimensions : tuple[int, int, int]
            the dimensions of the input to the forward model.
        input_dimensions : tuple[int, int, int]
            the dimensions of the output to the forward model.
        forward_model : Callable
            The forward model function.
        model : Union[str, Module], optional
            The model to be used, by default None.
        sde : SDE, optional
            The stochastic differential equation, by default None.
        checkpoints_directory : str, optional
            The directory to save checkpoints, by default None.
        low_pass_factor : float, optional
            The low-pass coefficient applied to the Fourier transform of the kernel to avoid numerical issues. By default, 
            it is equal to the 0th Fourier coefficient, i.e. the most stringent regularisation. 
        **hyperparameters : dict
            Additional hyperparameters.

        Raises:
        -------
        AssertionError
            If the kernel does not have 3 dimensions.

        Notes:
        ------
        This class overwrites DSM loss to include correlation from the forward model. We use the approximation 
        that the forward model can be represented as a convolution. The effective kernel of this convolution
        is computed from the central pixel of the model.

        This approximation is useful because it can be used in the case where the Jacobian is quite large 
        or does not fit into memory. It allows us to use the convolution theorem to simplify the 
        Gaussian transition score function used as target for DSM. Every functions used in this class
        are of linear complexity with relation to the number of pixels in either the row or column space of A.
        
        This class also handles Tikhonov regularisation of the covariance matrix of the noise through 
        the low_pass_factor, a number between 0 and 1.
       
        This class also handles the caracterisation of the forward factor. 
        The forward factor encodes the loss of power from the forward model. When taking a normal vector in
        row space and projecting it to column space, some power can be loss if the effective kernel is not normalized, 
        which can happen when performing super-resolution. In practice, this factor is the square root of the 
        (0, 0)th Fourier coefficient of the power spectrum.
        It is very important to caracterize this quantity in order to perform diffusion in the column or row space of A, 
        where we need to multiply/divide the learned score by the forward factor. (eventually cite paper for explanations)
        """

        super().__init__(model, sde=sde, checkpoints_directory=checkpoints_directory, **hyperparameters)
        self.idim = input_dimensions
        self.odim = output_dimensions
        self.forward_model = forward_model
        
        self.low_pass_factor = low_pass_factor
        self.hyperparameters.update({"low_pass_factor": low_pass_factor})

        # Compute effective kernel from central pixel in input (sane default for now)
        kernel = effective_kernel(forward_model, self.idim, self.odim, 0, self.idim[1]//2, self.idim[2]//2) 
        self.kernel = torch.as_tensor(kernel).float().to(self.device)
       
        # Compute power spectrum of the Brownian random variable
        power_spectrum = torch.abs(torch.fft.fft2(self.kernel))**2
        self.forward_factor = power_spectrum[..., 0, 0].squeeze().item()**(1/2) # This breaks when I have multiple observations
        self._transition_kernel_precision = 1 / (power_spectrum + self.forward_factor**2 * low_pass_factor) # Tikhonov regularisation
        
    def slic_score(self, t, x, y, *args):
        """
        See Legin et al. (2023), https://iopscience.iop.org/article/10.3847/2041-8213/acd645. 
        By definition, this is the score in the row space of A, so we divide by the forward factor.
        """
        _, *D = x.shape
        y_hat, vjp_func = vjp(self.forward_model, x)
        sigma = self.sde.sigma(t).view(-1, *[1]*len(D)) 
        return - vjp_func(self.model(t, y - y_hat, *args))[0] / sigma / self.forward_factor
    
    def score(self, t, x, *args):
        """
        By definition, this is the score in the column space of A, so we multiply by the forward factor.
        """
        _, *D = x.shape
        return self.model(t, x, *args) / self.sde.sigma(t).view(-1, *[1]*len(D)) * self.forward_factor
    
    def _transition_kernel_score(self, z):
        """
        Analytical formula for the score of the transition kernel of the SDE, to be used
        in our modified version of DSM. We use the effective kernel to compute a precision matrix 
        in Fourier space (diagonal by assumption that the forward model can be approximated by a convolution). 
        """
        z_tilde = torch.fft.fft2(z)
        score_tilde = z_tilde * self._transition_kernel_precision 
        score = - torch.fft.ifft2(score_tilde).real
        return score
    
    def loss_fn(self, samples:Tensor, *args:list[Tensor,...]) -> Tensor:
        B, *D = samples.shape
        sde = self.sde
        # Generate noise in tangent space and then correlate it with forward model
        z_input = torch.randn(B, *self.idim).to(self.device)
        z = self.forward_model(z_input)
        target = self._transition_kernel_score(z)
        t = torch.rand(B).to(self.device) * (sde.T - sde.epsilon) + sde.epsilon
        mean, sigma = sde.marginal_prob(t, samples)
        # Redefine target with forward factor
        u = self.model(t, mean + sigma * z, *args) - target * self.forward_factor
        return torch.sum(u**2) / B
    
    @torch.no_grad()
    def sample(self, batch_size, steps, *args):
        """
        An Euler-Maruyama integration of the model SDE
        
        steps: Number of Euler-Maruyam steps to perform
        """
        sampling_from = "column space"
        z = self.sde.prior(self.idim).sample([batch_size]).to(self.device)
        x = self.forward_model(z)
        dt = -(self.sde.T - self.sde.epsilon) / steps
        t = torch.ones(batch_size).to(self.device) * self.sde.T
        for _ in (pbar := tqdm(range(steps))):
            pbar.set_description(f"Sampling from the {sampling_from} | t = {t[0].item():.1f} | sigma = {self.sde.sigma(t)[0].item():.1e}"
                                 f"| scale ~ {x.std().item():.1e}")
            t += dt
            if t[0] < self.sde.epsilon: # Accounts for numerical error in the way we discretize t.
                break
            g = self.sde.diffusion(t, x)
            f = self.sde.drift(t, x) - g**2 * self.score(t, x, *args) 
            dw = self.forward_model(torch.randn_like(z)) * (-dt)**(1/2)
            x_mean = x + f * dt
            x = x_mean + g * dw 
            if torch.any(torch.isnan(x)):
                print("Diffusion is not stable, NaN were produced. Stopped sampling.")
                break
        return x_mean
 

def effective_kernel(
        forward_model:Callable,
        input_dimensions: list[int, ...], 
        output_dimensions: list[int, ...], 
        channel: int,
        row: int,
        column: int,
        device=DEVICE,
    ):
    """
    Compute the effective kernel in the cotangent space of the forward model for an image to image forward model.

    This function uses automatic differentiation to approximate the entire forward model as a convolution between 
    a tangent vector and an 'effective' kernel that represents the forward model.
    
    The effective kernel is computed using the JVP operations and a tangent vector that select the pixel in input
    space representative of the whole forward model. Generally, this pixel should be chosen to be the central 
    pixel.

    Parameters:
    - input_dimensions (list[int, ...]): The dimensions of the input image. It should be a list of integers representing 
        the number of channels, height, and width of the image.
    - output_dimensions (list[int, ...]): The dimensions of the output image. It should be a list of integers representing 
        the number of channels, height, and width of the image.
    - channel (int): The index of the channel in the output image.
    - row (int): The index of the row in the output image.
    - column (int): The index of the column in the output image.
    - device (str): The device to perform the computation on. Defaults to DEVICE.

    Returns:
    - kernel (torch.Tensor): The effective kernel in the cotangent space of the forward model. 
        It is a tensor with output_dimensions shape.

    Raises:
    - AssertionError: If the length of input_dimensions is not 3 or the length of output_dimensions is not 3.

    """
    # This method leverages instead the JVP, since all we care about is the Jacobian dotted with a specific vector
    assert len(input_dimensions) == 3, "input_dimensions should be a list of length 3"
    assert len(output_dimensions) == 3, "output_dimensions should be a list of length 3"
    x = torch.randn(input_dimensions).unsqueeze(0).to(device)
    v = torch.zeros(input_dimensions).unsqueeze(0).to(device)
    v[..., channel, row, column] = 1.
    _, kernel = jvp(forward_model, (x, ), (v, ))
    return kernel


if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()

    parser.add_argument("--observation_pixels", default=64,    type=int,          help="Has to correspond to the size of the noise images, otherwise the script will break. ")
    parser.add_argument("--observation_pixel_size", default=0.05, type=float,       help="Pixel size for the fake observation, in arcseconds. Should correspond "
                                                                                         "to the pixel size of the noise dataset used (e.g. for HST this should be roughly 0.04 arcseconds.")
    parser.add_argument("--model_pixels",       default=128,     type=int,         help="Number of pixels on a side for the (prior) model ")
    parser.add_argument("--model_pixel_size",   default=0.025,  type=float,          help="Size of a pixel for the (prior) model, in arcseconds")
    parser.add_argument("--super_sampling_factor", default=2,   type=int,           help="Factor by which the PSF is super sampled. ")
    parser.add_argument("--zero_padding",       default=0,      type=int,            help="Zero padding in the forward model. Default is no zero-padding")
    args = parser.parse_args()
    
    import json
    import numpy as np
    from forward_model_old import make_forward_model
    import matplotlib.pyplot as plt
    from astropy.io import fits 

    with open("psf_deconvolution/ncsnpp_hst_noise.json", "r") as f:
        hp = json.load(f)
    with fits.open("../data/F814w_WFC3UV_psf.fits") as data:
        psf = data["PRIMARY"].data.astype(np.float32)[None]

    f = make_forward_model(args, psf)
    idim = [1, args.model_pixels, args.model_pixels]
    odim = [1, args.observation_pixels, args.observation_pixels]
    kernel = effective_kernel(f, idim, odim, 0, args.model_pixels//2, args.model_pixels//2) 

    plt.imshow(kernel.squeeze())
    plt.show()
    model = KernelSLIC(idim, odim, f, "ncsnpp", sigma_min=1e-2, sigma_max=20, low_pass_factor=0.5, **hp)
    print(model.low_pass_factor)
    print(model.forward_factor)
    x = torch.randn(5, 1, args.model_pixels, args.model_pixels)
    t = torch.rand(5)
    y = torch.randn(1, 1, args.observation_pixels, args.observation_pixels)
    print(model.slic_score(t, x, y).shape)
    
    noise = np.load("../data/hst_cutouts_noclip.npy")[0, :64, :64]
    print(noise.shape)
    noise = torch.tensor(noise).float().view(1, 1, 64, 64)
    loss = model.loss_fn(f(x) + noise)
    # plt.imshow(loss[0, 0].detach())
    # plt.colorbar()
    # plt.show()
    print(loss)


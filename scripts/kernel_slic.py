from typing import Callable, Union

import torch
from torch import Tensor
from torch.nn import Module
from torch.func import vjp, jacrev
from score_models.sde import SDE
from score_models import ScoreModel
from score_models.utils import DEVICE

class KernelSLIC(ScoreModel):
    def __init__(
            self, 
            kernel:Tensor, # Kernel of the forward model
            input_dimensions,
            forward_model:Callable,
            model: Union[str, Module] = None, 
            sde: SDE=None, 
            checkpoints_directory=None, 
            low_pass:float=1e-2,
            **hyperparameters
            ):
        """
        Parameters:
        -----------
        kernel : Tensor
            The kernel of the forward model. Should be an image with channels first.
        input_dimensions : Tuple[int, int, int]
            The dimensions of the input to the forward model.
        forward_model : Callable
            The forward model function.
        model : Union[str, Module], optional
            The model to be used, by default None.
        sde : SDE, optional
            The stochastic differential equation, by default None.
        checkpoints_directory : str, optional
            The directory to save checkpoints, by default None.
        low_pass : float, optional
            The low-pass filters applied to the Fourier transform of the kernel to avoid numerical issues, by default 1e-2. 
            Note that the inverse square of the low-pass coefficient is largest numerical value in the precision. The 
            low-pass coefficient is also used to redefine the neural network output to make learning easier. 
        **hyperparameters : dict
            Additional hyperparameters.

        Raises:
        -------
        AssertionError
            If the kernel does not have 3 dimensions.

        Notes:
        ------
        This class overwrites DSM loss to include correlation from the forward model. We use the approximation 
        that the forward model can be represented as a convolution. The kernel of this convolution
        is provided by the user. 

        This approximation is useful because it can be used in the case where the Jacobian is quite large 
        or does not fit into memory. It allows us to use the convolution theorem to simplify the 
        Gaussian transition score function used as target for DSM. 
        
        In this class, it is assumed that the kernel has the same shape as the output of the forward model. See 
        the method to construct an effective kernel from the forward model.
        """

        super().__init__(model, sde=sde, checkpoints_directory=checkpoints_directory, **hyperparameters)
        assert len(kernel.shape) == 3, "Kernel should be an image with channels first." 
        C, H, W = kernel.shape
        self.kernel = torch.as_tensor(kernel).float().to(self.device)
        self.forward_model = forward_model
        self.input_dimensions = input_dimensions
        self.low_pass = low_pass
       
        # Eigenvalues of the effective forward model.
        Lambda = torch.fft.fft2(torch.fft.fftshift(self.kernel)) 
        
        # Construct the diagonal precision matrix in Fourier space
        precision = torch.zeros_like(Lambda)
        precision[Lambda.abs() >= low_pass] = 1/Lambda[Lambda.abs() >= low_pass] / Lambda[Lambda.abs() >= low_pass].conj()
        precision[Lambda.abs() < low_pass] = 1/low_pass**2 # cap the precision of high frequencies in the score (form of low-pass filter)
        self._transition_kernel_precision = precision.view(1, *self.kernel.shape)
        
    def slic_score(self, t, x, y, *args):
        """
        See Legin et al. (2023), https://iopscience.iop.org/article/10.3847/2041-8213/acd645
        """
        y_hat, vjp_func = vjp(self.forward_model, x)
        return - vjp_func(self.score(t, y - y_hat, *args))[0]
    
    def score(self, t, x, *args):
        _, *D = x.shape
        return self.model(t, x, *args) / self.sde.sigma(t).view(-1, *[1]*len(D)) / self.low_pass
    
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
        """
        We rewrite DSM with the appropriate weighting from Girsanov Theorem
        """
        B, *D = samples.shape
                                                          
        sde = self.sde
        # Generate noise in tangent space and then correlate it with forward model
        z = self.forward_model(torch.randn(B, *self.input_dimensions)) 
        target = self._transition_kernel_score(z)
        t = torch.ones(B).to(self.device)*1.#torch.rand(B).to(self.device) * (sde.T - sde.epsilon) + sde.epsilon
        mean, sigma = sde.marginal_prob(t, samples)
        _, vjp_func = vjp(self.forward_model, torch.randn(B, *self.input_dimensions))
        # Redefinition of the model output with the low_pass factor to help learning
        u = vjp_func(target*self.low_pass + self.model(t, mean + sigma * z, *args))[0]
        return torch.sum(u**2) / B
        

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
    assert len(input_dimensions) == 3, "input_dimensions should be a list of length 3"
    assert len(output_dimensions) == 3, "output_dimensions should be a list of length 3"
    x = torch.randn(input_dimensions).unsqueeze(0).to(device)
    A = jacrev(forward_model)(x)
    kernel = A.view([*output_dimensions, *input_dimensions])[..., channel, row, column]
    return kernel


if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()

    parser.add_argument("--observation_pixels", default=32,    type=int,          help="Has to correspond to the size of the noise images, otherwise the script will break. ")
    parser.add_argument("--observation_pixel_size", default=0.05, type=float,       help="Pixel size for the fake observation, in arcseconds. Should correspond "
                                                                                         "to the pixel size of the noise dataset used (e.g. for HST this should be roughly 0.04 arcseconds.")
    parser.add_argument("--model_pixels",       default=64,     type=int,         help="Number of pixels on a side for the (prior) model ")
    parser.add_argument("--model_pixel_size",   default=0.025,  type=float,          help="Size of a pixel for the (prior) model, in arcseconds")
    parser.add_argument("--super_sampling_factor", default=2,   type=int,           help="Factor by which the PSF is super sampled. ")
    parser.add_argument("--zero_padding",       default=0,      type=int,            help="Zero padding in the forward model. Default is no zero-padding")
    args = parser.parse_args()
    import json
    import numpy as np
    import scipy.stats as st
    from forward_model_old import make_forward_model

    with open("psf_deconvolution/ncsnpp_hst_noise.json", "r") as f:
        hp = json.load(f)

    def gkern(kernlen=21, nsig=10):
        """Returns a 2D Gaussian kernel."""

        x = np.linspace(-nsig, nsig, kernlen+1)
        kern1d = np.diff(st.norm.cdf(x))
        kern2d = np.outer(kern1d, kern1d)
        return kern2d/kern2d.sum()
    
    psf = gkern(args.observation_pixels).reshape(1, args.observation_pixels, args.observation_pixels)
    f = make_forward_model(args, psf)
    idim = [1, args.model_pixels, args.model_pixels]
    odim = [1, args.observation_pixels, args.observation_pixels]
    kernel = effective_kernel(f, idim, odim, 0, args.observation_pixels//2, args.observation_pixels//2) 
    model = KernelSLIC(kernel, idim, f, "ncsnpp", sigma_min=1e-2, sigma_max=20, low_pass=1e-2, **hp)
    x = torch.randn(5, 1, args.model_pixels, args.model_pixels)
    t = torch.rand(5)
    y = torch.randn(1, 1, args.observation_pixels, args.observation_pixels)
    print(model.slic_score(t, x, y).shape)
    
    loss = model.loss_fn(f(x)) # loss is computed in cotangent space
    print(loss)


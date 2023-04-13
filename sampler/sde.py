"""
This module contains the implementation of stochastic differential equation (SDE)
classes for use in score-based diffusion models.

Classes:
- SDE: Abstract base class for stochastic differential equations.
- VESDE: Implementation of the Variance Exploding Stochastic Differential
         Equation (VE-SDE) class, a subclass of SDE.
"""

from abc import ABC, abstractmethod
import torch

__all__ = ["SDE", "VESDE"]

class SDE(ABC):
    """Abstract base class for stochastic differential equations (SDEs)
    used in score-based diffusion models.

    Attributes:
      sigma_min (torch.Tensor): The minimum value for the noise level.
      sigma_max (torch.Tensor): The maximum value for the noise level.
      N (int): Number of steps in the reverse SDE process.
      t_min (torch.Tensor): The minimum value for the time variable.
      t_max (torch.Tensor): The maximum value for the time variable.

    """
    
    def __init__(
        self,
        sigma_min: float,
        sigma_max: float,
        N: int = 1000,
        t_min: float = 0.0,
        t_max: float = 1.0,
    ):

        self.sigma_min = torch.as_tensor(sigma_min)
        self.sigma_max = torch.as_tensor(sigma_max)
        self.N = N
        self.t_min = torch.as_tensor(t_min)
        self.t_max = torch.as_tensor(t_max)

    @abstractmethod
    def sigma(self, t: torch.Tensor) -> torch.Tensor:
        """Abstract method for computing the noise level function.

        Args:
          t (torch.Tensor): The time variable.

        Returns:
          torch.Tensor: The noise level at the given time `t`.

        """
        pass

    def g(self, t: torch.Tensor) -> torch.Tensor:
        """Compute the scale factor based on the ratio of noise levels.
        
        Args:
          t (torch.Tensor): The time variable.

        Returns:
          torch.Tensor: The scale factor for the given time `t`.

        """
        return self.sigma(t) * torch.sqrt(
            2 * torch.log(self.sigma_max / self.sigma_min)
        )

    def reverse_sde(self, X, score, D):
        """Perform the reverse SDE process to denoise the input tensor(s).

        Args:
          X (torch.Tensor or Tuple[torch.Tensor]): Input tensor(s) representing the
              noisy observations.
          score (Callable): Function to compute the score for the input tensor(s).
          D (int): The diffusion step.

        Returns:
          torch.Tensor or Tuple[torch.Tensor]: Denoised input tensor(s).

        """
        B, *CLW = X[0].shape

        dt = torch.tensor((self.t_max - self.t_min) / self.N) * torch.ones((B, 1))

        T = (
            torch.flip(torch.linspace(self.t_min, self.t_max, self.N), (0,))
            .reshape(-1, 1)
            .repeat(1, B)
        )

        for t in T[1:]:
            g_t = self.g(t)
            S = score(X, t, D)
            if isinstance(X, tuple):
                dW = tuple(torch.randn_like(x) for x in X)
                X = tuple(
                    x + g_t ** 2 * s * dt + g_t * dw * torch.sqrt(dt)
                    for x, s, dw in zip(X, S, dW)
                )
            else:
                dw = torch.randn_like(X)
                X = X + g_t ** 2 * S * dt + g_t * dw * torch.sqrt(dt)

        return X


class VESDE(SDE):
    """Implementation of the Variance Exploding Stochastic Differential
    Equation (VE-SDE) class, a subclass of SDE.
    
    The VE-SDE is designed to have a noise level that varies with
    time, which can be useful in certain score-based diffusion models.

    """
    def sigma(self, t: torch.Tensor) -> torch.Tensor:
        """Compute the noise level function for the VE-SDE.

        Args:
          t (torch.Tensor): The time variable.

        Returns:
          torch.Tensor: The noise level at the given time `t`.

        """
        return self.sigma_min * (self.sigma_max / self.sigma_min) ** (
            (t - self.t_min) / (self.t_max - self.t_min)
        )

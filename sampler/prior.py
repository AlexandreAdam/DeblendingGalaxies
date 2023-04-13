"""
This module contains implementations of prior classes for score-based diffusion models.

Classes:
- Prior: Abstract base class for priors.
- FlatPrior: Uniform prior distribution over a specified range.
- JointPrior: Joint prior distribution formed by combining individual priors.
"""

from abc import ABC, abstractmethod
from typing import Tuple

import torch

__all__ = ["Prior", "FlatPrior", "JointPrior"]

class Prior(ABC):
    """
    Abstract base class for priors used in score-based diffusion models.
    
    Attributes:
      sde (SDE): The stochastic differential equation (SDE) associated with the prior.
    """
    
    def __init__(self, sde: "SDE"):
        self.sde = sde

    @abstractmethod
    def draw(self) -> Union[torch.Tensor, Tuple[torch.Tensor]]:
        """Draw a high temperature sample to initialize the prior.

        Construct a tensor which can be used as initial conditions at
        high temperature for the score based diffusion modeling
        process.

        Returns:
          Union[torch.Tensor, Tuple[torch.Tensor]]: A tensor or tuple of tensors
          representing the initial conditions at high temperature for the
          score-based diffusion modeling process.
        """
        pass

    @abstractmethod
    def score(
        self, X: Union[torch.Tensor, Tuple[torch.Tensor]], t: torch.Tensor
    ) -> Union[torch.Tensor, Tuple[torch.Tensor]]:
        """Compute score at current parameter space location.

        Compute the score for a given input tuple of tensors. The
        input X is formatted with the same shape(s) as the `self.draw`
        function outputs. The input t gives the "time" variable in the
        SDE where zero is the reference distribution and one is the
        Gaussian convolved distribution.

        Args:
          X (Union[torch.Tensor, Tuple[torch.Tensor]]): Input tensor(s) formatted
              with the same shape(s) as the `self.draw` function outputs.
          t (torch.Tensor): "Time" variable in the SDE where zero is the reference
              distribution and one is the Gaussian convolved distribution.

        Returns:
          Union[torch.Tensor, Tuple[torch.Tensor]]: The score for the input tensor(s)
          at the current parameter space location.
        """
        pass


class FlatPrior(Prior):
    """Uniform prior distribution over a specified range.

    Attributes:
      xmin (torch.Tensor): The lower bound of the uniform distribution.
      xmax (torch.Tensor): The upper bound of the uniform distribution.
      tau (float): A positive scaling factor for the score computation.

    """
    def __init__(self, xmin, xmax, sde, tau=10):
        super().__init__(sde)

        self.xmin = xmin
        self.xmax = xmax
        self.tau = tau

    def draw(self):
        """Draw a random sample from the uniform distribution."""
        return self.xmin + (self.xmax - self.xmin) * torch.rand(self.xmin.size)

    def score(self, X, t):
        """
        Compute the score for a given input tensor based on the uniform distribution.
        """
        delta = self.xmax - self.xmin
        return (self.tau / delta) * (
            torch.exp((X[0] - self.xmax) * self.tau / delta)
            - torch.exp(-(X[0] - self.xmin) * self.tau / delta)
        )


class JointPrior(Prior):
    """Joint prior distribution formed by combining individual priors.
    
    Attributes:
      priors (Tuple[Prior]): A tuple of individual prior objects.

    """
    def __init__(self, sde, *priors):
        super().__init__(sde)
        self.priors = priors

    def draw(self):
        """Draw random samples from the individual priors."""
        return tuple(P.draw() for P in self.priors)

    def score(self, X, t):
        """Compute the joint score for a given input tuple of tensors by
        combining the scores of the individual priors.

        """
        return tuple(P.score(x, t) for x in X)

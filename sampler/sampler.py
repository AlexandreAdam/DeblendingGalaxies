"""
This module contains the implementation of the Sampler class, which is used to
compute log-likelihood and generate posterior samples for score-based diffusion models.

Classes:
- Sampler: Abstract base class for sampler objects.
"""

from abc import ABC, abstractmethod
from typing import Tuple, Optional
import torch

__all__ = ["Sampler"]

class Sampler(ABC):
    """Abstract base class for sampler objects used in score-based
    diffusion models.

    Attributes:
      prior (Prior): The prior distribution to use for sampling.
      sde (SDE): The stochastic differential equation (SDE) associated with the diffusion model.

    """

    def __init__(self, prior: "Prior", sde: "SDE"):

        self.prior = prior
        self.sde = sde

    @abstractmethod
    def log_likelihood(
        self, X: Union[torch.Tensor, Tuple[torch.Tensor]], *args
    ) -> Union[torch.Tensor, Tuple[torch.Tensor]]:
        """Abstract method for computing the log-likelihood for a given input
        tensor or tuple of tensors.

        Args:
          X (Union[torch.Tensor, Tuple[torch.Tensor]]): Input tensor(s).
          *args: Additional arguments required for log-likelihood computation.

        Returns:
          Union[torch.Tensor, Tuple[torch.Tensor]]: The log-likelihood for the input tensor(s).

        """
        pass

    def score(
        self, X: Union[torch.Tensor, Tuple[torch.Tensor]], t: torch.Tensor, D: tuple
    ) -> Union[torch.Tensor, Tuple[torch.Tensor]]:
        """Compute the score for a given input tensor or tuple of tensors.
    
        Args:
            X (Union[torch.Tensor, Tuple[torch.Tensor]]): Input tensor(s).
            t (torch.Tensor): The time variable.
            D (tuple): Additional arguments required for log-likelihood computation.
    
        Returns:
            Union[torch.Tensor, Tuple[torch.Tensor]]: The score for the input tensor(s).

        """
        return tuple(
            P + L for P, L in zip(self.prior(X, t), self.log_likelihood(X, *D))
        )

    def posterior_sample(
        self, *D, X: Optional[Union[torch.Tensor, Tuple[torch.Tensor]]] = None
    ) -> Union[torch.Tensor, Tuple[torch.Tensor]]:
        """
        Generate a posterior sample using the prior and SDE.
    
        Args:
            *D: Additional arguments required for log-likelihood computation.
            X (Optional[Union[torch.Tensor, Tuple[torch.Tensor]]], optional): Input tensor(s).
                If not provided, a sample will be drawn from the prior.
    
        Returns:
            Union[torch.Tensor, Tuple[torch.Tensor]]: A sample from the posterior distribution.
        """
        if X is None:
            X = self.prior.draw()

        return self.sde(X, self.score, D)

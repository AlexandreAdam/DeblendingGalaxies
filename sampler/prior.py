from abc import ABC, abstractmethod
from typing import Tuple

import torch

class Prior(ABC):

    def __init__(self, sde: "SDE"):
        self.sde = sde
        
    @abstractmethod
    def draw(self) -> Tuple[torch.Tensor]:
        """Draw a high temperature sample to initialize the prior.

        Construct a tensor which can be used as initial conditions at
        high temperature for the score based diffusion modeling
        process.

        """
        pass
    
    @abstractmethod
    def score(self, X: Tuple[torch.Tensor], t: torch.Tensor) -> Tuple[torch.Tensor]:
        """Compute score at current parameter space location.

        Compute the score for a given input tuple of tensors. The
        input X is formatted with the same shape(s) as the `self.draw`
        function outputs. The input t gives the "time" variable in the
        SDE where zero is the reference distribution and one is the
        Gaussian convolved distribution.

        """
        pass

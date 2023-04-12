from abc import ABC, abstractmethod
from typing import Tuple, Optional
import torch

class Sampler(ABC):

    def __init__(self, prior: "Prior", sde: "SDE"):

        self.prior = prior
        self.sde = sde

    @abstractmethod
    def log_likelihood(self, X: Tuple[torch.Tensor], *args) -> Tuple[torch.Tensor]:
        pass

    def score(self, X: Tuple[torch.Tensor], t: torch.Tensor, D: tuple) -> Tuple[torch.Tensor]:
        return tuple(self.prior.score(x, t) + LL for x, LL in zip(X, self.log_likelihood(X, *D)))
        
    def posterior_sample(self, *D, X: Optional[Tuple[torch.Tensor]] = None) -> Tuple[torch.Tensor]:
        if X is None:
            X = self.prior.draw()

        return self.sde(X, self.score, D)
        

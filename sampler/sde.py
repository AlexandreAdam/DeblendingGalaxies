from abc import ABC, abstractmethod


class SDE(ABC):

    def __init__(self, sigma_min: float, sigma_max: float, N: int = 1000, t_min: float = 0., t_max: float = 1.):

        self.sigma_min = torch.as_tensor(sigma_min)
        self.sigma_max = torch.as_tensor(sigma_max)
        self.N = N
        self.t_min = torch.as_tensor(t_min)
        self.t_max = torch.as_tensor(t_max)

    @abstractmethod
    def sigma(self, t: torch.Tensor) -> torch.Tensor:
        pass

    def g(self, t: torch.Tensor) -> torch.Tensor:
        return self.sigma(t) * torch.sqrt(2 * torch.log(self.sigma_max / self.sigma_min))
    
    def reverse_sde(self, X, score, D):

        B, *CLW = X[0].shape
        
        dt = torch.tensor((self.t_max - self.t_min) / self.N) * torch.ones((B, 1))
        
        T = torch.flip(torch.linspace(self.t_min, self.t_max, self.N), (0,)).reshape(-1, 1).repeat(1,B)
        
        for t in T[1:]:
            g_t = self.g(t)
            S = score(X, t, D)
            dW = tuple(torch.randn_like(x) for x in X)
            X = tuple(x + g_t**2 * s * dt + g_t * dw * torch.sqrt(dt) for x,s,dw in zip(X, S, dW))

        return X
        
class VESDE(SDE):

    def sigma(self, t: torch.Tensor) -> torch.Tensor:
        return self.sigma_min * (self.sigma_max / self.sigma_min) ** ((t - self.t_min) / (self.t_max - self.t_min))

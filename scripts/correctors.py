import torch
from definitions import DEVICE

def simps(f, a: float, b: float, T: int = 128, device=DEVICE) -> torch.Tensor:
    if T % 2 == 1:
        T += 1
    dx = (b - a) / T
    x = torch.linspace(a, b, T + 1, device=device)
    y = f(x)
    S = dx / 3 * torch.sum(y[0:-1:2] + 4 * y[1::2] + y[2::2], axis=0)
    return S

def accept(log_alpha):
    """
    log_alpha: The log of the acceptance ratio. In details, this
        is the difference between the log probability of the proposed state x'
        and the previous state x, plus the difference between
        the transition kernel probability q in case it is asymmetrical:
            delta_alpha = log p(x') - log p(x) + log q(x | x') - log q(x' | x)

    :return: A boolean vector, whether the state is accepted or not
    """
    return torch.log(torch.rand_like(log_alpha)) < log_alpha


def compute_delta_logp(current_state, proposed_state, score_fn, log_prob_fn=None, delta_logp_steps=2):
    """
    Computes the log probability difference between the proposed state x' and the current state x
        log p(x') - log p(x)
    """
    if log_prob_fn is not None:
        return log_prob_fn(proposed_state) - log_prob_fn(current_state)
    # else, compute the line integral
    v = proposed_state - current_state
    cs = current_state
    T = delta_logp_steps + 1
    B, *D = cs.shape

    def integrand(t):
        """
        No assumption is made about the dimension of the state vector, other than
        it can be decomposed into a batch (or walkers) dimension and a state dimension (shape = [B, *D]).
        The dot product assumes a uniform weight matrix.

        t: Time vector of size T
        :return: The integrand, which is a vector of size [T, B] with value score(gamma(t))^T gamma'(t)
            with gamma(t) = t * v + cs
        """
        gamma_t = t.view(T, 1, *[1]*len(D)) * v.unsqueeze(0) + cs.unsqueeze(0)  # broadcast onto new time dimension
        score = score_fn(gamma_t.view(T * B, *D))  # compress time and batch dimension together for network
        v_repeated = v.repeat(T, *[1]*len(D))  # copy batch T times to match compressed dimension
        return torch.einsum("td, td -> t", score.view(T*B, -1), v_repeated.view(T*B, -1)).view(T, B) # compute dot product across D
    return simps(integrand, 0., 1., delta_logp_steps, device=current_state.device)


def hmc_step(current_state, epsilon, score_fn, mass, leapfrog_steps, **kwargs):
    W, *D = current_state.shape
    epsilon = epsilon.view(-1, *[1]*len(D))
    p_0 = torch.randn_like(current_state)
    p_t = torch.clone(p_0)
    x_t = torch.clone(current_state)
    delta_logp = 0
    for _ in range(leapfrog_steps):
        # First half of the leapfrog integrator
        p_half = p_t + 0.5 * epsilon * score_fn(x_t)
        x_t_p1 = x_t + epsilon * p_half / mass
        # Update delta_logp
        v = x_t_p1 - x_t
        score = score_fn(x_t_p1)
        delta_logp = delta_logp + (score * v).view(W, -1).sum(dim=1)
        # Second half of the leapfrog integrator
        p_t = p_half + 0.5 * epsilon * score
        x_t = x_t_p1
    proposed_state = x_t
    current_kinetic_energy = - 0.5 * torch.sum(p_0.view(W, -1)**2, dim=1) / mass
    proposal_kinetic_energy = - 0.5 * torch.sum(p_t.view(W, -1)**2, dim=1) / mass
    log_alpha = delta_logp + proposal_kinetic_energy - current_kinetic_energy
    accepted = accept(log_alpha)
    proposed_state[~accepted] = current_state[~accepted]
    return proposed_state


def ula_step(current_state, epsilon, score_fn, **kwargs):
    W, *D = current_state.shape
    epsilon = epsilon.view(-1, *[1]*len(D))
    score = score_fn(current_state)
    # Langevin step
    z = torch.randn_like(current_state)
    proposed_state = current_state + epsilon * score + (2 * epsilon)**(1/2) * z
    return proposed_state


def mala_step(current_state, epsilon, score_fn, log_prob_fn=None, delta_logp_steps=2, **kwargs):
    W, *D = current_state.shape
    epsilon = epsilon.view(-1, *[1]*len(D))
    score = score_fn(current_state)
    # Langevin step
    z = torch.randn_like(current_state)
    proposed_state = current_state + epsilon * score + (2 * epsilon)**(1/2) * z
    # Metropolis adjustment
    delta_logp = compute_delta_logp(current_state, proposed_state, score_fn, log_prob_fn, delta_logp_steps)
    kernel_forward = torch.sum(
            ((proposed_state - current_state - epsilon * score) ** 2).flatten(1), 
            dim=1) / 4 / epsilon.squeeze()
    kernel_backward = torch.sum(
            ((current_state - proposed_state - epsilon * score_fn(proposed_state)) ** 2).flatten(1), 
            dim=1) / 4 / epsilon.squeeze()
    log_alpha = delta_logp - kernel_backward + kernel_forward
    accepted = accept(log_alpha)
    proposed_state[~accepted] = current_state[~accepted]
    return proposed_state



if __name__ == "__main__":
    x = torch.randn(5, 1, 8, 8)
    score_fn = lambda x: x
    log_prob_fn = lambda x: torch.sum(x.flatten(1), dim=1)
    epsilon = torch.linspace(0, 1, 5).float()
    
    print(mala_step(x, epsilon, score_fn, log_prob_fn))
    print(mala_step(x, epsilon, score_fn))
    print(hmc_step(x, epsilon,  score_fn, mass=1., leapfrog_steps=10))
    print(ula_step(x, epsilon, score_fn))
    
    

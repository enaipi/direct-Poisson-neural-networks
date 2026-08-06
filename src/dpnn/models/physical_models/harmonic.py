"""Harmonic oscillator particle system models.
 
N particles in D dimensions, each subject to an independent harmonic potential. 
The state is represented in canonical coordinates z = [q, p]. The Hamiltonian is
 
    H(q, p) = sum_i p_i^2 / (2M) + 1/2 k sum_i q_i^2
 
i.e. N * D independent 1D harmonic oscillators sharing a mass M and
spring constant k.
"""


import torch
 
from dpnn.training import DEFAULT_folder_name
from .base import load_models
 
 
def _as_batched_tensor(value, device="cpu", dtype=torch.float32):
    """Convert an initial state to shape (batch, dimension)."""
    tensor = torch.as_tensor(value, dtype=dtype, device=device)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim != 2:
        raise ValueError(f"Expected a 1D or 2D state tensor, got shape {tuple(tensor.shape)}")
    return tensor


class HarmonicCN(object):
    """N particles in D dimensions with a harmonic potential, Crank-Nicolson step."""
 
    def __init__(
        self,
        N,
        D,
        M,
        dt,
        init_q,
        init_p,
        k=1.0,
        device="cpu",
        dtype=torch.float32,
    ):
        self.dtype = dtype
        self.N = N
        self.D = D
        self.M = M
        self.dt = dt
        self.k = k
        self.device = device
 
        self.q = _as_batched_tensor(init_q, device=device, dtype=dtype)
        self.p = _as_batched_tensor(init_p, device=device, dtype=dtype)
        if self.q.shape != self.p.shape:
            raise ValueError(f"init_q and init_p must have the same shape, got {self.q.shape} and {self.p.shape}")
 
        self.dimensions = self.q.shape[1]
        if self.dimensions != N * D:
            raise ValueError(f"Expected state of size N*D={N * D}, got {self.dimensions}")
        self.dim = 2 * self.dimensions

    def _split_state(self, z):
        return z[:, :self.dimensions], z[:, self.dimensions:]

    def _state(self):
        return torch.cat([self.q, self.p], dim=1)
 
    def _potential_gradient(self, q):
        return self.k * q

    def z_dot(self, z):
        q, p = self._split_state(z)
        q_dot = p / self.M
        p_dot = -self._potential_gradient(q)
        return torch.cat([q_dot, p_dot], dim=1)

    def get_E(self, z):
        """Calculate total energy."""
        q, p = self._split_state(z)
        kinetic = 0.5 * (p * p).sum(dim=1) / self.M
        potential = 0.5 * self.k * (q * q).sum(dim=1)
        return kinetic + potential

    def get_L(self, z):
        """Get the canonical 2(N*D) x 2(N*D) Poisson matrix."""
        batch_size = z.shape[0]
        eye = torch.eye(self.dimensions, dtype=z.dtype, device=z.device)
        zeros = torch.zeros_like(eye)
        L_single = torch.cat(
            [
                torch.cat([zeros, eye], dim=1),
                torch.cat([-eye, zeros], dim=1),
            ],
            dim=0,
        )
        return L_single.unsqueeze(0).repeat(batch_size, 1, 1)

    def m_new(self, solver_iterations=200, tol=1e-6):
        """Advance one step using fixed-point Crank-Nicolson iteration."""
        z_old = self._state()
        z_new = z_old.clone()
        z_dot_old = self.z_dot(z_old)
 
        for _ in range(solver_iterations):
            z_prev = z_new.clone()
            z_new = z_old + 0.5 * self.dt * (z_dot_old + self.z_dot(z_new))
 
            rel_error = torch.norm(z_new - z_prev, dim=1) / (torch.norm(z_prev, dim=1) + 1e-12)
            if torch.all(rel_error < tol):
                break
 
        self.q, self.p = self._split_state(z_new)
        return self.q, self.p


class HarmonicIMR(HarmonicCN):
    """N particles in D dimensions with a harmonic potential, implicit midpoint step."""
 
    def m_new(self, solver_iterations=200, tol=1e-6):
        """Advance one step using fixed-point implicit midpoint iteration."""
        z_old = self._state()
        z_new = z_old.clone()
 
        for _ in range(solver_iterations):
            z_prev = z_new.clone()
            z_mid = 0.5 * (z_old + z_new)
            z_new = z_old + self.dt * self.z_dot(z_mid)
 
            rel_error = torch.norm(z_new - z_prev, dim=1) / (torch.norm(z_prev, dim=1) + 1e-12)
            if torch.all(rel_error < tol):
                break
 
        self.q, self.p = self._split_state(z_new)
        return self.q, self.p


class HarmonicNeural(HarmonicCN):
    """N-particle harmonic system evolved with learned energy and Poisson-structure networks."""
 
    def __init__(
        self,
        N,
        D,
        M,
        dt,
        init_q,
        init_p,
        k=1.0,
        device="cpu",
        method="without",
        name=DEFAULT_folder_name,
        dtype=torch.float32,
    ):
        super(HarmonicNeural, self).__init__(
            N, D, M, dt, init_q, init_p, k=k, device=device, dtype=dtype
        )
        self.energy_net, self.L_net, self.J_net, self.A = load_models(
            name=name, method=method, device=device
        )
        self.method = method
 
    def neural_zdot(self, z):
        z = z.clone().detach().requires_grad_(True)
        energy = self.energy_net(z)
        energy_grad = torch.autograd.grad(energy.sum(), z, only_inputs=True, retain_graph=True)[0]
 
        if self.method in ("soft", "without"):
            L = self.L_net(z)
            return torch.bmm(L, energy_grad.unsqueeze(-1)).squeeze(-1)
 
        raise Exception("Implicit neural harmonic-particle simulation is not implemented yet.")
 
    def get_cass(self, z):
        """Get Casimir invariant from neural network."""
        J, cass = self.J_net(z)
        return cass
 
    def get_L(self, z):
        """Get Poisson bivector from neural network."""
        return self.L_net(z)
 
    def get_E(self, z):
        """Get energy from neural network."""
        return self.energy_net(z)
 
    def m_new(self, solver_iterations=200, tol=1e-6):
        """Advance one step using neural Crank-Nicolson dynamics."""
        z_old = self._state()
        z_new = z_old.clone()
        z_dot_old = self.neural_zdot(z_old)
 
        for _ in range(solver_iterations):
            z_prev = z_new.clone()
            z_new = z_old + 0.5 * self.dt * (z_dot_old + self.neural_zdot(z_new))
 
            rel_error = torch.norm(z_new - z_prev, dim=1) / (torch.norm(z_prev, dim=1) + 1e-12)
            if torch.all(rel_error < tol):
                break
 
        self.q, self.p = self._split_state(z_new)
        return self.q, self.p


class HarmonicNeuralIMR(HarmonicNeural):
    """N-particle harmonic system evolved with learned networks and implicit midpoint stepping."""
 
    def m_new(self, solver_iterations=200, tol=1e-6):
        """Advance one step using neural implicit midpoint dynamics."""
        z_old = self._state()
        z_new = z_old.clone()
 
        for _ in range(solver_iterations):
            z_prev = z_new.clone()
            z_mid = 0.5 * (z_old + z_new)
            z_new = z_old + self.dt * self.neural_zdot(z_mid)
 
            rel_error = torch.norm(z_new - z_prev, dim=1) / (torch.norm(z_prev, dim=1) + 1e-12)
            if torch.all(rel_error < tol):
                break
 
        self.q, self.p = self._split_state(z_new)
        return self.q, self.p

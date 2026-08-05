"""Fermi-Pasta-Ulam chain models.

The FPU chain is represented in canonical coordinates
z = [q_0, ..., q_{N-1}, p_0, ..., p_{N-1}], with fixed boundary positions
q_{-1} = q_N = 0. The Hamiltonian is

    H(q, p) = sum_i p_i^2 / (2M)
              + sum_bonds 1/2 k dq^2 + alpha/3 dq^3 + beta/4 dq^4

where each bond stretch is dq = q_i - q_{i-1}, including the two fixed-end
boundary bonds.
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


class FPUCN(object):
    """Fermi-Pasta-Ulam chain with a Crank-Nicolson step."""

    def __init__(
        self,
        M,
        dt,
        alpha,
        beta,
        init_q,
        init_p,
        k=1.0,
        device="cpu",
        dtype=torch.float32,
    ):
        self.dtype = dtype
        self.M = M
        self.dt = dt
        self.alpha = alpha
        self.beta = beta
        self.k = k
        self.device = device

        self.q = _as_batched_tensor(init_q, device=device, dtype=dtype)
        self.p = _as_batched_tensor(init_p, device=device, dtype=dtype)
        if self.q.shape != self.p.shape:
            raise ValueError(f"init_q and init_p must have the same shape, got {self.q.shape} and {self.p.shape}")

        self.dimensions = self.q.shape[1]
        self.dim = 2 * self.dimensions

    def _split_state(self, z):
        return z[:, :self.dimensions], z[:, self.dimensions:]

    def _state(self):
        return torch.cat([self.q, self.p], dim=1)

    def _bond_stretches(self, q):
        zeros = torch.zeros(q.shape[0], 1, dtype=q.dtype, device=q.device)
        q_with_boundaries = torch.cat([zeros, q, zeros], dim=1)
        return q_with_boundaries[:, 1:] - q_with_boundaries[:, :-1]

    def _potential_gradient(self, q):
        stretches = self._bond_stretches(q)
        bond_forces = self.k * stretches + self.alpha * stretches.pow(2) + self.beta * stretches.pow(3)
        return bond_forces[:, :-1] - bond_forces[:, 1:]

    def z_dot(self, z):
        q, p = self._split_state(z)
        q_dot = p / self.M
        p_dot = -self._potential_gradient(q)
        return torch.cat([q_dot, p_dot], dim=1)

    def get_E(self, z):
        """Calculate total FPU Hamiltonian energy."""
        q, p = self._split_state(z)
        stretches = self._bond_stretches(q)
        kinetic = 0.5 * (p * p).sum(dim=1) / self.M
        potential_terms = (
            0.5 * self.k * stretches.pow(2)
            + (self.alpha / 3.0) * stretches.pow(3)
            + 0.25 * self.beta * stretches.pow(4)
        )
        return kinetic + potential_terms.sum(dim=1)

    def get_L(self, z):
        """Get the canonical 2N x 2N Poisson bivector."""
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


class FPUIMR(FPUCN):
    """Fermi-Pasta-Ulam chain with an implicit midpoint rule step."""

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


class FPUNeural(FPUCN):
    """FPU chain evolved with learned energy and Poisson-structure networks."""

    def __init__(
        self,
        M,
        dt,
        alpha,
        beta,
        init_q,
        init_p,
        k=1.0,
        device="cpu",
        method="without",
        name=DEFAULT_folder_name,
        dtype=torch.float32,
    ):
        super(FPUNeural, self).__init__(
            M, dt, alpha, beta, init_q, init_p, k=k, device=device, dtype=dtype
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

        raise Exception("Implicit neural FPU simulation is not implemented yet.")

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


class FPUNeuralIMR(FPUNeural):
    """FPU chain evolved with learned networks and implicit midpoint stepping."""

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

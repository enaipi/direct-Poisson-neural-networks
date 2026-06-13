#2D Particle and Shivamoggi models

from scipy.optimize import fsolve
import numpy as np
import torch

from dpnn.training import DEFAULT_folder_name
from .base import load_models


class Particle2DIMR(object): #Implicit midpont rule
    """2D Particle with implicit midpoint rule."""
    
    def __init__(self, M, dt, alpha, init_rx,  init_ry, init_mx, init_my, zeta, device="cpu"):
        self.M = M #Hamiltonian = 1/2 p^2/M + 1/2 alpha r^2
        self.r = torch.stack([init_rx, init_ry], dim=1)
        self.p = torch.stack([init_mx, init_my], dim=1)
        self.alpha = alpha
        self.dt = dt
        self.zeta = zeta
        self.device = device

    def get_E(self, m):
        """Calculate total energy."""
        return 0.5*(m[:, 2]**2 + m[:, 3]**2)/self.M + 0.5 *self.alpha * (m[:, 0]**2 + m[:, 1]**2)

    def get_L(self, m):
        """Get 4x4 Poisson bivector."""
        B = m.shape[0]
        
        zeros = torch.zeros((B,), dtype=m.dtype, device=m.device)
        ones  = torch.ones((B,), dtype=m.dtype, device=m.device)

        L = torch.stack([
            torch.stack([zeros, zeros,  ones, zeros], dim=1),
            torch.stack([zeros, zeros, zeros,  ones], dim=1),
            torch.stack([-ones, zeros, zeros, zeros], dim=1),
            torch.stack([zeros, -ones, zeros, zeros], dim=1),
        ], dim=1)

        return L

    def f(self, rpNew, rpOld=None):
        """Residual function for implicit solver."""
        if rpOld is None:
            rpOld = np.array([self.r[0], self.r[1], self.p[0], self.p[1]])
        rp_mid = 0.5*(np.array(rpNew)+rpOld)
        rmdot = np.concatenate([rp_mid[2:4]/self.M, -self.alpha*rp_mid[0:2]])
        rmdot += -self.zeta*np.array((0.0, 0.0, rp_mid[2], rp_mid[3])) #dissipation

        rpres = rpOld-rpNew + self.dt*rmdot
        return (rpres[0], rpres[1], rpres[2], rpres[3]) 

    def m_new(self, solver_iterations=200, tol=1e-6):
        """Solve for new state using implicit midpoint rule."""
        rp_old = torch.cat([self.r, self.p], dim=1)

        rp_new = rp_old.clone()

        for _ in range(solver_iterations):
            rp_prev = rp_new.clone()

            rp_mid = 0.5*(rp_new + rp_old)
            rmdot = torch.cat([rp_mid[:, 2:4]/self.M, -self.alpha*rp_mid[:, 0:2]], dim=1)
            rmdot += -self.zeta * torch.cat([torch.zeros_like(rp_mid[:, 0:2]), rp_mid[:, 2:4]], dim=1)

            rp_new = rp_old + self.dt*rmdot

            rel_error = torch.norm(rp_new - rp_prev, dim=1) / (torch.norm(rp_prev, dim=1) + 1e-12)
            if torch.all(rel_error < tol):
                break

        else:
            not_converged = (rel_error >= tol)

            if not_converged.any():
                print(f"Max iterations reached! {not_converged.sum().item()} examples did not converge. Falling back to fsolve...")

                rp_new_np = rp_new.detach().cpu().numpy()
                rp_old_np = rp_old.detach().cpu().numpy()

                for idx in torch.where(not_converged)[0]:
                    rp_sol = fsolve(lambda x: self.f(x, rpOld=rp_old_np[idx]), rp_new_np[idx])
                    rp_new[idx] = torch.tensor(rp_sol, dtype=rp_old.dtype, device=rp_old.device)

        self.r = rp_new[:, 0:2]
        self.p = rp_new[:, 2:4]

        return rp_new[:, 0:2], rp_new[:, 2:4]


class Particle2DNeural(Particle2DIMR):
    """2D Particle with neural network models."""
    
    def __init__(self, M, dt, alpha, init_rx,  init_ry, init_mx, init_my, zeta, device="cpu", method = "without", name = DEFAULT_folder_name):
        super(Particle2DNeural, self).__init__(M, dt, alpha, init_rx, init_ry, init_mx, init_my, zeta, device=device)
        
        self.device = device
        self.energy_net, self.L_net, self.J_net, self.A = load_models(name = name, method = method, device = device)
        self.method = method

    def neural_zdot(self, z):
        """Calculate Hamiltonian using neural network models."""
        z_tensor = torch.tensor(z, dtype=torch.float32, requires_grad=True, device=self.device)
        En = self.energy_net(z_tensor)

        E_z = torch.autograd.grad(En.sum(), z_tensor, only_inputs=True)[0]
        E_z = torch.flatten(E_z)

        if self.method == "soft" or self.method == "without":
            L = self.L_net(z_tensor).detach().cpu().numpy()[0]
            hamiltonian = np.matmul(L, E_z.detach().cpu().numpy())
        else:
            raise Exception("Implicit not implemented for P2D yet.")

        return hamiltonian

    def f(self, rpNew, rpOld=None):
        """Residual function for implicit solver."""
        if rpOld is None:
            rpOld = np.concatenate([self.r.cpu().numpy(), self.p.cpu().numpy()])
        rp_mid = 0.5*(np.array(rpNew)+rpOld)

        zd = self.neural_zdot(rp_mid)
        res = np.array(rpOld) - np.array(rpNew) + self.dt*zd
        return res

    def get_cass(self, z):
        """Get Casimir invariant from neural network."""
        J, cass = self.J_net(z)
        return cass

    def get_L(self, z):
        """Get Poisson bivector from neural network."""
        L = self.L_net(z)
        return L

    def get_E(self, z):
        """Get energy from neural network."""
        E = self.energy_net(z)
        return E

    def m_new(self, solver_iterations=200, tol=1e-6):
        """Solve for new state using neural networks."""
        rp_old = torch.cat([self.r, self.p], dim=1)

        rp_new = rp_old.clone()

        for _ in range(solver_iterations):
            rp_prev = rp_new.clone()

            rp_mid = 0.5*(rp_new + rp_old).requires_grad_(True)
            
            En = self.energy_net(rp_mid)
            E_z = torch.autograd.grad(En.sum(), rp_mid, only_inputs=True, retain_graph=True)[0]
            L = self.L_net(rp_mid)
            zd = torch.bmm(L, E_z.unsqueeze(-1)).squeeze(-1)

            rp_new = rp_old + self.dt*zd

            rel_error = torch.norm(rp_new - rp_prev, dim=1) / (torch.norm(rp_prev, dim=1) + 1e-12)
            if torch.all(rel_error < tol):
                break

        else:
            not_converged = (rel_error >= tol)

            if not_converged.any():
                print(f"Max iterations reached! {not_converged.sum().item()} examples did not converge. Falling back to fsolve...")

                rp_new_np = rp_new.detach().cpu().numpy()
                rp_old_np = rp_old.detach().cpu().numpy()

                for idx in torch.where(not_converged)[0]:
                    rp_sol = fsolve(lambda x: self.f(x, rpOld=rp_old_np[idx]), rp_new_np[idx])
                    rp_new[idx] = torch.tensor(rp_sol, dtype=rp_old.dtype, device=rp_old.device)

        self.r = rp_new[:, 0:2]
        self.p = rp_new[:, 2:4]

        return rp_new[:, 0:2], rp_new[:, 2:4]

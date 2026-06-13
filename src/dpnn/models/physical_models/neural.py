#RigidBody with Neural Network models for energy and structure tensors

from scipy.optimize import fsolve
import numpy as np
import torch

from dpnn.training import DEFAULT_folder_name
from .base import RigidBody, load_models


class Neural(RigidBody):#SeRe forward Euler
    """RigidBody with neural network energy and structure tensor models."""
    
    def __init__(self, Ix, Iy, Iz, d2E, mx, my, mz, dt, alpha, method = "without", name = DEFAULT_folder_name, device = "cpu"):
        super(Neural, self).__init__(Ix, Iy, Iz, d2E, mx, my, mz, dt, alpha, device=device)

        self.device = device

        self.energy_net, self.L_net, self.J_net, self.A = load_models(name = name, method = method, mx = self.mx, device = device)

        self.method = method
        
        self.energy_net.to(self.device)
        if hasattr(self, 'L_net') and isinstance(self.L_net, torch.nn.Module): self.L_net.to(self.device)
        if hasattr(self, 'J_net') and self.J_net is not None: self.J_net.to(self.device)

    # Get gradient of energy from NN
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
            J, cass = self.J_net(z_tensor)
            J = J.detach().cpu().numpy()
            hamiltonian = np.cross(J, E_z.detach().cpu().numpy())

        return hamiltonian

    def f(self, mNew, mOld = None):
        """Residual function for implicit solver."""
        if mOld is None:
            mOld = [self.mx, self.my, self.mz]

        zdo = self.neural_zdot(mOld)
        zd = self.neural_zdot(mNew)

        res = mOld - mNew + self.dt/2*(zdo + zd)

        return (res[0], res[1], res[2])

    def get_cass(self, z):
        """Get Casimir invariant from neural network."""
        z.requires_grad_(True)
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
    
    def _hamiltonian(self, z_tensor):
        z_tensor.requires_grad_(True)
        En = self.energy_net(z_tensor).squeeze(-1)
        
        E_z = torch.autograd.grad(En.sum(), z_tensor, create_graph=True)[0]
        
        if self.method == "soft" or self.method == "without":
            L = self.L_net(z_tensor)
            hamiltonian = torch.matmul(L, E_z.unsqueeze(-1)).squeeze(-1)
        else: # "implicit"
            J, cass = self.J_net(z_tensor)
            hamiltonian = torch.cross(J, E_z, dim=-1)
            
        return hamiltonian
    
    def m_new(self, with_entropy=False, solver_iterations=200, tol=1e-6):
        """Solve for new state using neural network model."""
        m_old = torch.stack([self.mx, self.my, self.mz], dim=1)
        m_new = m_old.clone()

        zd_old = self._hamiltonian(m_old)

        for _ in range(solver_iterations):
            m_prev = m_new.clone()

            zd_new = self._hamiltonian(m_prev)
            m_new = m_old + 0.5 * self.dt * (zd_old + zd_new)

            diff = torch.norm(m_new - m_prev, dim=1)
            denom = torch.norm(m_prev, dim=1) + 1e-12
            rel_error = diff / denom

            if torch.all(rel_error < tol):
                break
        else:
            not_converged = (rel_error >= tol)
            
            if not_converged.any():
                print(f"Max iterations reached! {not_converged.sum().item()} examples did not converge. Falling back to fsolve...")

                m_new_np = m_new.detach().cpu().numpy()
                m_old_np = m_old.detach().cpu().numpy()

                for idx in torch.where(not_converged)[0]:
                    m_sol = fsolve(lambda x: self.f(x, mOld=m_old_np[idx]), m_new_np[idx])
                    m_new[idx] = torch.tensor(m_sol, dtype=m_old.dtype, device=m_old.device)

        self.mx = m_new[:, 0]
        self.my = m_new[:, 1]
        self.mz = m_new[:, 2]

        return m_new


class RBNeuralIMR(Neural):#implicit midpoint rule
    """RigidBody with neural networks using implicit midpoint rule."""
    
    def __init__(self, Ix, Iy, Iz, d2E, mx, my, mz, dt, alpha, method = "without", name = DEFAULT_folder_name, device = "cpu"):
        super(RBNeuralIMR, self).__init__(Ix, Iy, Iz, d2E, mx, my, mz, dt, alpha, method = method, name = name, device=device)

    def f(self, mNew, mOld=None):
        """Residual function for implicit midpoint solver."""
        if mOld is None:
            mOld = [self.mx, self.my, self.mz]

        m_mid = [0.5*(mOld[i]+mNew[i]) for i in range(len(mOld))]

        zd = self.neural_zdot(m_mid)

        res = mOld - mNew + self.dt*zd

        return (res[0], res[1], res[2])

    def m_new(self, with_entropy = False, solver_iterations=200, tol=1e-6):
        """Solve for new state using implicit midpoint rule with neural networks."""
        m_old = torch.stack([self.mx, self.my, self.mz], dim=1)

        m_new = m_old.clone()

        for _ in range(solver_iterations):
            m_prev = m_new.clone()
            m_mid = 0.5 * (m_old + m_prev)
            m_mid.requires_grad_(True)

            hamiltonian = self._hamiltonian(m_mid)
            m_new = m_old + self.dt * hamiltonian

            diff = torch.norm(m_new - m_prev, dim=1)
            denom = torch.norm(m_prev, dim=1) + 1e-12
            rel_error = diff / denom

            if torch.all(rel_error < tol):
                break
        else:
            not_converged = (rel_error >= tol)
            
            if not_converged.any():
                print(f"Max iterations reached! {not_converged.sum().item()} examples did not converge. Falling back to fsolve...")

                m_new_np = m_new.detach().cpu().numpy()
                m_old_np = m_old.detach().cpu().numpy()

                for idx in torch.where(not_converged)[0]:
                    m_sol = fsolve(lambda x: self.f(x, mOld=m_old_np[idx]), m_new_np[idx])
                    m_new[idx] = torch.tensor(m_sol, dtype=m_old.dtype, device=m_old.device)

        self.mx = m_new[:, 0]
        self.my = m_new[:, 1]
        self.mz = m_new[:, 2]
        
        return m_new

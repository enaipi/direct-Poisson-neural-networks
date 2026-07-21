#Heavy Top models with different integrators and neural network variants

from scipy.optimize import fsolve
import numpy as np
import torch

from dpnn.training import DEFAULT_folder_name
from .rigid_body import RigidBody
from .base import load_models


class HeavyTopCN(RigidBody): #Crank-Nicolson
    """Heavy Top with Crank-Nicolson integrator."""
    
    def __init__(self, Ix, Iy, Iz, d2E, mx, my, mz, dt, alpha, Mgl, init_rx,  init_ry,  init_rz, device="cpu"):
        super(HeavyTopCN, self).__init__(Ix, Iy, Iz, d2E, mx, my, mz, dt, alpha, device=device)
        self.Mgl = Mgl #Hamiltonian = 1/2 M I^{-1} M + Mgl r . chi
        self.chi = torch.tensor((0.0, 0.0, 1.0), device=device)

        self.rx = torch.as_tensor(init_rx, device=self.device)
        self.ry = torch.as_tensor(init_ry, device=self.device)
        self.rz = torch.as_tensor(init_rz, device=self.device)

    def get_E(self, m):
        """Calculate total energy."""
        return super(HeavyTopCN, self).energy() + self.Mgl * self.rz

    def get_L(self, m):
        """Get 6x6 Poisson bivector for heavy top."""
        zeros = torch.zeros_like(self.mx)
        L = torch.stack([
                torch.stack([zeros, -self.mz, self.my, zeros, -self.rz, self.ry], dim=1),
                torch.stack([self.mz, zeros, -self.mx, self.rz, zeros, -self.rx], dim=1),
                torch.stack([self.my, -self.mx, zeros, -self.ry, self.rx, zeros], dim=1),
                torch.stack([zeros, -self.rz, self.ry, zeros, zeros, zeros], dim=1),
                torch.stack([self.rz, zeros, -self.rx, zeros, zeros, zeros], dim=1),
                torch.stack([-self.ry, self.rx, zeros, zeros, zeros, zeros], dim=1)
                ], dim=1)
        return L

    def f(self, mrnew, mrold = None):
        """Residual function for implicit solver."""
        m_new = np.array((mrnew[0], mrnew[1], mrnew[2]))
        r_new = np.array((mrnew[3], mrnew[4], mrnew[5]))

        if mrold is None:
            m_old = np.array((self.mx, self.my, self.mz))
            r_old = np.array((self.rx, self.ry, self.rz))
        else:
            m_old = np.array((mrold[0], mrold[1], mrold[2]))
            r_old = np.array((mrold[3], mrold[4], mrold[5]))

        m_dot_old = np.dot(self.d2E, m_old)
        m_ham_old = np.cross(m_old, m_dot_old)
        m_r_old = np.cross(r_old, self.Mgl*self.chi.cpu().numpy())
        r_m_old = np.cross(r_old, m_dot_old)

        m_dot_new = np.dot(self.d2E, m_new)
        m_ham_new = np.cross(m_new, m_dot_new)
        m_r_new = np.cross(r_new, self.Mgl*self.chi.cpu().numpy())
        r_m_new = np.cross(r_new, m_dot_new)

        m_res = m_old - m_new + (self.dt / 2) * (m_ham_old + m_r_old + m_ham_new + m_r_new)
        r_res = r_old - r_new + (self.dt / 2) * (r_m_old + r_m_new)

        return np.concatenate((m_res, r_res))
    
    def m_new(self, with_entropy = False, solver_iterations=300, tol=1e-6):
        """Solve for new state using Crank-Nicolson."""
        m_old = torch.stack([self.mx, self.my, self.mz], dim=1)
        r_old = torch.stack([self.rx, self.ry, self.rz], dim=1)

        chi_batched = self.chi.unsqueeze(0).expand(r_old.shape[0], -1)

        m_new = m_old.clone()
        r_new = r_old.clone()
    
        m_dot_old = (self.d2E @ m_old.T).T
        
        m_ham_old = torch.cross(m_old, m_dot_old, dim=1)
        m_r_old = torch.cross(r_old, self.Mgl * chi_batched, dim=1)
        r_m_old = torch.cross(r_old, m_dot_old, dim=1)

        for _ in range(solver_iterations):
            m_dot_new = (self.d2E @ m_new.T).T

            m_ham_new = torch.cross(m_new, m_dot_new, dim=1)
            m_r_new = torch.cross(r_new, self.Mgl * chi_batched, dim=1)
            r_m_new = torch.cross(r_new, m_dot_new, dim=1)

            m_next = m_old + (self.dt / 2) * (m_ham_old + m_r_old + m_ham_new + m_r_new)
            r_next = r_old + (self.dt / 2) * (r_m_old + r_m_new)

            rel_m_error = torch.norm(m_next - m_new, dim =1) / (torch.norm(m_new, dim=1) + 1e-12)
            rel_r_error = torch.norm(r_next - r_new, dim =1) / (torch.norm(r_new, dim=1) + 1e-12)

            if torch.all(rel_m_error < tol) and torch.all(rel_r_error < tol):
                m_new, r_new = m_next, r_next
                break
            
            m_new, r_new = m_next, r_next
        else:
            not_converged = torch.logical_or((rel_m_error >= tol), (rel_r_error >= tol))
            
            if not_converged.any():
                print(f"Max iterations reached! {not_converged.sum().item()} examples did not converge. Falling back to fsolve...")

                m_new_np = m_new.detach().cpu().numpy()
                m_old_np = m_old.detach().cpu().numpy()
                r_new_np = r_new.detach().cpu().numpy()
                r_old_np = r_old.detach().cpu().numpy()

                for idx in torch.where(not_converged)[0]:
                    mr_new = np.concatenate((m_new_np[idx], r_new_np[idx]))
                    mr_old = np.concatenate((m_old_np[idx], r_old_np[idx]))
                    mr_sol = fsolve(lambda x: self.f(x, mrold=mr_old), mr_new)
                    m_new[idx] = torch.tensor(mr_sol[:3], dtype=m_new.dtype, device=m_new.device)
                    r_new[idx] = torch.tensor(mr_sol[3:], dtype=r_new.dtype, device=r_new.device)

        self.mx, self.my, self.mz = m_new[:, 0], m_new[:, 1], m_new[:, 2]
        self.rx, self.ry, self.rz = r_new[:, 0], r_new[:, 1], r_new[:, 2]

        return (m_new, r_new)

    
class HeavyTopIMR(HeavyTopCN): #implicit midpoint rule
    """Heavy Top with implicit midpoint rule integrator."""

    def f(self, mrnew, mrold = None):
        """Residual function for implicit midpoint solver."""
        m_new = np.array((mrnew[0], mrnew[1], mrnew[2]))
        r_new = np.array((mrnew[3], mrnew[4], mrnew[5]))

        if mrold is None:
            m_old = np.array((self.mx, self.my, self.mz))
            r_old = np.array((self.rx, self.ry, self.rz))
        else:
            m_old = np.array((mrold[0], mrold[1], mrold[2]))
            r_old = np.array((mrold[3], mrold[4], mrold[5]))
        
        m_mid = [0.5*(m_old[i]+m_new[i]) for i in range(len(m_old))]
        r_mid = [0.5*(r_old[i]+r_new[i]) for i in range(len(r_old))]

        m_dot = np.dot(self.d2E, m_mid)
        m_ham = np.cross(m_mid, m_dot)
        m_r = np.cross(r_mid, self.Mgl*self.chi.cpu().numpy())
        r_m = np.cross(r_mid, m_dot)

        m_res = m_old - m_new + self.dt*(m_ham + m_r)
        r_res = r_old - r_new + self.dt*r_m 

        return np.concatenate((m_res, r_res))

    def m_new(self, with_entropy = False, solver_iterations=300, tol=1e-6):
        """Solve for new state using implicit midpoint rule."""
        m = torch.stack([self.mx, self.my, self.mz], dim=1)
        r = torch.stack([self.rx, self.ry, self.rz], dim=1)

        m_old = m.clone()
        r_old = r.clone()

        chi_batched = self.chi.unsqueeze(0).expand(r_old.shape[0], -1)

        for _ in range(solver_iterations):
            m_mid = 0.5 * (m_old + m)
            r_mid = 0.5 * (r_old + r)

            m_dot = (self.d2E @ m_mid.T).T

            m_ham = torch.cross(m_mid, m_dot, dim=1)
            m_r = torch.cross(r_mid, self.Mgl * chi_batched, dim=1)
            r_m = torch.cross(r_mid, m_dot, dim=1)

            m_new = m_old + self.dt * (m_ham + m_r)
            r_new = r_old + self.dt * r_m

            rel_m_error = torch.norm(m - m_new, dim =1) / (torch.norm(m, dim=1) + 1e-12)
            rel_r_error = torch.norm(r - r_new, dim =1) / (torch.norm(r, dim=1) + 1e-12)

            if torch.all(rel_m_error < tol) and torch.all(rel_r_error < tol):
                m, r = m_new, r_new
                break

            m, r = m_new, r_new
        else:
            not_converged = torch.logical_or((rel_m_error >= tol), (rel_r_error >= tol))
            
            if not_converged.any():
                print(f"Max iterations reached! {not_converged.sum().item()} examples did not converge. Falling back to fsolve...")

                m_new_np = m.detach().cpu().numpy()
                m_old_np = m_old.detach().cpu().numpy()
                r_new_np = r.detach().cpu().numpy()
                r_old_np = r_old.detach().cpu().numpy()

                for idx in torch.where(not_converged)[0]:
                    mr_new = np.concatenate((m_new_np[idx], r_new_np[idx]))
                    mr_old = np.concatenate((m_old_np[idx], r_old_np[idx]))
                    mr_sol = fsolve(lambda x: self.f(x, mrold=mr_old), mr_new)
                    m_new[idx] = torch.tensor(mr_sol[:3], dtype=m_new.dtype, device=m_new.device)
                    r_new[idx] = torch.tensor(mr_sol[3:], dtype=r_new.dtype, device=r_new.device)
        
        self.mx, self.my, self.mz = m[:, 0], m[:, 1], m[:, 2]
        self.rx, self.ry, self.rz = r[:, 0], r[:, 1], r[:, 2]

        return (m, r)


class HeavyTopNeural(HeavyTopCN):
    """Heavy Top with neural network models."""
    
    def __init__(self, Ix, Iy, Iz, d2E, mx, my, mz, dt, alpha, Mgl, init_rx, init_ry, init_rz, device="cpu", method = "without", name = DEFAULT_folder_name):
        super(HeavyTopNeural, self).__init__(Ix, Iy, Iz, d2E, mx, my, mz, dt, alpha, Mgl, init_rx, init_ry, init_rz, device=device)
        
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
            raise Exception("Implicit not implemented for HT yet.")

        return hamiltonian

    def f(self, mrNew, mrOld = None):
        """Residual function for implicit solver."""
        if mrOld is None:
            mOld = [self.mx, self.my, self.mz]
            rOld = [self.rx, self.ry, self.rz]
            mrOld = np.concatenate([mOld, rOld])

        zdo = self.neural_zdot(mrOld)
        zd = self.neural_zdot(mrNew)

        res = np.array(mrOld) - np.array(mrNew) + self.dt/2*(zdo + zd)

        return res

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
    
    def m_new(self, with_entropy = False, solver_iterations=300, tol=1e-6):
        """Solve for new state using neural networks."""
        mr_old = torch.stack([self.mx, self.my, self.mz, self.rx, self.ry, self.rz], dim=1)
        mr = mr_old.clone()
        mr.requires_grad_(True)

        En = self.energy_net(mr)
        E_z = torch.autograd.grad(En.sum(), mr, only_inputs=True, retain_graph=True)[0]
        
        L = self.L_net(mr)
        zdo = torch.bmm(L, E_z.unsqueeze(-1)).squeeze(-1)

        for _ in range(solver_iterations):
            mr_prev = mr.clone()

            mr.requires_grad_(True)
            En = self.energy_net(mr)
            E_z = torch.autograd.grad(En.sum(), mr, only_inputs=True, retain_graph=True)[0]

            L = self.L_net(mr)
            zd = torch.bmm(L, E_z.unsqueeze(-1)).squeeze(-1)

            mr = mr_old + self.dt * (zdo + zd) / 2

            rel_error = torch.norm(mr - mr_prev, dim =1) / (torch.norm(mr_prev, dim=1) + 1e-12)
            if torch.all(rel_error < tol):
                break
        else:
            not_converged = (rel_error >= tol)
            
            if not_converged.any():
                print(f"Max iterations reached! {not_converged.sum().item()} examples did not converge. Falling back to fsolve...")

                mr_new_np = mr.detach().cpu().numpy()
                mr_old_np = mr_old.detach().cpu().numpy()

                for idx in torch.where(not_converged)[0]:
                    mr_sol = fsolve(lambda x: self.f(x, mrOld=mr_old_np[idx]), mr_new_np[idx])
                    mr[idx] = torch.tensor(mr_sol, dtype=mr_old.dtype, device=mr_old.device)

        self.mx, self.my, self.mz = mr[:, 0], mr[:, 1], mr[:, 2]
        self.rx, self.ry, self.rz = mr[:, 3], mr[:, 4], mr[:, 5]

        return mr[:, :3], mr[:, 3:] 


class HeavyTopNeuralIMR(HeavyTopNeural):
    """Heavy Top with neural networks using implicit midpoint rule."""
    
    def f(self, mrNew, mrOld=None):
        """Residual function for implicit midpoint solver."""
        if mrOld is None:
            mOld = [self.mx, self.my, self.mz]
            rOld = [self.rx, self.ry, self.rz]
            mrOld = np.concatenate([mOld, rOld])
        
        mr_mid = 0.5*(np.array(mrNew)+mrOld)
        zd = self.neural_zdot(mr_mid)
        res = np.array(mrOld) - np.array(mrNew) + self.dt*zd

        return res

    def m_new(self, with_entropy = False, solver_iterations=300, tol=1e-6):
        """Solve for new state using implicit midpoint rule."""
        mr_old = torch.stack([self.mx, self.my, self.mz, self.rx, self.ry, self.rz], dim=1)
        mr = mr_old.clone()

        for _ in range(solver_iterations):
            mr_prev = mr.clone()

            mr_mid = 0.5 * (mr_old + mr)
            mr_mid.requires_grad_(True)
            
            En = self.energy_net(mr_mid)
            E_z = torch.autograd.grad(En.sum(), mr_mid, only_inputs=True, retain_graph=True)[0]

            L = self.L_net(mr_mid)
            zd = torch.bmm(L, E_z.unsqueeze(-1)).squeeze(-1)

            mr = mr_old + self.dt * zd

            rel_error = torch.norm(mr - mr_prev, dim =1) / (torch.norm(mr_prev, dim=1) + 1e-12)
            if torch.all(rel_error < tol):
                break

        else:
            not_converged = (rel_error >= tol)
            
            if not_converged.any():
                print(f"Max iterations reached! {not_converged.sum().item()} examples did not converge. Falling back to fsolve...")

                mr_new_np = mr.detach().cpu().numpy()
                mr_old_np = mr_old.detach().cpu().numpy()

                for idx in torch.where(not_converged)[0]:
                    mr_sol = fsolve(lambda x: self.f(x, mrOld=mr_old_np[idx]), mr_new_np[idx])
                    mr[idx] = torch.tensor(mr_sol, dtype=mr_old.dtype, device=mr_old.device)

        self.mx, self.my, self.mz = mr[:, 0], mr[:, 1], mr[:, 2]
        self.rx, self.ry, self.rz = mr[:, 3], mr[:, 4], mr[:, 5]

        return mr[:, :3], mr[:, 3:]

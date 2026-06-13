#3D Particle models with various integrators and neural network variants

from scipy.optimize import fsolve
import numpy as np
import torch

from dpnn.training import DEFAULT_folder_name
from .base import load_models


class Particle3DCN(object): #Crank-Nicolson
    """3D Particle with Crank-Nicolson integrator."""
    
    def __init__(self, M, dt, alpha, init_rx, init_ry, init_rz, init_mx, init_my, init_mz, device="cpu", dtype=torch.float32):
        self.dtype = dtype
        self.M = M #Hamiltonian = 1/2 p^2/M + 1/2 alpha r^2
        self.r = torch.stack([init_rx, init_ry, init_rz], dim=1).to(dtype=self.dtype)
        self.p = torch.stack([init_mx, init_my, init_mz], dim=1).to(dtype=self.dtype)
        self.alpha = alpha
        self.dt = dt

        self.device = device

    def get_E(self, m):
        """Calculate total energy."""
        return 0.5*(m[:, 3]**2 + m[:, 4]**2 + m[:, 5]**2)/self.M + 0.5 *self.alpha * (m[:, 0]**2 + m[:, 1]**2 + m[:, 2]**2)

    def get_L(self, m):
        """Get 6x6 Poisson bivector."""
        B = m.shape[0]
        zeros = torch.zeros((B,), dtype=m.dtype, device=m.device)
        ones  = torch.ones((B,), dtype=m.dtype, device=m.device)

        L = torch.stack([
            torch.stack([zeros, zeros, zeros,  ones, zeros, zeros], dim=1),
            torch.stack([zeros, zeros, zeros, zeros,  ones, zeros], dim=1),
            torch.stack([zeros, zeros, zeros, zeros, zeros,  ones], dim=1),
            torch.stack([-ones, zeros, zeros, zeros, zeros, zeros], dim=1),
            torch.stack([zeros, -ones, zeros, zeros, zeros, zeros], dim=1),
            torch.stack([zeros, zeros, -ones, zeros, zeros, zeros], dim=1),
        ], dim=1)
        return L

    def f(self, rpNew, rpOld=None):
        """Residual function for implicit solver."""
        if rpOld is None:
            rpOld = np.array([self.r[0], self.r[1], self.r[2], self.p[0], self.p[1], self.p[2]])
        rmdot = np.concatenate([rpOld[3:6]/self.M, -self.alpha*rpOld[0:3]])
        rmdotNew = np.concatenate([rpNew[3:6]/self.M, -self.alpha*rpNew[0:3]])

        rpres = rpOld-rpNew + self.dt/2*(rmdot+rmdotNew)

        return (rpres[0], rpres[1], rpres[2], rpres[3], rpres[4], rpres[5]) 

    def m_new(self, solver_iterations=200, tol=1e-6):
        """Solve for new state using Crank-Nicolson."""
        rp_old = torch.cat([self.r, self.p], dim=1)
        rp_new = rp_old.clone()

        rmdot_old = torch.cat([self.p/self.M, -self.alpha*self.r], dim=1)

        for _ in range(solver_iterations):
            rp_prev = rp_new.clone()

            rmdot_new = torch.cat([rp_new[:, 3:6]/self.M, -self.alpha*rp_new[:, 0:3]], dim=1)

            rp_new = rp_old + self.dt/2*(rmdot_old + rmdot_new)

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

        self.r = rp_new[:, 0:3]
        self.p = rp_new[:, 3:6]
        
        return rp_new[:, 0:3], rp_new[:, 3:6]


class Particle3DIMR(Particle3DCN):
    """3D Particle with implicit midpoint rule."""
    
    def f(self, rpNew, rpOld=None):
        """Residual function for implicit midpoint solver."""
        if rpOld is None:
            rpOld = np.array([self.r[0], self.r[1], self.r[2], self.p[0], self.p[1], self.p[2]])
        rp_mid = 0.5*(np.array(rpNew)+rpOld)
        rmdot = np.concatenate([rp_mid[3:6]/self.M, -self.alpha*rp_mid[0:3]])

        rpres = rpOld-rpNew + self.dt*rmdot

        return (rpres[0], rpres[1], rpres[2], rpres[3], rpres[4], rpres[5])
    
    def m_new(self, solver_iterations=200, tol=1e-6):
        """Solve for new state using implicit midpoint rule."""
        rp_old = torch.cat([self.r, self.p], dim=1)
        rp_new = rp_old.clone()

        for _ in range(solver_iterations):
            rp_prev = rp_new.clone()
            rp_mid = 0.5*(rp_old + rp_new)
            
            rmdot = torch.cat([rp_mid[:, 3:6]/self.M, -self.alpha*rp_mid[:, 0:3]], dim=1)

            rp_new = rp_old + self.dt/2*rmdot

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

        self.r = rp_new[:, 0:3]
        self.p = rp_new[:, 3:6]
        
        return rp_new[:, 0:3], rp_new[:, 3:6]


class Particle3DNeural(Particle3DCN):
    """3D Particle with neural network models."""
    
    def __init__(self, M, dt, alpha, init_rx,  init_ry,  init_rz, init_mx, init_my, init_mz, device="cpu", method = "without", name = DEFAULT_folder_name):
        super(Particle3DNeural, self).__init__(M, dt, alpha, init_rx, init_ry, init_rz, init_mx, init_my, init_mz, device=device)
        
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
            raise Exception("Implicit not implemented for P3D yet.")

        return hamiltonian

    def f(self, rpNew, rpOld=None):
        """Residual function for implicit solver."""
        if rpOld is None:
            rpOld = np.concatenate([self.r.cpu().numpy(), self.p.cpu().numpy()])

        zdo = self.neural_zdot(rpOld)
        zd = self.neural_zdot(rpNew)

        res = np.array(rpOld) - np.array(rpNew) + self.dt/2*(zdo + zd)

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
        rp_old = torch.cat([self.r, self.p], dim=1).requires_grad_(True)
        rp_new = rp_old.clone()

        En_old = self.energy_net(rp_old)
        E_z_old = torch.autograd.grad(En_old.sum(), rp_old, only_inputs=True, retain_graph=True)[0]
        
        L = self.L_net(rp_old)
        zd_old = torch.bmm(L, E_z_old.unsqueeze(-1)).squeeze(-1)

        for _ in range(solver_iterations):
            rp_prev = rp_new.clone()

            En_new = self.energy_net(rp_new)
            E_z_new = torch.autograd.grad(En_new.sum(), rp_new, only_inputs=True, retain_graph=True)[0]

            L = self.L_net(rp_old)
            zd_new = torch.bmm(L, E_z_new.unsqueeze(-1)).squeeze(-1)

            rp_new = rp_old + self.dt/2*(zd_new + zd_old)

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

        self.r = rp_new[:, 0:3]
        self.p = rp_new[:, 3:6]

        return rp_new[:, 0:3], rp_new[:, 3:6]


class Particle3DNeuralIMR(Particle3DNeural):
    """3D Particle with neural networks using implicit midpoint rule."""
    
    def f(self, rpNew, rpOld=None):
        """Residual function for implicit midpoint solver."""
        if rpOld is None:
            rpOld = np.concatenate([self.r.cpu().numpy(), self.p.cpu().numpy()])
        rp_mid = 0.5*(np.array(rpNew)+rpOld)

        zd = self.neural_zdot(rp_mid)

        res = np.array(rpOld) - np.array(rpNew) + self.dt*zd

        return res
    
    def m_new(self, solver_iterations=200, tol=1e-6):
        """Solve for new state using implicit midpoint rule."""
        rp_old = torch.cat([self.r, self.p], dim=1)
        rp_new = rp_old.clone()

        for _ in range(solver_iterations):
            rp_prev = rp_new.clone()

            rp_mid = 0.5*(rp_new + rp_old).requires_grad_(True)

            En_mid = self.energy_net(rp_mid)
            E_z_mid = torch.autograd.grad(En_mid.sum(), rp_mid, only_inputs=True, retain_graph=True)[0]

            L = self.L_net(rp_mid)
            zd_new = torch.bmm(L, E_z_mid.unsqueeze(-1)).squeeze(-1)

            rp_new = rp_old + self.dt*zd_new

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

        self.r = rp_new[:, 0:3]
        self.p = rp_new[:, 3:6]

        return rp_new[:, 0:3], rp_new[:, 3:6]


class Particle3DKeplerIMR(Particle3DIMR):
    """3D Particle with 1/r^2 potential (Kepler problem) using implicit midpoint rule."""
    
    def f(self, rpNew, rpOld=None):
        """Residual function accounting for Kepler potential."""
        if rpOld is None:
            rpOld = np.array([self.r[0], self.r[1], self.r[2], self.p[0], self.p[1], self.p[2]])
        rp_mid = 0.5*(np.array(rpNew)+rpOld)
        r_mid = rp_mid[0:3]
        rmdot = np.concatenate([rp_mid[3:6]/self.M, -self.alpha*r_mid/(np.dot(r_mid, r_mid)**(1.5)+1.0e-06)])

        rpres = rpOld-rpNew + self.dt*rmdot

        return (rpres[0], rpres[1], rpres[2], rpres[3], rpres[4], rpres[5]) 
    
    def m_new(self, solver_iterations=300, tol=1e-6):
        """Solve for new state accounting for Kepler potential."""
        rp_old = torch.cat([self.r, self.p], dim=1)
        rp_new = rp_old.clone()

        for _ in range(solver_iterations):
            rp_prev = rp_new.clone()
            rp_mid = 0.5*(rp_old + rp_new)

            r_mid, p_mid = rp_mid[:, 0:3], rp_mid[:, 3:6]

            r_norm_sq = (r_mid * r_mid).sum(dim=1, keepdim=True)
            denom = (r_norm_sq.sqrt()**3 + 1.0e-6)
            rmdot = torch.cat([p_mid / self.M,
                            -self.alpha * r_mid / denom], dim=1)

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

        self.r = rp_new[:, 0:3]
        self.p = rp_new[:, 3:6]
        
        return rp_new[:, 0:3], rp_new[:, 3:6]

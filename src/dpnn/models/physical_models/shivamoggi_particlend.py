#Shivamoggi and N-dimensional particle models

from scipy.optimize import fsolve
import numpy as np
import torch

from dpnn.training import DEFAULT_folder_name
from .base import load_models


class ShivamoggiIMR(object):
    """Shivamoggi model with implicit midpoint rule."""
    
    def __init__(self, M, dt, alpha, init_rx,  init_ry, init_rz, init_u, device="cpu"):
        self.M = M #Hamiltonian = 1/2 p^2/M + 1/2 alpha r^2
        self.u = init_u
        self.x = torch.stack((init_rx, init_ry, init_rz), dim=1)
        self.alpha = alpha
        self.dt = dt
        self.device = device

    def get_E(self, m):
        """Calculate energy."""
        return m[:, 3]**2 + m[:, 0]**2 - m[:, 2]**2

    def get_UV(self, m):
        """Get U and V vectors for Poisson structure."""
        u = m[:, 0]
        x = m[:, 1]
        y = m[:, 2]
        z = m[:, 3]
        zeros = torch.zeros_like(u, dtype=m.dtype, device=m.device)

        U = torch.cat([zeros.unsqueeze(-1), (2*u*(x+z)).unsqueeze(-1), zeros.unsqueeze(-1)], dim=1)
        V = torch.cat([x.unsqueeze(-1), zeros.unsqueeze(-1), (-z).unsqueeze(-1)], dim=1)
        return U, V

    def get_L(self, m = (0.0, 0.0, 0.0, 0.0)):
        """Get 4x4 Poisson bivector."""
        U, V = self.get_UV(m)
        zeros = torch.zeros_like(U[:,0], dtype=U.dtype, device=U.device)
        L = torch.stack([
            torch.stack([zeros, -U[:,0], -U[:,1], -U[:,2]], dim=1),
            torch.stack([ U[:,0], zeros, -V[:,2],  V[:,1]], dim=1),
            torch.stack([ U[:,1],  V[:,2], zeros, -V[:,0]], dim=1),
            torch.stack([ U[:,2], -V[:,1],  V[:,0], zeros], dim=1)
        ], dim=1)
        
        denom = (m[:,0] + m[:,3]).unsqueeze(-1).unsqueeze(-1)
        denom = denom + 1e-12 * torch.sign(denom)
        L = L / denom
        return L

    def f(self, mNew, mOld=None):
        """Residual function for implicit solver."""
        if mOld is None:
            mOld = np.array([self.u, self.x[0], self.x[1], self.x[2]])
        m_mid = 0.5*(np.array(mNew)+mOld)
        mdot = np.array([-m_mid[0]*m_mid[2], m_mid[3]*m_mid[2], m_mid[3]*m_mid[1]-m_mid[0]**2, m_mid[1]*m_mid[2]])

        mres = mOld-mNew + self.dt*mdot
        return (mres[0], mres[1], mres[2], mres[3]) 

    def _mdot(self, m):
        return torch.stack([
            -m[:, 0] * m[:, 2],
            m[:, 3] * m[:, 2],
            m[:, 3] * m[:, 1] - m[:, 0]**2,
            m[:, 1] * m[:, 2]
        ], dim=1)

    def _residual(self, mNew, mOld):
        m_mid = 0.5 * (mNew + mOld)
        mdot_val = self._mdot(m_mid)
        return mOld - mNew + self.dt * mdot_val

    def _jacobian(self, mNew, mOld):
        """Compute Jacobian matrix for Newton's method."""
        m_mid = 0.5 * (mNew + mOld)
        batch_size, dim = m_mid.shape
        
        J_mdot = torch.zeros(batch_size, dim, dim, device=self.device)
        m0, m1, m2, m3 = m_mid[:, 0], m_mid[:, 1], m_mid[:, 2], m_mid[:, 3]
        
        J_mdot[:, 0, 0] = -m2
        J_mdot[:, 0, 2] = -m0
        J_mdot[:, 1, 2] = m3
        J_mdot[:, 1, 3] = m2
        J_mdot[:, 2, 0] = -2 * m0
        J_mdot[:, 2, 1] = m3
        J_mdot[:, 2, 3] = m1
        J_mdot[:, 3, 1] = m2
        J_mdot[:, 3, 2] = m1
        
        I = torch.eye(dim, device=self.device).expand(batch_size, -1, -1)
        return -I + 0.5 * self.dt * J_mdot

    def solve(self, mOld, max_iter=20, tol=1e-8):
        """Solve using Newton's method with line search."""
        mNew = mOld.clone()
        converged_mask = torch.zeros(mOld.shape[0], dtype=torch.bool, device=self.device)

        for i in range(max_iter):
            if torch.all(converged_mask):
                print(f"All systems converged in {i} iterations.")
                return mNew
            
            active_mask = ~converged_mask
            mNew_active, mOld_active = mNew[active_mask], mOld[active_mask]
            
            f_val_active = self._residual(mNew_active, mOld_active)
            residual_norms_active = torch.linalg.norm(f_val_active, dim=1)
            
            newly_converged_mask = residual_norms_active < tol
            converged_mask[active_mask] = newly_converged_mask
            
            update_mask = ~newly_converged_mask
            if not torch.any(update_mask):
                continue

            mNew_update = mNew_active[update_mask]
            mOld_update = mOld_active[update_mask]
            f_val_update = f_val_active[update_mask]
            residual_norms_update = residual_norms_active[update_mask]

            J_val = self._jacobian(mNew_update, mOld_update)
            delta_m = torch.linalg.solve(J_val, -f_val_update)

            num_updates = mNew_update.shape[0]
            alphas = torch.ones(num_updates, 1, device=self.device)
            mNew_candidate = mNew_update + alphas * delta_m
            
            for _ in range(10):
                new_norms = torch.linalg.norm(self._residual(mNew_candidate, mOld_update), dim=1)
                worse_mask = new_norms > residual_norms_update
                if not torch.any(worse_mask): break
                alphas[worse_mask] *= 0.5
                mNew_candidate[worse_mask] = mNew_update[worse_mask] + alphas[worse_mask] * delta_m[worse_mask]
            
            active_indices = torch.where(active_mask)[0]
            update_indices = active_indices[update_mask]
            mNew[update_indices] = mNew_candidate

        not_converged = ~converged_mask
        if not_converged.any():
            print(f"Warning: {not_converged.sum().item()} systems did not converge. Using fsolve as fallback...")

            mOld_np = mOld[not_converged].detach().cpu().numpy()
            mNew_np = mNew[not_converged].detach().cpu().numpy()
            indices = torch.where(not_converged)[0]

            for i, idx in enumerate(indices):
                def fun(x):
                    return np.array(self.f(x, mOld=mOld_np[i]))
                sol = fsolve(fun, mNew_np[i])
                mNew[idx] = torch.tensor(sol, dtype=mNew.dtype, device=mNew.device)
        return mNew
    
    def m_new(self, solver_iterations=300, tol=1e-6):
        """Solve for new state using Newton's method."""
        um_old = torch.cat([self.u.unsqueeze(-1), self.x], dim=1)
        um_new = self.solve(um_old, max_iter=solver_iterations, tol=tol)
        
        self.u = um_new[:, 0].detach()
        self.x = um_new[:, 1:4].detach()
        
        return um_new.detach()


class ShivamoggiNeural(ShivamoggiIMR):
    """Shivamoggi model with neural network energy and Poisson structure."""
    
    def __init__(self, M, dt, alpha, init_rx,  init_ry, init_rz, init_u, device="cpu", method = "without", name = DEFAULT_folder_name):
        super(ShivamoggiNeural, self).__init__(M, dt, alpha, init_rx,  init_ry, init_rz, init_u, device=device)
        
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
            raise Exception("Implicit not implemented for Shivamoggi yet.")

        return hamiltonian

    def f(self, mNew, mOld=None):
        """Residual function for implicit solver."""
        if mOld is None:
            mOld = np.array([self.u, self.x[0], self.x[1], self.x[2]])
        m_mid = 0.5*(np.array(mNew)+mOld)

        zd = self.neural_zdot(m_mid)
        res = np.array(mOld) - np.array(mNew) + self.dt*zd
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

    def m_new(self, solver_iterations=300, tol=5e-6):
        """Solve for new state using neural networks."""
        um_old = torch.cat([self.u.unsqueeze(-1), self.x], dim=1)
        um_new = um_old.clone()

        for _ in range(solver_iterations):
            um_prev = um_new.clone()

            um_mid = 0.5*(um_old + um_new).requires_grad_(True)

            En = self.energy_net(um_mid)
            E_z = torch.autograd.grad(En.sum(), um_mid, only_inputs=True, retain_graph=True)[0]
            L = self.L_net(um_mid)
            umdot = torch.bmm(L, E_z.unsqueeze(-1)).squeeze(-1)
            
            um_new = um_old + self.dt*umdot

            rel_error = torch.norm(um_new - um_prev, dim=1) / (torch.norm(um_prev, dim=1) + 1e-12)
            if torch.all(rel_error < tol):
                break
        else:
            not_converged = (rel_error >= tol)

            if not_converged.any():
                print(f"Max iterations reached! {not_converged.sum().item()} examples did not converge. Falling back to fsolve...")

                um_new_np = um_new.detach().cpu().numpy()
                um_old_np = um_old.detach().cpu().numpy()

                for idx in torch.where(not_converged)[0]:
                    um_sol = fsolve(lambda x: self.f(x, mOld=um_old_np[idx]), um_new_np[idx])
                    um_new[idx] = torch.tensor(um_sol, dtype=um_old.dtype, device=um_old.device)

        self.u = um_new[:, 0]
        self.x = um_new[:, 1:4]

        return um_new


class ParticleNDCN(object):  # Crank–Nicolson, arbitrary dimension
    """N-dimensional harmonic oscillator with Crank-Nicolson integrator."""
    
    def __init__(self, D, M, dt, alpha, init_r=None, init_p=None, B=1, device="cpu"):
        """
        N-dimensional harmonic oscillator
        Hamiltonian: H = 1/2 p^2 / M + 1/2 alpha r^2
        """
        self.D = D
        self.M = M
        self.dt = dt
        self.alpha = alpha
        self.device = device

        if init_r is None:
            init_r = torch.randn(B, D, device=device)
        if init_p is None:
            init_p = torch.randn(B, D, device=device)

        self.r = init_r.to(device)
        self.p = init_p.to(device)

    def get_E(self, m):
        """Calculate total energy."""
        r = m[:, :self.D]
        p = m[:, self.D:]
        return 0.5 * (p.pow(2).sum(dim=1)) / self.M + 0.5 * self.alpha * (r.pow(2).sum(dim=1))

    def get_L(self, m):
        """Get Poisson bivector."""
        B = m.shape[0]
        D = self.D
        device = m.device
        dtype = m.dtype

        L = torch.zeros((B, 2 * D, 2 * D), dtype=dtype, device=device)
        I = torch.eye(D, dtype=dtype, device=device).expand(B, -1, -1)
        L[:, :D, D:] = I / self.M
        L[:, D:, :D] = -self.alpha * I
        return L

    def f(self, rpNew, rpOld=None):
        """Residual function for implicit solver."""
        if rpOld is None:
            rpOld = np.concatenate([self.r[0].cpu().numpy(), self.p[0].cpu().numpy()])

        r_old, p_old = rpOld[: self.D], rpOld[self.D :]
        r_new, p_new = rpNew[: self.D], rpNew[self.D :]

        rmdot_old = np.concatenate([p_old / self.M, -self.alpha * r_old])
        rmdot_new = np.concatenate([p_new / self.M, -self.alpha * r_new])

        rpres = rpOld - rpNew + self.dt / 2 * (rmdot_old + rmdot_new)

        return rpres

    def m_new(self, solver_iterations=200, tol=1e-6):
        """Solve for new state using Crank-Nicolson."""
        rp_old = torch.cat([self.r, self.p], dim=1)  # (B, 2D)
        rp_new = rp_old.clone()

        rmdot_old = torch.cat([self.p / self.M, -self.alpha * self.r], dim=1)

        for _ in range(solver_iterations):
            rp_prev = rp_new.clone()

            rmdot_new = torch.cat([rp_new[:, self.D:] / self.M, -self.alpha * rp_new[:, : self.D]], dim=1)

            rp_new = rp_old + self.dt / 2 * (rmdot_old + rmdot_new)

            rel_error = torch.norm(rp_new - rp_prev, dim=1) / (torch.norm(rp_prev, dim=1) + 1e-12)
            if torch.all(rel_error < tol):
                break
        else:
            not_converged = (rel_error >= tol)

            if not_converged.any():
                print(
                    f"Max iterations reached! {not_converged.sum().item()} examples did not converge. Falling back to fsolve..."
                )

                rp_new_np = rp_new.detach().cpu().numpy()
                rp_old_np = rp_old.detach().cpu().numpy()

                for idx in torch.where(not_converged)[0]:
                    rp_sol = fsolve(lambda x: self.f(x, rpOld=rp_old_np[idx]), rp_new_np[idx])
                    rp_new[idx] = torch.tensor(rp_sol, dtype=rp_old.dtype, device=rp_old.device)

        self.r = rp_new[:, : self.D]
        self.p = rp_new[:, self.D :]

        return self.r, self.p


class ParticleNDCNNeural(ParticleNDCN):
    """N-dimensional particle with neural network energy and Poisson structure."""
    
    def __init__(self, D, M, dt, alpha, init_r=None, init_p=None, B=1,
                 device="cpu", method="without", name="models"):
        
        super().__init__(D, M, dt, alpha, init_r=init_r, init_p=init_p, B=B, device=device)

        self.device = device
        self.energy_net, self.L_net, self.J_net, self.A = load_models(name = name, method = method, device = device)
        self.method = method

    def neural_zdot(self, z):
        """Calculate Hamiltonian using neural network models."""
        if isinstance(z, np.ndarray):
            z_tensor = torch.tensor(z, dtype=torch.float32, requires_grad=True, device=self.device)
        else:
            z_tensor = z.clone().detach().requires_grad_(True).to(self.device)

        En = self.energy_net(z_tensor)
        E_z = torch.autograd.grad(En.sum(), z_tensor, only_inputs=True)[0]

        if self.method in ("soft", "without"):
            L = self.L_net(z_tensor)  # expect shape (B, 2D, 2D)
            zdot = torch.bmm(L, E_z.unsqueeze(-1)).squeeze(-1)
        else:
            raise Exception("Implicit method not yet supported for ParticleNDCN.")

        if isinstance(z, np.ndarray):
            return zdot.detach().cpu().numpy()
        return zdot

    def f(self, rpNew, rpOld=None):
        """Residual function for implicit solver."""
        if rpOld is None:
            rpOld = torch.cat([self.r, self.p], dim=1).detach().cpu().numpy()

        zdo = self.neural_zdot(rpOld)
        zd = self.neural_zdot(rpNew)

        return np.array(rpOld) - np.array(rpNew) + self.dt / 2 * (zdo + zd)

    def get_E(self, z):
        """Get energy from neural network."""
        if isinstance(z, np.ndarray):
            z = torch.tensor(z, dtype=torch.float32, device=self.device)
        return self.energy_net(z)

    def get_L(self, z):
        """Get Poisson bivector from neural network."""
        if isinstance(z, np.ndarray):
            z = torch.tensor(z, dtype=torch.float32, device=self.device)
        return self.L_net(z)

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
            L = self.L_net(rp_new)
            zd_new = torch.bmm(L, E_z_new.unsqueeze(-1)).squeeze(-1)

            rp_new = rp_old + self.dt / 2 * (zd_new + zd_old)

            rel_error = torch.norm(rp_new - rp_prev, dim=1) / (torch.norm(rp_prev, dim=1) + 1e-12)
            if torch.all(rel_error < tol):
                break
        else:
            not_converged = (rel_error >= tol)
            if not_converged.any():
                print(f"Max iterations reached! {not_converged.sum().item()} did not converge. Falling back to fsolve...")

                rp_new_np = rp_new.detach().cpu().numpy()
                rp_old_np = rp_old.detach().cpu().numpy()

                for idx in torch.where(not_converged)[0]:
                    rp_sol = fsolve(lambda x: self.f(x, rpOld=rp_old_np[idx]), rp_new_np[idx])
                    rp_new[idx] = torch.tensor(rp_sol, dtype=rp_old.dtype, device=rp_old.device)

        self.r = rp_new[:, :self.D]
        self.p = rp_new[:, self.D:]

        return self.r, self.p

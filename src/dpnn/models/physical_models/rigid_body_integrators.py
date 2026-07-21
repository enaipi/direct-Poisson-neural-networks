#RigidBody integrator methods: Ehrenfest, Crank-Nicolson, Implicit Midpoint, RK4, Forward Euler

from scipy.optimize import fsolve
import numpy as np
import torch

from .rigid_body import RigidBody


class RBEhrenfest(RigidBody):#Ehrenfest scheme for the rigid body, Eq. 5.25a from https://doi.org/10.1016/j.physd.2019.06.006, τ=dt
    """Ehrenfest scheme for rigid body motion."""
    
    def __init__(self, Ix, Iy, Iz, d2E, mx, my, mz, dt, alpha, device="cpu"):
        super(RBEhrenfest, self).__init__(Ix, Iy, Iz, d2E, mx, my, mz, dt, alpha, device=device)

    def m_new(self, with_entropy = False):
        """Calculate new angular momentum state."""
        mOld = torch.stack([self.mx, self.my, self.mz], dim=1)

        ω = torch.matmul(self.d2E, mOld.T).T # (mx/Ix, my/Iy, mz/Iz) = dE/dm = ω
        ham = torch.cross(mOld, ω, dim=1) #m x E_m
        Mreg = torch.cross(mOld, (self.d2E @ ham.T).T, dim=1)
        Nreg = torch.cross(ham, ω, dim=1)
        reg = 0.5*self.dt * (Mreg+Nreg)

        m_new = mOld + self.dt*ham + self.dt*reg

        self.mx = m_new[:, 0]
        self.my = m_new[:, 1]
        self.mz = m_new[:, 2]

        return m_new


class RBESeReCN(RigidBody):#E-SeRe with Crank Nicolson
    """Crank-Nicolson integrator for self-regularized rigid body motion."""
    
    def __init__(self, Ix, Iy, Iz, d2E, mx, my, mz, dt, alpha, device="cpu"):
        super(RBESeReCN, self).__init__(Ix, Iy, Iz, d2E, mx, my, mz, dt, alpha, device=device)

    def f(self, mNew, mOld = None):
        """Residual function for implicit solver."""
        if mOld is None:
            mOld = [self.mx, self.my, self.mz]

        if torch.is_tensor(self.d2E):
            d2E = self.d2E.detach().cpu().numpy()

        dot = np.dot(d2E, mOld)
        ham = np.cross(mOld, dot)

        #regularized part t
        dotR = np.dot(d2E, ham)
        reg  = np.cross(dotR, mOld)

        #Hamiltionian part t+1
        dotNNew = np.dot(d2E, mNew)
        hamNew = np.cross(mNew, dotNNew)

        #regularized part t+1
        dotRNew = np.dot(d2E, hamNew)
        regNew  = np.cross(dotRNew, mNew)

        res = mOld - mNew + self.dt/2*(ham + hamNew)

        return (res[0], res[1], res[2])
    
    def _hamiltonian(self, m):
        dot = (self.d2E @ m.T).T
        ham = torch.cross(m, dot, dim=1)
        return ham
    
    def m_new(self, with_entropy = False, solver_iterations=200, tol=1e-6):
        """Solve for new state using Crank-Nicolson scheme."""
        m_old = torch.stack([self.mx, self.my, self.mz], dim=1)
        m_new = m_old.clone()

        ham_old = self._hamiltonian(m_old)

        for _ in range(solver_iterations):
            m_prev = m_new.clone()
            ham_new = self._hamiltonian(m_prev)

            m_new = m_old + 0.5 * self.dt * (ham_old + ham_new)

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


class RBIMR(RigidBody):#implicit midpoint
    """Implicit Midpoint Rule integrator for rigid body motion."""
    
    def __init__(self, Ix, Iy, Iz, d2E, mx, my, mz, dt, device="cpu", dtype=torch.float32):
        super(RBIMR, self).__init__(Ix, Iy, Iz, d2E, mx, my, mz, dt, 0.0, device=device, dtype=dtype)
        
        if torch.is_tensor(self.d2E):
             self.d2E = self.d2E.to(dtype=self.dtype)
        else:
             self.d2E = torch.as_tensor(self.d2E, device=device, dtype=self.dtype)

    def f(self, mNew, mOld = None):
        """Residual function for implicit midpoint solver."""
        if mOld is None:
            mOld = [self.mx, self.my, self.mz]
        
        if torch.is_tensor(self.d2E):
            d2E = self.d2E.detach().cpu().numpy()   

        m_mid = [0.5*(mOld[i]+mNew[i]) for i in range(len(mOld))]

        dot = np.dot(d2E, m_mid)
        ham = np.cross(m_mid, dot)

        res = mOld - mNew + self.dt*ham

        return (res[0], res[1], res[2])

    def m_new(self, with_entropy = False, solver_iterations=200, tol=1e-6):
        """Solve for new state using implicit midpoint rule."""
        m_old = torch.stack([self.mx, self.my, self.mz], dim=1)
        m_new = m_old.clone()

        for _ in range(solver_iterations):
            m_prev = m_new.clone()
            m_mid = 0.5 * (m_old + m_prev)
            m_mid.requires_grad_(True)

            dot = (self.d2E @ m_mid.T).T
            hamiltonian = torch.cross(m_mid, dot, dim=1)
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
                    m_new[idx] = torch.tensor(m_sol, dtype=self.dtype, device=self.device)

        self.mx = m_new[:, 0]
        self.my = m_new[:, 1]
        self.mz = m_new[:, 2]

        return m_new


class RBRK4(RigidBody):
    """4th order Runge-Kutta integrator for rigid body motion."""
    
    def __init__(self, Ix, Iy, Iz, d2E, mx, my, mz, dt, tau, device="cpu"):
        super(RBRK4, self).__init__(Ix, Iy, Iz, d2E, mx, my, mz, dt, 0.0, device=device)
        self.tau = tau

    def m_dot(self,m):
        L = self.get_L(m)

        d2E_m = (self.d2E @ m.T).T
        LdH = torch.bmm(L, d2E_m.unsqueeze(-1)).squeeze(-1)
        d2E_LdH = (self.d2E @ LdH.T).T

        M = 0.5 * torch.bmm(L, d2E_LdH.unsqueeze(-1)).squeeze(-1)
        
        return LdH + self.tau*M

    def m_new(self, with_entropy = False):
        """Compute next state using RK4 method."""
        m = torch.stack([self.mx, self.my, self.mz], dim=1)
        k1 = self.m_dot(m)
        k2 = self.m_dot(m + self.dt*k1/2)
        k3 = self.m_dot(m + self.dt*k2/2)
        k4 = self.m_dot(m + self.dt*k3)
        m_new = m + self.dt/6*(k1 + 2*k2 + 2*k3 + k4)

        self.mx = m_new[:, 0]
        self.my = m_new[:, 1]
        self.mz = m_new[:, 2]

        return m_new


class RBESeReFE(RigidBody):#SeRe forward Euler
    """Self-regularized Forward Euler integrator for rigid body motion."""
    
    def __init__(self, Ix, Iy, Iz, d2E, mx, my, mz, dt, alpha, device="cpu"):
        super(RBESeReFE, self).__init__(Ix, Iy, Iz, d2E, mx, my, mz, dt, alpha, device=device)

    def m_new(self, with_entropy = False):
        """Calculate new state using forward Euler method."""
        mOld = torch.stack([self.mx, self.my, self.mz], dim=1)
        
        dot = (self.d2E @ mOld.T).T
        ham = torch.cross(mOld, dot, dim=1)

        dotR = (self.d2E @ ham.T).T
        reg = torch.cross(dotR, mOld, dim=1)

        m = mOld + self.dt*ham - self.dt*self.tau/2*reg

        self.mx = m[:, 0]
        self.my = m[:, 1]
        self.mz = m[:, 2]

        if with_entropy: #calculate new entropy using explicit forward Euler
            sin_new = self.sin+ 0.5*(self.tau-self.dt)*self.dt/self.Ein_s() * ((self.my*self.mz*self.Jx)**2/self.Ix + (self.mz*self.mx*self.Jy)**2/self.Iy + (self.mx*self.my*self.Jz)**2/self.Iz)
            self.sin = sin_new

        if with_entropy:
            sin_new = (
                self.sin + 0.5 * (self.tau - self.dt) * self.dt / self.Ein_s() *
                ((self.my * self.mz * self.Jx) ** 2 / self.Ix +
                 (self.mz * self.mx * self.Jy) ** 2 / self.Iy +
                 (self.mx * self.my * self.Jz) ** 2 / self.Iz)
            )
            self.sin = sin_new

        return m

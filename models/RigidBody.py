 #This file contains the forward Euler solver for the self-regularized rigid body motion and the Crank-Nicolson solver for the energetic self-regularization of rigid body motion

from scipy.optimize import fsolve
from math import *
import numpy as np

import torch

from models.Model import EnergyNet, TensorNet, JacVectorNet
from learn import DEFAULT_folder_name

def load_models(name = DEFAULT_folder_name, method = "without", mx = torch.zeros((1,1)), device="cpu"):
    A, J_net = None, None
    if method == "soft":
        energy_net = torch.load(name+'/saved_models/soft_jacobi_energy', weights_only=False)  # changed weights_only=False
        energy_net.eval()   
        L_net = torch.load(name+'/saved_models/soft_jacobi_L', weights_only=False)  # changed weights_only=False
        L_net.eval()
    elif method == "without":
        energy_net = torch.load(name+'/saved_models/without_jacobi_energy', weights_only=False)  # changed weights_only=False
        energy_net.eval()

        obj = torch.load(name+'/saved_models/without_jacobi_L', weights_only=False)
        if isinstance(obj, torch.nn.Module): # old format
            L_net = obj
            L_net.eval()

        elif isinstance(obj, dict):
            L_type = obj.get('L_type', 'module')
            if L_type == 'constant':
                A = obj['A'].to(device)
                def L_net(z):
                    L = A - A.t()
                    return L.unsqueeze(0).repeat(z.size(0), 1, 1)
            elif L_type == 'module':
                L_net = obj['L_tensor']
                if isinstance(L_net, torch.nn.Module):
                    L_net.to(device)
                    L_net.eval()
            else:
                raise ValueError(f"Unknown L_type: {L_type}")
    elif method == "implicit":
        energy_net = torch.load(name+'/saved_models/implicit_jacobi_energy', weights_only=False)  # changed weights_only=False
        energy_net.eval()
        J_net = torch.load(name+'/saved_models/implicit_jacobi_J', weights_only=False)  # changed weights_only=False
        J_net.eval()
        J_net = J_net.to(device)
        def L_net(z):
            zeros = torch.zeros_like(mx)
            L = torch.stack([
                torch.stack([zeros, z[:, 2], -z[:, 1]], dim=1),
                torch.stack([-z[:, 2], zeros, z[:, 0]], dim=1),
                torch.stack([z[:, 1], -z[:, 0], zeros], dim=1)
            ], dim=1)
            return -L
    else:
        raise Exception("Unkonown method: ", method)
    
    return energy_net.to(device), L_net, J_net, A


class RigidBody(object):
    def __init__(self, Ix, Iy, Iz, d2E, mx, my, mz, dt, alpha, T=100, verbose = False, device = "cpu", dtype=torch.float32):
        self.Ix = Ix
        self.Iy = Iy
        self.Iz = Iz
        self.d2E= d2E
        self.dtype = dtype 

        if Iz > 0 and Iy > 0 and Iz > 0:
            self.Jx = 1/Iz - 1/Iy
            self.Jy = 1/Ix - 1/Iz
            self.Jz = 1/Iy - 1/Ix

        self.device = device
        self.mx = torch.as_tensor(mx, device=self.device, dtype=self.dtype)
        self.my = torch.as_tensor(my, device=self.device, dtype=self.dtype)
        self.mz = torch.as_tensor(mz, device=self.device, dtype=self.dtype)

        self.mx0 = torch.as_tensor(mx, device=self.device, dtype=self.dtype)
        self.my0 = torch.as_tensor(my, device=self.device, dtype=self.dtype)
        self.mz0 = torch.as_tensor(mz, device=self.device, dtype=self.dtype)

        self.dt = dt
        self.tau = dt*alpha

        self.hbar = 1.0545718E-34 #reduced Planck constant [SI]
        self.rho = 8.92E+03 #for copper
        self.myhbar = self.hbar * self.rho #due to rescaled mass
        self.kB = 1.38064852E-23 #Boltzmann constant
        self.umean = 4600 #mean sound speed in the low temperature solid (Copper) [SI]
        self.Einconst = pi**2/10 * pow(15/(2* pi**2), 4.0/3) * self.hbar * self.umean * pow(self.kB, -4.0/3) #Internal energy prefactor, Characterisitic volume = 1
        if verbose:
            print("Internal energy prefactor = ", self.Einconst)

        self.sin = self.ST(T) #internal entropy
        if verbose:
            print("Internal entropy set to Sin = ", self.sin, " at T=",T," K")

        self.Ein_init = 1
        self.Ein_init = self.Ein()
        self.sin_init = self.sin
        if verbose:
            print("Initial total energy = ", self.Etot())

        if verbose:
            print("RB set up.")

    def energy_x(self):
        """
        The function calculates the energy of an object in the x-direction.
        :return: the value of 0.5 times the square of the variable self.mx, divided by the variable self.Ix.
        """
        return 0.5*self.mx*self.mx/self.Ix

    def energy_y(self):
        """
        The function calculates the energy of an object in the y-direction.
        :return: the value of the expression 0.5*self.my*self.my/self.Iy.
        """
        return 0.5*self.my*self.my/self.Iy

    def energy_z(self):
        """
        The function calculates the energy of an object rotating around the z-axis.
        :return: the value of the expression 0.5*self.mz*self.mz/self.Iz.
        """
        return 0.5*self.mz*self.mz/self.Iz

    def energy(self):#returns kinetic energy
        """
        The function calculates the kinetic energy of an object based on its mass and moments of inertia.
        :return: the kinetic energy of the object.
        """
        return 0.5*(self.mx*self.mx/self.Ix+self.my*self.my/self.Iy+self.mz*self.mz/self.Iz)

    def omega_x(self):
        """
        The function calculates the angular velocity around the x-axis.
        :return: The value of `self.mx/self.Ix` is being returned.
        """
        return self.mx/self.Ix

    def omega_y(self):
        """
        The function calculates the omega_y value by dividing my by Iy.
        :return: the value of `self.my` divided by `self.Iy`.
        """
        return self.my/self.Iy

    def omega_z(self):
        """
        The function calculates the angular velocity around the z-axis.
        :return: the value of `self.mz/self.Iz`.
        """
        return self.mz/self.Iz

    def m2(self):#returns m^2
        """
        The function calculates the square of the magnitude of a vector.
        :return: the square of the magnitude of a vector.
        """
        return self.mx*self.mx+self.my*self.my+self.mz*self.mz

    def mx2(self):#returns mx^2
        """
        The function mx2 returns the value of mx squared.
        :return: the value of mx^2.
        """
        return self.mx*self.mx

    def my2(self):#returns my^2
        """
        The function `my2` returns the square of the value of `self.my`.
        :return: the square of the value of the variable "my".
        """
        return self.my*self.my

    def mz2(self):#returns mz^2
        """
        The function mz2 returns the square of the value of mz.
        :return: the square of the value of mz.
        """
        return self.mz*self.mz

    def m_magnitude(self):#returns |m|
        """
        The function returns the magnitude of a vector.
        :return: The magnitude of the vector, represented by the variable "m".
        """
        return sqrt(self.m2())

    def Ein(self):#returns normalized internal energy
        """
        The function returns the normalized internal energy.
        :return: the normalized internal energy.
        """
        #return exp(2*(self.sin-1))/self.Iz
        return self.Einconst*pow(self.sin,4.0/3)/self.Ein_init

    def Ein_s(self): #returns normalized derivative of internal energy with respect to entropy (inverse temperature)
        """
        The function returns the normalized derivative of internal energy with respect to entropy.
        :return: the normalized derivative of internal energy with respect to entropy (inverse temperature).
        """
        #return 2*exp(2*(self.sin-1))/self.Iz
        return self.Einconst*4.0/3*pow(self.sin, 1.0/3) / self.Ein_init

    def ST(self, T): #returns entropy of a Copper body with characteristic volume equal to one (Debye), [T] = K
        """
        The function calculates the entropy of a Copper body with a characteristic volume equal to one (Debye) at a given temperature.
        
        :param T: T is the temperature of the Copper body in Kelvin (K)
        :return: the entropy of a Copper body with a characteristic volume equal to one (Debye) at a given temperature T.
        """
        return 2 * pi**2/15 * self.kB * (self.kB/self.hbar *T/self.umean)**3

    def Etot(self):#returns normalized total energy
        """
        The function `Etot` returns the sum of the energy and the input energy.
        :return: the sum of the energy and the input energy, both of which are being calculated by other methods.
        """
        return self.energy() + self.Ein()

    def Sin(self): #returns normalized internal entorpy
        """
        The intenral entropy function returns the normalized internal entropy.
        :return: the normalized internal entropy.
        """
        return self.sin/self.sin_init

    def S_x(self):#kinetic entropy for rotation around x, beta = 1/4Iz
        """
        The function calculates the kinetic entropy for rotation around the x-axis.
        :return: the kinetic entropy for rotation around the x-axis.
        """
        m2 = self.m2()
        return -m2/self.Ix - 0.5*0.25/self.Iz*(m2-self.mx0*self.mx0)**2

    def S_z(self):#kinetic entropy for rotation around z
        """
        The function calculates the kinetic entropy for rotation around the z-axis.
        :return: the kinetic entropy for rotation around the z-axis.
        """
        m2 = self.m2()
        return -m2/self.Iz - 0.5*0.25/self.Iz*(m2-self.mz0*self.mz0)**2

    def Phi_x(self): #Returns the Phi potential for rotation around the x-axis
        """
        The function Phi_x returns the sum of the energy and the S_x potential for rotation around the
        x-axis.
        :return: the sum of the energy and the S_x value.
        """
        return self.energy() + self.S_x()

    def Phi_z(self):
        """
        The function Phi_z returns the sum of the energy and the S_z value.
        :return: the sum of the results of two other functions: `self.energy()` and `self.S_z()`.
        """
        return self.energy() + self.S_z()
        
    def get_L(self, m):
        """
        The function `get_L` returns a 3x3 matrix `L` (Poisson bivector) based on the input parameter `m`.
        
        :param m: The parameter "m" is a scalar value
        :return: The function `get_L` returns a 3x3 numpy array `L` which is calculated using the values of `self.mx`, `self.my`, and `self.mz`.
        """
        # L = -1*np.array([[0.0, self.mz, -self.my],[-self.mz, 0.0, self.mx],[self.my, -self.mx, 0.0]])
        # return L
        zeros = torch.zeros_like(self.mx)
        L = torch.stack([
            torch.stack([zeros, self.mz, -self.my], dim=1),
            torch.stack([-self.mz, zeros, self.mx], dim=1),
            torch.stack([self.my, -self.mx, zeros], dim=1)
        ], dim=1)
        return - L
        
    def get_E(self, m):
        """
        The function "get_E" returns the energy of an object.
        
        :param m: The parameter "m" is not used in the code snippet provided. It is not clear what it represents or how it is related to the function
        :return: The method `get_E` is returning the result of the method `energy()` called on the object `self`.
        """
        return self.energy()

class RBEhrenfest(RigidBody):#Ehrenfest scheme for the rigid body, Eq. 5.25a from https://doi.org/10.1016/j.physd.2019.06.006, τ=dt
    def __init__(self, Ix, Iy, Iz, d2E, mx, my, mz, dt, alpha, device="cpu"):
        """
        The above function is the constructor for the RBEhrenfest class, which is a subclass of another
        class.
        
        :param Ix: The parameter Ix represents the moment of inertia about the x-axis
        :param Iy: The parameter Iy represents the moment of inertia about the y-axis
        :param Iz: The parameter "Iz" represents the moment of inertia about the z-axis
        :param d2E: The parameter "d2E" likely represents the second derivative of the energy with respect to time. It could be used in calculations related to the dynamics or evolution of the system
        :param mx: The parameter "mx" likely represents the x-component of the angular momentum. It could be a value that represents the angular momentumum in the x-direction
        :param my: The parameter "my" represents the angular momentumum in the y-direction
        :param mz: The parameter "mz" represents the angular momentumum in the z-direction. 
        :param dt: dt is the time step size for the simulation. It determines how small the time interval are between each step in the simulation
        :param alpha: The parameter "alpha" is a damping parameter. 
        """
        super(RBEhrenfest, self).__init__(Ix, Iy, Iz, d2E, mx, my, mz, dt, alpha, device=device)

    def m_new(self, with_entropy = False): #return new m and update RB
        """
        The function `m_new` calculates and returns a new value for the angular momentum `m` based on the
        current values of `mx`, `my`, and `mz`, as well as other variables and matrices.
        
        :param with_entropy: The parameter "with_entropy" is a boolean flag that determines whether or not to include entropy in the calculation of the new angular momentum. If it is set to True, entropy will be included in the calculation. If it is set to False (the default value), entropy will not be included, defaults to False (optional)
        :return: the updated value of the angular momentum vector, `m_new`.
        """
        #calculate
        mOld = torch.stack([self.mx, self.my, self.mz], dim=1)

        ω = torch.matmul(self.d2E, mOld.T).T # (mx/Ix, my/Iy, mz/Iz) = dE/dm = ω
        ham = torch.cross(mOld, ω, dim=1) #m x E_m
        Mreg = torch.cross(mOld, (self.d2E @ ham.T).T, dim=1)
        Nreg = torch.cross(ham, ω, dim=1)
        reg = 0.5*self.dt * (Mreg+Nreg)

        m_new = mOld + self.dt*ham + self.dt*reg

        #update
        self.mx = m_new[:, 0]
        self.my = m_new[:, 1]
        self.mz = m_new[:, 2]

        return m_new


class RBESeReCN(RigidBody):#E-SeRe with Crank Nicolson
    def __init__(self, Ix, Iy, Iz, d2E, mx, my, mz, dt, alpha, device="cpu"):
        """
        The above function is the constructor for a class called RBESeReCN, which is a subclass of another
        class.
        
        :param Ix: The parameter Ix represents the moment of inertia about the x-axis
        :param Iy: The parameter Iy represents the moment of inertia about the y-axis
        :param Iz: The parameter "Iz" represents the moment of inertia about the z-axis. It is a measure of an object's resistance to changes in its rotational motion about the z-axis
        :param d2E: The parameter "d2E" is the Hessian of energy. 
        :param mx: The parameter "mx" represents the x-component of the m vector
        :param my: The parameter "my" represents the moment of inertia in the y-direction
        :param mz: The parameter "mz" represents the moment of inertia in the z-direction
        :param dt: dt is the time step size for the simulation. It determines the granularity of the simulation and how frequently the system is updated
        :param alpha: The alpha parameter is a constant that determines the weight of the regularization.
        """
        super(RBESeReCN, self).__init__(Ix, Iy, Iz, d2E, mx, my, mz, dt, alpha, device=device)

    def f(self, mNew, mOld = None):#defines the function f zero of which is sought
        """
        The function `f` calculates the difference between the old and new values of `m` and returns the
        result.
        
        :param mNew: The parameter `mNew` represents the new value of the vector `m` in the function `f`
        :return: a tuple containing the values of `res[0]`, `res[1]`, and `res[2]`.
        """

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

        res = mOld - mNew + self.dt/2*(ham + hamNew) #+ self.dt*self.tau/4*(reg + regNew)

        return (res[0], res[1], res[2])
    
    def _hamiltonian(self, m):
        dot = (self.d2E @ m.T).T
        ham = torch.cross(m, dot, dim=1)
        return ham
    
    def m_new(self, with_entropy = False, solver_iterations=200, tol=1e-6):
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
                    m_sol = fsolve(lambda x: self.f(x, m0ld=m_old_np[idx]), m_new_np[idx])
                    m_new[idx] = torch.tensor(m_sol, dtype=m_old.dtype, device=m_old.device)

        self.mx = m_new[:, 0]
        self.my = m_new[:, 1]
        self.mz = m_new[:, 2]

        return m_new

class RBIMR(RigidBody):#implicit midpoint
    def __init__(self, Ix, Iy, Iz, d2E, mx, my, mz, dt, device="cpu", dtype=torch.float32):

        super(RBIMR, self).__init__(Ix, Iy, Iz, d2E, mx, my, mz, dt, 0.0, device=device, dtype=dtype)
        
        if torch.is_tensor(self.d2E):
             self.d2E = self.d2E.to(dtype=self.dtype)
        else:
             self.d2E = torch.as_tensor(self.d2E, device=device, dtype=self.dtype)

    def f(self, mNew, mOld = None):#defines the function f zero of which is sought
        """
        The function `f` calculates the residual of a given angular field vector `mNew` by using the
        previous angular field vector `mOld` and other variables.
        
        :param mNew: The parameter `mNew` represents the new values of the angular field components `mx`, `my`, and `mz`
        :return: a tuple containing the values of `res[0]`, `res[1]`, and `res[2]`.
        """

        if mOld is None:
            mOld = [self.mx, self.my, self.mz]
        
        if torch.is_tensor(self.d2E):
            d2E = self.d2E.detach().cpu().numpy()   

        m_mid = [0.5*(mOld[i]+mNew[i]) for i in range(len(mOld))]

        dot = np.dot(d2E, m_mid)
        ham = np.cross(m_mid, dot)

        res = mOld - mNew + self.dt*ham

        return (res[0], res[1], res[2])

    def m_new(self, with_entropy = False, solver_iterations=200, tol=1e-6): #return new m and update RB
        """
        The function `m_new` calculates and returns new values for `mx`, `my`, and `mz`, and updates the
        corresponding variables in the class.
        
        :param with_entropy: The "with_entropy" parameter is a boolean flag that determines whether or not to include entropy in the calculation of the new value of m. If set to True, entropy will be considered in the calculation. If set to False, entropy will not be considered, defaults to False (optional)
        :return: the updated values of mx, my, and mz as a tuple.
        """
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

        #update
        self.mx = m_new[:, 0]
        self.my = m_new[:, 1]
        self.mz = m_new[:, 2]

        return m_new


class RBRK4(RigidBody):
    def __init__(self, Ix, Iy, Iz, d2E, mx, my, mz, dt, tau, device="cpu"):
        """
        The function initializes an instance of the RBRK4 class with given parameters.
        
        :param Ix: The moment of inertia about the x-axis
        :param Iy: The parameter "Iy" represents the moment of inertia about the y-axis. It is a measure of an object's resistance to changes in rotation about the y-axis
        :param Iz: The parameter "Iz" represents the moment of inertia about the z-axis. It is a measure of an object's resistance to changes in its rotational motion about the z-axis
        :param d2E: The parameter "d2E" likely represents the second derivative of the energy function. It could be a function or a value that represents the rate of change of energy with respect to time
        :param mx: The parameter "mx" represents the x-component of the moment of inertia
        :param my: The parameter "my" represents the moment of inertia about the y-axis
        :param mz: The parameter "mz" represents the moment of inertia about the z-axis
        :param dt: The parameter "dt" represents the time step or time interval between each iteration or calculation in the RBRK4 class.
        """
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
        """
        The function `m_new` calculates and returns new values for `mx`, `my`, and `mz`, and updates the
        corresponding variables in the class.
        
        :param with_entropy: The "with_entropy" parameter is a boolean flag that determines whether or not to include entropy in the calculation of the new value of m. If set to True, entropy will be considered in the calculation. If set to False, entropy will not be considered, defaults to False (optional)
        :return: the updated values of mx, my, and mz as a tuple.
        """
        #calculate
        m = torch.stack([self.mx, self.my, self.mz], dim=1)
        k1 = self.m_dot(m)
        k2 = self.m_dot(m + self.dt*k1/2)
        k3 = self.m_dot(m + self.dt*k2/2)
        k4 = self.m_dot(m + self.dt*k3)
        m_new = m + self.dt/6*(k1 + 2*k2 + 2*k3 + k4)

        #update
        self.mx = m_new[:, 0]
        self.my = m_new[:, 1]
        self.mz = m_new[:, 2]

        return m_new


class RBESeReFE(RigidBody):#SeRe forward Euler
    def __init__(self, Ix, Iy, Iz, d2E, mx, my, mz, dt, alpha, device="cpu"):
        """
        The above function is the constructor for a class called RBESeReFE, which is a subclass of another
        class.
        
        :param Ix: The parameter Ix represents the moment of inertia about the x-axis
        :param Iy: The parameter "Iy" likely represents the moment of inertia about the y-axis. Moment of inertia is a measure of an object's resistance to changes in rotational motion.
        :param Iz: The parameter "Iz" represents the moment of inertia about the z-axis. 
        :param d2E: The parameter "d2E" is the Hessian of energy. 
        :param mx: The parameter "mx" likely represents the x-component of the angular field
        :param my: The parameter "my" likely represents the angular momentum in the y-direction
        :param mz: The parameter "mz" represents the angular field in the z-direction
        :param dt: dt is the time step size for the simulation. It determines the granularity of the simulation and how frequently the calculations are performed
        :param alpha: The alpha parameter is a constant that determines the strength of the regularization term in the RBESeReFE algorithm. 
        """
        super(RBESeReFE, self).__init__(Ix, Iy, Iz, d2E, mx, my, mz, dt, alpha, device=device)

    def m_new(self, with_entropy = False):
        """
        The function `m_new` calculates the new value of the angular momentum `m` based on the old value
        `mOld` and updates the values of `mx`, `my`, and `mz`, and optionally calculates the new entropy if
        `with_entropy` is True.
        
        :param with_entropy: A boolean parameter that determines whether or not to calculate the new entropy using explicit forward Euler. If set to True, the entropy will be calculated and updated. If set to False, the entropy will not be calculated, defaults to False (optional)
        :return: the updated value of the angular momentum vector, `m`.
        """
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

class Neural(RigidBody):#SeRe forward Euler
    def __init__(self, Ix, Iy, Iz, d2E, mx, my, mz, dt, alpha, method = "without", name = DEFAULT_folder_name, device = "cpu"):
        """
        The function initializes a Neural object with specified parameters and loads a pre-trained neural
        network based on the chosen method.
        
        :param Ix: The parameter Ix represents the x-component of the input vector. It is used in the initialization of the Neural class
        :param Iy: The parameter "Iy" represents the y-component of the moment of inertia of the system. Moment of inertia is a measure of an object's resistance to changes in rotational motion. In this context, it is used to describe the rotational behavior of the system along the y-axis
        :param Iz: Iz is the moment of inertia about the z-axis. It represents the resistance of an object to changes in its rotational motion around the z-axis
        :param d2E: The parameter `d2E` is a function that represents the second derivative of the energy function. It takes as input the current state of the system and returns the second derivative of the energy with respect to the state variables
        :param mx: The parameter `mx` represents the x-component of the angular momentumum 
        :param my: The parameter `my` represents the value of the angular momentumum in the y-direction. It is used in the initialization of the `Neural` class
        :param mz: The parameter `mz` represents the value of the angular momentumum in the z-direction
        :param dt: dt is the time step size used in the simulation. It determines the granularity of the time intervals at which the neural network is updated and the system dynamics are computed
        :param alpha: The parameter "alpha" is a value used in the initialization of the Neural class, damping parameter.
        :param method: The "method" parameter is used to specify the method for calculating the energy and L matrices in the Neural class. It can take one of the following values:, defaults to without (optional)
        :param name: The `name` parameter is a string that represents the folder name where the saved models are located. It is used to load the pre-trained neural network models for energy and L (Lagrangian) calculations. The `name` parameter is used to construct the file paths for loading the models
        """
        super(Neural, self).__init__(Ix, Iy, Iz, d2E, mx, my, mz, dt, alpha, device=device)

        self.device = device

        self.energy_net, self.L_net, self.J_net, self.A = load_models(name = name, method = method, mx = self.mx, device = device)

        self.method = method
        
        self.energy_net.to(self.device)
        if hasattr(self, 'L_net') and isinstance(self.L_net, torch.nn.Module): self.L_net.to(self.device)
        if hasattr(self, 'J_net') and self.J_net is not None: self.J_net.to(self.device)

    # Get gradient of energy from NN
    def neural_zdot(self, z):
        """
        The function `neural_zdot` calculates the Hamiltonian of a neural network given a set of input
        values.
        
        :param z: The parameter `z` is the input to the `neural_zdot` function. It is a tensor or array that represents the input data for the neural network.
        :return: the hamiltonian, which is a numpy array.
        """
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

    def f(self, mNew, mOld = None):#defines the function f zero of which is sought
        """
        The function `f` calculates the difference between two sets of values and returns the result.
        
        :param mNew: The parameter `mNew` represents the new values of `mx`, `my`, and `mz`
        :return: a tuple containing the values of `res[0]`, `res[1]`, and `res[2]`.
        """
        if mOld is None:
            mOld = [self.mx, self.my, self.mz]

        zdo = self.neural_zdot(mOld)
        zd = self.neural_zdot(mNew)

        res = mOld - mNew + self.dt/2*(zdo + zd)

        return (res[0], res[1], res[2])

    def get_cass(self, z):
        """
        The function `get_cass` takes in a parameter `z`, converts it to a tensor, passes it through a
        neural network `J_net`, and returns the output `cass` as a numpy array.
        
        :param z: The parameter `z` is a numerical input that is used as an input to the neural network `J_net`. It is converted to a tensor using `torch.tensor` with a data type of `torch.float32` and `requires_grad` set to `True`. The `requires_grad` flag
        :return: the value of `cass` as a NumPy array.
        """
        # z_tensor = torch.tensor(z, dtype=torch.float32, requires_grad=True, device=self.device)
        z.requires_grad_(True)
        J, cass = self.J_net(z)
        return cass

    def get_L(self, z):
        """
        The function `get_L` takes a tensor `z`, passes it through a neural network `L_net`, and returns the
        output `L` as a numpy array.
        
        :param z: The parameter `z` is a numerical input that is used as an input to the `L_net` neural network. It is converted to a tensor using `torch.tensor` with a data type of `torch.float32`. The `requires_grad=True` argument indicates that gradients will be computed for this
        :return: the value of L, which is obtained by passing the input z through the L_net neural network and converting the result to a numpy array.
        """
        # z_tensor = torch.tensor(z, dtype=torch.float32, requires_grad=True, device=self.device)
        L = self.L_net(z)
        return L

    def get_E(self, z):
        """
        The function `get_E` takes a parameter `z`, converts it to a tensor, passes it through a neural
        network called `energy_net`, and returns the resulting energy value.
        
        :param z: The parameter `z` is a numerical input that is used as an input to the `energy_net` neural network. It is converted to a tensor using `torch.tensor` and is set to have a data type of `torch.float32`. The `requires_grad=True` argument indicates that gradients will
        :return: the value of E, which is the output of the energy_net model when given the input z.
        """
        # z_tensor = torch.tensor(z, dtype=torch.float32, requires_grad=True, device=self.device)
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
            # J = J.squeeze(0)
            # cass = cass.squeeze(0)
            hamiltonian = torch.cross(J, E_z, dim=-1)
            
        return hamiltonian
    
    def m_new(self, with_entropy=False, solver_iterations=200, tol=1e-6):
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
    def __init__(self, Ix, Iy, Iz, d2E, mx, my, mz, dt, alpha, method = "without", name = DEFAULT_folder_name, device = "cpu"):
        """
        The above function is the constructor for a class called RBNeuralIMR, which is a subclass of another
        class.
        
        :param Ix: The parameter Ix represents the moment of inertia about the x-axis. It is a measure of an object's resistance to changes in its rotational motion about the x-axis
        :param Iy: The parameter "Iy" represents the moment of inertia about the y-axis. 
        :param Iz: The parameter "Iz" represents the moment of inertia about the z-axis. 
        :param d2E: The parameter "d2E" represents the second derivative of the energy function. 
        :param mx: The parameter "mx" represents the x-component of the angular momentum. It is used in the RBNeuralIMR class initialization to define the initial angular momentum state of the system
        :param my: The parameter "my" represents the angular momentum in the y-direction. It is used in the RBNeuralIMR class initialization to set the initial angular momentum of the system
        :param mz: The parameter "mz" represents the angular momentum in the z-direction
        :param dt: dt is the time step size for the simulation. It determines the granularity of the simulation, with smaller values resulting in more accurate but slower simulations
        :param alpha: The alpha parameter is a constant that determines the strength of the damping in the system. It is used in the calculation of the damping force in the RBNeuralIMR class
        :param method: The "method" parameter is used to specify the method to be used for the calculation. The default value is set to "without", defaults to without (optional)
        :param name: The name parameter is used to specify the folder name where the results of the RBNeuralIMR class will be saved. If no name is provided, it will use the DEFAULT_folder_name
        """
        super(RBNeuralIMR, self).__init__(Ix, Iy, Iz, d2E, mx, my, mz, dt, alpha, method = method, name = name, device=device)

    def f(self, mNew, mOld=None):#defines the function f zero of which is sought
        """
        The function `f` calculates the difference between the old and new values of `m` and adds the
        product of the time step `dt` and the derivative of `z` to it.
        
        :param mNew: The parameter `mNew` represents the new values of `mx`, `my`, and `mz`
        :return: a tuple containing three values: res[0], res[1], and res[2].
        """
        if mOld is None:
            mOld = [self.mx, self.my, self.mz]

        m_mid = [0.5*(mOld[i]+mNew[i]) for i in range(len(mOld))]

        zd = self.neural_zdot(m_mid)

        res = mOld - mNew + self.dt*zd

        return (res[0], res[1], res[2])


    def m_new(self, with_entropy = False, solver_iterations=200, tol=1e-6):
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


class HeavyTopCN(RigidBody): #Crank-Nicolson
    def __init__(self, Ix, Iy, Iz, d2E, mx, my, mz, dt, alpha, Mgl, init_rx,  init_ry,  init_rz, device="cpu"):
        """
        The function initializes the HeavyTopCN class with given parameters. CN stands for the Crank-Nicholson method.
        
        :param Ix: The moment of inertia about the x-axis
        :param Iy: The parameter Iy represents the moment of inertia about the y-axis of a heavy top
        :param Iz: The parameter "Iz" represents the moment of inertia of the heavy top around the z-axis. 
        :param d2E: The parameter "d2E" represents the second derivative of the energy function. It is a function that calculates the second derivative of the energy with respect to the angles of rotation
        :param mx: The mx parameter represents the x-component of the angular momentumum vector
        :param my: The parameter "my" represents the mass of the object along the y-axis in the HeavyTopCN class
        :param mz: The parameter `mz` represents the z-component of the angular momentumum vector
        :param dt: dt is the time step size for the simulation. It determines how small the time intervals are between each update of the system
        :param alpha: The parameter alpha is a constant used in the calculation of the time derivative of the angular momentumum. It is typically a small value that determines the accuracy and stability of the numerical integration method used to solve the equations of motion
        :param Mgl: The parameter Mgl represents the potential energy due to gravity for the heavy top system. It is used in the Hamiltonian equation to calculate the total energy of the system
        :param init_rx: The initial x-coordinate of the position vector r
        :param init_ry: The parameter `init_ry` represents the initial value for the y-coordinate of the vector `r`. It is used in the initialization of the `HeavyTopCN` class
        :param init_rz: The parameter `init_rz` represents the initial value of the z-coordinate of the vector `r`. It is used in the initialization of the `HeavyTopCN` class
        """
        super(HeavyTopCN, self).__init__(Ix, Iy, Iz, d2E, mx, my, mz, dt, alpha, device=device)
        self.Mgl = Mgl #Hamiltonian = 1/2 M I^{-1} M + Mgl r . chi
        # self.chi = np.array((0.0, 0.0, 1.0))
        self.chi = torch.tensor((0.0, 0.0, 1.0), device=device)

        # self.r = np.array((init_rx, init_ry, init_rz))
        self.rx = torch.as_tensor(init_rx, device=self.device)
        self.ry = torch.as_tensor(init_ry, device=self.device)
        self.rz = torch.as_tensor(init_rz, device=self.device)
        

    def get_E(self, m):
        """
        The function `get_E` calculates the total energy of an object by adding the energy of the object's
        parent class and the product of the object's mass, gravitational acceleration, and the dot product
        of the object's position vector and a given vector.
        
        :param m: The parameter `m` represents the mass of the object
        :return: the sum of the energy calculated by the parent class (using the `energy()` method) and the dot product of `self.r` and `self.chi`, multiplied by `self.Mgl`.
        """
        #return super(HeavyTopCN,self).energy() + self.Mgl*np.dot(self.r, self.chi)
        return super(HeavyTopCN, self).energy() + self.Mgl * self.rz

    def get_L(self, m):
        """
        The function `get_L` returns a 6x6 numpy array `L` based on the input parameter `m` and the
        attributes `self.mz`, `self.my`, `self.mx`, and `self.r`.
        
        :param m: The parameter `m` is not defined in the code snippet you provided. It is a variable that represents the moment of inertia.
        :return: The function `get_L` returns a numpy array `L` which is a 6x6 matrix.
        """
        """L = np.array([
            [0.0, -self.mz, self.my, 0.0, -self.r[2], self.r[1]],
            [self.mz, 0.0, -self.mx, self.r[2], 0.0, -self.r[0]],
            [self.my, -self.mx, 0.0, -self.r[1], self.r[0], 0.0],
            [0.0, -self.r[2], self.r[1], 0.0, 0.0, 0.0],
            [self.r[2], 0.0, -self.r[0], 0.0, 0.0, 0.0],
            [-self.r[1], self.r[0], 0.0, 0.0, 0.0, 0.0]])"""
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

    def f(self, mrnew, mrold = None):#defines the function f zero of which is sought
        """
        The function `f` calculates the residuals for a given set of input variables.
        
        :param mrnew: The parameter `mrnew` is a tuple containing the values for `mNew` and `rNew`
        :return: a tuple containing the values of `m_res[0]`, `m_res[1]`, `m_res[2]`, `r_res[0]`, `r_res[1]`, and `r_res[2]`.
        """
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
        m_r_old = np.cross(r_old, self.Mgl*self.chi)
        r_m_old = np.cross(r_old, m_dot_old)

        m_dot_new = np.dot(self.d2E, m_new)
        m_ham_new = np.cross(m_new, m_dot_new)
        m_r_new = np.cross(r_new, self.Mgl*self.chi)
        r_m_new = np.cross(r_new, m_dot_new)

        m_res = m_old - m_new + (self.dt / 2) * (m_ham_old + m_r_old + m_ham_new + m_r_new)
        r_res = r_old - r_new + (self.dt / 2) * (r_m_old + r_m_new)

        return np.concat((m_res, r_res))
    
    def m_new(self, with_entropy = False, solver_iterations=300, tol=1e-6):
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
                    mr_new = np.concat((m_new_np[idx], r_new_np[idx]))
                    mr_old = np.concat((m_old_np[idx], r_old_np[idx]))
                    mr_sol = fsolve(lambda x: self.f(x, mrold=mr_old), mr_new)
                    m_new[idx] = torch.tensor(mr_sol[:3], dtype=m_new.dtype, device=m_new.device)
                    r_new[idx] = torch.tensor(mr_sol[3:], dtype=r_new.dtype, device=r_new.device)

        self.mx, self.my, self.mz = m_new[:, 0], m_new[:, 1], m_new[:, 2]
        self.rx, self.ry, self.rz = r_new[:, 0], r_new[:, 1], r_new[:, 2]

        return (m_new, r_new)
    
class HeavyTopIMR(HeavyTopCN): #implicit midpoint rule
    #def __init__(self, Ix, Iy, Iz, d2E, mx, my, mz, dt, alpha, Mgl, init_rx,  init_ry,  init_rz):
    #    super(HeavyTopIMR, self).__init__(Ix, Iy, Iz, d2E, mx, my, mz, dt, alpha, Mgl, init_rx,  init_ry,  init_rz)

    def f(self, mrnew, mrold = None):#defines the function f zero of which is sought
        """
        The function `f` calculates the residuals for a given set of inputs and returns them as a tuple.
        
        :param mrnew: The parameter `mrnew` is a list or tuple containing the following elements:
        :return: a tuple containing the values of `m_res[0]`, `m_res[1]`, `m_res[2]`, `r_res[0]`, `r_res[1]`, and `r_res[2]`.
        """
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
        m_r = np.cross(r_mid, self.Mgl*self.chi)
        r_m = np.cross(r_mid, m_dot)

        m_res = m_old - m_new + self.dt*(m_ham + m_r)
        r_res = r_old - r_new + self.dt*r_m 

        return np.concat((m_res, r_res))

    def m_new(self, with_entropy = False, solver_iterations=300, tol=1e-6):
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
                    mr_new = np.concat((m_new_np[idx], r_new_np[idx]))
                    mr_old = np.concat((m_old_np[idx], r_old_np[idx]))
                    mr_sol = fsolve(lambda x: self.f(x, mrold=mr_old), mr_new)
                    m_new[idx] = torch.tensor(mr_sol[:3], dtype=m_new.dtype, device=m_new.device)
                    r_new[idx] = torch.tensor(mr_sol[3:], dtype=r_new.dtype, device=r_new.device)
        
        self.mx, self.my, self.mz = m[:, 0], m[:, 1], m[:, 2]
        self.rx, self.ry, self.rz = r[:, 0], r[:, 1], r[:, 2]

        return (m, r)


class HeavyTopNeural(HeavyTopCN):
    def __init__(self, Ix, Iy, Iz, d2E, mx, my, mz, dt, alpha, Mgl, init_rx, init_ry, init_rz, device="cpu", method = "without", name = DEFAULT_folder_name):
        """
        The function initializes a HeavyTopNeural object with specified parameters and loads a neural
        network model based on the chosen method.
        
        :param Ix: The parameter Ix represents the moment of inertia of the heavy top about the x-axis
        :param Iy: The parameter "Iy" represents the moment of inertia about the y-axis of a heavy top.
        :param Iz: Iz is the moment of inertia about the z-axis. 
        :param d2E: The parameter `d2E` represents the second derivative of the energy function. 
        :param mx: The parameter `mx` represents the x-component of the angular momentum of the heavy top
        :param my: The parameter `my` represents the angular momentum in the y-axis of a heavy top.
        :param mz: The parameter `mz` represents the angular momentum of the heavy top along the z-axis
        :param dt: dt is the time step size for the simulation. It determines how small the time intervals are between each iteration of the simulation
        :param alpha: A damping parameter.
        :param Mgl: Mgl is the product of the mass of the heavy top and the acceleration due to gravity. It represents the gravitational potential energy of the system
        :param init_rx: The parameter `init_rx` represents the initial value of the x-coordinate of the heavy top's center of mass
        :param init_ry: The parameter `init_ry` represents the initial value of the y-component of the angular velocity of the heavy top. It is used to initialize the simulation of the heavy top's motion
        :param init_rz: The parameter `init_rz` represents the initial value of the rotation about the z-axis (yaw) for the HeavyTopNeural object. It is used to specify the initial orientation of the heavy top
        :param method: The "method" parameter is used to specify the method for solving the equations of motion in the HeavyTopNeural class. It can take one of the following values:, defaults to without (optional)
        :param name: The `name` parameter is a string that represents the name of the folder where the saved models are located. It is used to load the pre-trained neural network models for energy and angular momentum calculations
        """
        super(HeavyTopNeural, self).__init__(Ix, Iy, Iz, d2E, mx, my, mz, dt, alpha, Mgl, init_rx, init_ry, init_rz, device=device)
        
        self.device = device
        # implicit not implemented - exception would jaev neem raised earlier
        self.energy_net, self.L_net, self.J_net, self.A = load_models(name = name, method = method, device = device)

    # Get gradient of energy from NN
    def neural_zdot(self, z):
        """
        The `neural_zdot` function calculates the Hamiltonian of a neural network given a set of input
        values.
        
        :param z: The parameter `z` is the input to the `neural_zdot` function. It is expected to be a numerical value or an array-like object that can be converted to a tensor
        :return: The function `neural_zdot` returns the variable `hamiltonian`.
        """
        z_tensor = torch.tensor(z, dtype=torch.float32, requires_grad=True, device=self.device)
        En = self.energy_net(z_tensor)

        E_z = torch.autograd.grad(En.sum(), z_tensor, only_inputs=True)[0]
        E_z = torch.flatten(E_z)

        if self.method == "soft" or self.method == "without":
            L = self.L_net(z_tensor).detach().cpu().numpy()[0]
            hamiltonian = np.matmul(L, E_z.detach().cpu().numpy())
        else:
            raise Exception("Implicit not implemented for HT yet.")
            J, cass = self.J_net(z_tensor)
            J = J.detach().numpy()
            hamiltonian = np.cross(J, E_z.detach().numpy())

        return hamiltonian

    def f(self, mrNew, mrOld = None):#defines the function f zero of which is sought
        """
        The function `f` calculates the difference between the old and new values of `mr` and adds the
        average of the neural network outputs for the old and new values multiplied by the time step.
        
        :param mrNew: mrNew is a list containing the new values for the variables mx, my, mz, rx, ry, and rz
        :return: the result of the calculation, which is stored in the variable "res".
        """
        if mrOld is None:
            mOld = [self.mx, self.my, self.mz]
            rOld = self.r
            mrOld = np.concatenate([mOld, rOld])
        #mNew = [mrNew[0], mrNew[1], mrNew[2]]
        #rNew = [mrNew[3], mrNew[4], mrNew[5]]

        zdo = self.neural_zdot(mrOld)
        zd = self.neural_zdot(mrNew)

        res = np.array(mrOld) - np.array(mrNew) + self.dt/2*(zdo + zd)

        return res

    def get_cass(self, z):
        """
        The function `get_cass` takes in a parameter `z`, converts it to a tensor, passes it through a
        neural network `J_net`, and returns the output `cass` as a numpy array.
        
        :param z: The parameter `z` is a numerical input that is used as an input to the neural network. It is converted to a tensor using `torch.tensor` with a data type of `torch.float32`. The `requires_grad=True` argument indicates that gradients will be computed for this tensor during backprop
        :return: the value of `cass` as a NumPy array.
        """
        # z_tensor = torch.tensor(z, dtype=torch.float32, requires_grad=True, device=self.device)
        z.requires_grad_(True)
        J, cass = self.J_net(z)
        return cass

    def get_L(self, z):
        """
        The function `get_L` takes a tensor `z`, passes it through a neural network `L_net`, and returns the
        output `L` as a numpy array.
        
        :param z: The parameter `z` is a numerical input that is used as an input to the `L_net` neural network. It is converted to a tensor using `torch.tensor` and is set to have a data type of `torch.float32`. The `requires_grad=True` argument indicates that gradients will
        :return: the value of L, which is obtained by passing the input z through the L_net neural network and converting the result to a numpy array.
        """
        # z_tensor = torch.tensor(z, dtype=torch.float32, requires_grad=True, device=self.device)
        L = self.L_net(z)
        return L

    def get_E(self, z):
        """
        The function `get_E` takes a parameter `z`, converts it to a tensor, passes it through a neural
        network called `energy_net`, and returns the resulting energy value.
        
        :param z: The parameter `z` is a numerical input that is used as an input to the `energy_net` neural network. It is converted to a tensor using `torch.tensor` and is set to have a data type of `torch.float32`. The `requires_grad=True` argument indicates that gradients will
        :return: the value of E, which is the output of the energy_net model when given the input z.
        """
        # z_tensor = torch.tensor(z, dtype=torch.float32, requires_grad=True, device=self.device)
        E = self.energy_net(z)
        return E
    
    def m_new(self, with_entropy = False, solver_iterations=300, tol=1e-6):
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

# The `HeavyTopNeuralIMR` class is a subclass of `HeavyTopNeural` that defines a function `f` which
# calculates the difference between old and new values of `mr` and adds the product of `dt` and the
# derivative of `z` with respect to `mr`. IMR stands for the Implicit Midpoint Rule
class HeavyTopNeuralIMR(HeavyTopNeural):
    def f(self, mrNew):#defines the function f zero of which is sought
        """
        The function `f` calculates the difference between the old and new values of `mr` and adds the
        product of `dt` and the derivative of `z` with respect to `mr`.
        
        :param mrNew: The parameter `mrNew` is a list or array containing the new values for `mx`, `my`, `mz`, and `r`
        :return: the value of the variable "res".
        """

        mOld = [self.mx, self.my, self.mz]
        rOld = self.r
        mrOld = np.concatenate([mOld, rOld])
        mr_mid = 0.5*(np.array(mrNew)+mrOld)

        zd = self.neural_zdot(mr_mid)

        res = np.array(mrOld) - np.array(mrNew) + self.dt*zd

        return res

    def m_new(self, with_entropy = False, solver_iterations=300, tol=1e-6):
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

class Particle3DCN(object): #Crank-Nicolson
    def __init__(self, M, dt, alpha, init_rx, init_ry, init_rz, init_mx, init_my, init_mz, device="cpu", dtype=torch.float32):
        """
        The function initializes the variables M, dt, alpha, init_rx, init_ry, init_rz, init_mx, init_my,
        and init_mz.
        
        :param M: The parameter M represents the mass of the particle in the Hamiltonian equation
        :param dt: dt is the time step size. It determines the size of each time step in the simulation
        :param alpha: The alpha parameter represents the strength of the harmonic potential in the Hamiltonian. It determines how tightly the particle is confined in space. A larger value of alpha corresponds to a stronger confinement
        :param init_rx: The initial x-coordinate of the position vector
        :param init_ry: The parameter `init_ry` represents the initial y-coordinate of the position vector `r`. It is used to specify the initial position of the particle in the y-direction
        :param init_rz: The parameter `init_rz` represents the initial position in the z-direction. It is used to initialize the position vector `self.r` in the `__init__` method of the class
        :param init_mx: The initial momentum in the x-direction
        :param init_my: The parameter `init_my` represents the initial momentum in the y-direction
        :param init_mz: The parameter `init_mz` represents the initial value of the momentum component in the z-direction
        """
        
        self.dtype = dtype
        self.M = M #Hamiltonian = 1/2 p^2/M + 1/2 alpha r^2
        # self.r = np.array((init_rx, init_ry, init_rz))
        self.r = torch.stack([init_rx, init_ry, init_rz], dim=1).to(dtype=self.dtype)
        self.p = torch.stack([init_mx, init_my, init_mz], dim=1).to(dtype=self.dtype)
        self.alpha = alpha
        self.dt = dt

        self.device = device

    def get_E(self, m):
        """
        The function `get_E` calculates the total energy `E` based on the input `m` and some constants `M`
        and `alpha`.
        
        :param m: The parameter `m` is a list or tuple containing six elements. The elements represent the values of `m[0]`, `m[1]`, `m[2]`, `m[3]`, `m[4]`, and `m[5]`, where the first three give the position while the latter three position
        :return: The function `get_E` returns the value of the expression `0.5*(m[3]**2 + m[4]**2 + m[5]**2)/self.M + 0.5 *self.alpha * (m[0]**2 + m[1]**2 + m[2]**2)`.
        """
        return 0.5*(m[:, 3]**2 + m[:, 4]**2 + m[:, 5]**2)/self.M + 0.5 *self.alpha * (m[:, 0]**2 + m[:, 1]**2 + m[:, 2]**2)

    def get_L(self, m):
        """
        The function `get_L` returns a 6x6 numpy array representing a transformation matrix.
        
        :param m: The parameter `m` is a tuple with three elements representing the x, y, and z coordinates respectively. The default value for `m` is (0.0, 0.0, 0.0)
        :return: The function `get_L` returns a 6x6 numpy array `L` with the following values:
        """
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

    def f(self, rpNew, rpOld=None):#defines the function f zero of which is sought
        """
        The function `f` calculates the residual of a given set of input parameters `rpNew` by performing a
        series of mathematical operations.
        
        :param rpNew: The parameter `rpNew` is a list or array containing the new values of `r` and `p`. It should have a length of 6, where the first 3 elements represent the new values of `r` and the last 3 elements represent the new values of `p`
        :return: a tuple containing the values of `rpres[0]`, `rpres[1]`, `rpres[2]`, `rpres[3]`, `rpres[4]`, and `rpres[5]`.
        """
        if rpOld is None:
            rpOld = np.array([self.r[0], self.r[1], self.r[2], self.p[0], self.p[1], self.p[2]])
        #dEOld = np.concatenate([self.p/self.M, self.alpha*self.r])
        #rmdot = self.get_L(rpOld).dot(dEOld)
        rmdot = np.concatenate([rpOld[3:6]/self.M, -self.alpha*rpOld[0:3]])
        #print("rmdot=",rmdot)

        #Hamiltionian part t+1
        #dENew = np.concatenate([np.array(rpNew[0:3])/self.M, self.alpha*np.array(rpNew[3:6])])
        #rmdotNew = self.get_L(rpNew).dot(dENew)
        rmdotNew = np.concatenate([rpNew[3:6]/self.M, -self.alpha*rpNew[0:3]])

        rpres = rpOld-rpNew + self.dt/2*(rmdot+rmdotNew)

        return (rpres[0], rpres[1], rpres[2], rpres[3], rpres[4], rpres[5]) 
        #return rpres

    def m_new(self, solver_iterations=200, tol=1e-6): #return new r and p
        """
        The function `m_new` calculates new values for `r` and `p` using the `fsolve` function and returns
        the updated values.

        :return: The function `m_new` returns a tuple containing two tuples. The first tuple contains the values `(rx, ry, rz)` and the second tuple contains the values `(px, py, pz)`.
        """
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
    def f(self, rpNew):#defines the function f zero of which is sought
        """
        The function `f` calculates the residual of a given input `rpNew` by performing a series of
        mathematical operations.
        
        :param rpNew: The parameter `rpNew` is a numpy array that represents the new values of position and momentum. It has the following structure:
        :return: a tuple containing the values of `rpres[0]`, `rpres[1]`, `rpres[2]`, `rpres[3]`, `rpres[4]`, and `rpres[5]`.
        """
        rpOld = np.array([self.r[0], self.r[1], self.r[2], self.p[0], self.p[1], self.p[2]])
        rp_mid = 0.5*(np.array(rpNew)+rpOld)
        rmdot = np.concatenate([rp_mid[3:6]/self.M, -self.alpha*rp_mid[0:3]])

        rpres = rpOld-rpNew + self.dt*rmdot

        return (rpres[0], rpres[1], rpres[2], rpres[3], rpres[4], rpres[5])
    
    def m_new(self, solver_iterations=200, tol=1e-6): #return new r and p
        """
        The function `m_new` calculates new values for `r` and `p` using the `fsolve` function and returns
        the updated values.

        :return: The function `m_new` returns a tuple containing two tuples. The first tuple contains the values `(rx, ry, rz)` and the second tuple contains the values `(px, py, pz)`.
        """
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
    def __init__(self, M, dt, alpha, init_rx,  init_ry,  init_rz, init_mx, init_my, init_mz, device="cpu", method = "without", name = DEFAULT_folder_name):
        """
        The function initializes a Particle3DNeural object with specified parameters and loads a neural
        network based on the chosen method.
        
        :param M: The parameter M represents the mass of the particle
        :param dt: dt is the time step size for the simulation. It determines how much time elapses between each iteration of the simulation
        :param alpha: The parameter "alpha" is a value used in the calculation of the forces acting on the particle. It represents the strength of the forces relative to the mass of the particle. 
        :param init_rx: The parameter `init_rx` is the initial x-coordinate of the particle's position
        :param init_ry: The parameter `init_ry` represents the initial value for the y-component of the particle's position. It is used to initialize the particle's position in the y-direction
        :param init_rz: The parameter `init_rz` represents the initial value of the z-coordinate of the particle's position. It is used to initialize the particle's position in the z-direction
        :param init_mx: The parameter `init_mx` is the initial x-component of the particle's angular momentum
        :param init_my: The parameter `init_my` is the initial y-component of the particle's angular momentum 
        :param init_mz: The parameter `init_mz` represents the initial value of the z-component of the particle's angular momentum. It is used to initialize the particle's angular momentum in the `__init__` method of the `Particle3DNeural` class
        :param method: The "method" parameter is used to specify the method for solving the equations of motion for the Particle3DNeural object. There are three possible values for this parameter:, defaults to without (optional)
        :param name: The `name` parameter is a string that represents the name of the folder where the saved models are located. It is used to load the pre-trained neural network models for energy and L (angular momentum) calculations
        """
        super(Particle3DNeural, self).__init__(M, dt, alpha, init_rx, init_ry, init_rz, init_mx, init_my, init_mz, device=device)
        
        self.device = device
        # implicit not implemented - exception would jaev neem raised earlier
        self.energy_net, self.L_net, self.J_net, self.A = load_models(name = name, method = method, device = device)

    # Get gradient of energy from NN
    def neural_zdot(self, z):
        """
        The function `neural_zdot` calculates the Hamiltonian of a neural network model given a set of
        input parameters.
        
        :param z: The parameter `z` is a tensor representing the input to the neural network. It is of type `torch.Tensor` and has a shape determined by the specific neural network architecture being used
        :return: the Hamiltonian, which is a scalar value representing the energy of the system.
        """
        z_tensor = torch.tensor(z, dtype=torch.float32, requires_grad=True, device=self.device)
        En = self.energy_net(z_tensor)

        E_z = torch.autograd.grad(En.sum(), z_tensor, only_inputs=True)[0]
        E_z = torch.flatten(E_z)

        if self.method == "soft" or self.method == "without":
            L = self.L_net(z_tensor).detach().cpu().numpy()[0]
            hamiltonian = np.matmul(L, E_z.detach().cpu().numpy())
        else:
            raise Exception("Implicit not implemented for P3D yet.")
            J, cass = self.J_net(z_tensor)
            J = J.detach().numpy()
            hamiltonian = np.cross(J, E_z.detach().numpy())

        return hamiltonian

    def f(self, rpNew):#defines the function f zero of which is sought
        """
        The function `f` calculates the residual of a given input `rpNew` by subtracting it from `rpOld` and
        adding the product of `self.dt/2` and the sum of `zdo` and `zd`.
        
        :param rpNew: The parameter `rpNew` is a numpy array that represents the new values of `r` and `p`
        :return: the value of the variable "res".
        """

        rpOld = np.concatenate([self.r, self.p])

        zdo = self.neural_zdot(rpOld)
        zd = self.neural_zdot(rpNew)

        res = np.array(rpOld) - np.array(rpNew) + self.dt/2*(zdo + zd)

        return res

    def get_cass(self, z):
        """
        The function `get_cass` takes in a parameter `z`, converts it to a tensor, passes it through a
        neural network `J_net`, and returns the output `cass` as a numpy array.
        
        :param z: The parameter `z` is a numerical input that is used as an input to the neural network. It is converted to a tensor using `torch.tensor` with a data type of `torch.float32`. The `requires_grad=True` argument indicates that gradients will be computed for this tensor during backprop
        :return: the value of `cass` as a NumPy array.
        """
        #z_tensor = torch.tensor(z, dtype=torch.float32, requires_grad=True, device=self.device)
        #J, cass = self.J_net(z_tensor)
        #return cass.detach().cpu().numpy()
        J, cass = self.J_net(z)
        return cass

    def get_L(self, z):
        """
        The function `get_L` takes a tensor `z`, passes it through a neural network `L_net`, and returns the
        output `L` as a numpy array.
        
        :param z: The parameter `z` is a numerical input that is used as an input to the `L_net` neural network. It is converted to a tensor using `torch.tensor` with a data type of `torch.float32`. The `requires_grad=True` argument indicates that gradients will be computed with respect
        :return: the value of L, which is obtained by passing the input z through the L_net neural network and converting the result to a numpy array.
        """
        # z_tensor = torch.tensor(z, dtype=torch.float32, requires_grad=True, device=self.device)
        #L = self.L_net(z_tensor).detach().cpu().numpy()[0]
        L = self.L_net(z)
        return L

    def get_E(self, z):
        """
        The function `get_E` takes a parameter `z`, converts it to a tensor, passes it through a neural
        network called `energy_net`, and returns the resulting energy value.
        
        :param z: The parameter `z` is a numerical input that is used as an input to the `energy_net` neural network. It is converted to a tensor using `torch.tensor` and is set to have a data type of `torch.float32`. The `requires_grad=True` argument indicates that gradients will
        :return: the value of E, which is the output of the energy_net model when given the input z.
        """
        # z_tensor = torch.tensor(z, dtype=torch.float32, requires_grad=True, device=self.device)
        #E = self.energy_net(z_tensor).detach().cpu().numpy()[0]
        E = self.energy_net(z)
        return E

    #@tf.function
    def m_new(self, solver_iterations=200, tol=1e-6): #return new r and p
        """
        The function `m_new` calculates new values for `r` and `p` using the `fsolve` function and returns
        them as a tuple.

        :return: a tuple containing the updated values of `self.r` and `self.p`.
        """
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
    def f(self, rpNew, rpOld=None):#defines the function f zero of which is sought
        """
        The function `f` calculates the residual between the old and new values of `rp` and the time
        derivative of `z` using the midpoint method.
        
        :param rpNew: rpNew is a numpy array that represents the new values of the variables r and p
        :return: the value of the variable "res".
        """
        if rpOld is None:
            rpOld = np.concatenate([self.r, self.p])
        rp_mid = 0.5*(np.array(rpNew)+rpOld)

        zd = self.neural_zdot(rp_mid)

        res = np.array(rpOld) - np.array(rpNew) + self.dt*zd

        return res
    
    def m_new(self, solver_iterations=200, tol=1e-6): #return new r and p
        """
        The function `m_new` calculates new values for `r` and `p` using the `fsolve` function and returns
        them as a tuple.

        :return: a tuple containing the updated values of `self.r` and `self.p`.
        """
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
    def f(self, rpNew, rpOld=None):#defines the function f zero of which is sought
        """
        The function f calculates the residual of a given set of inputs and returns it as a tuple.
        
        :param rpNew: The parameter `rpNew` is a numpy array that represents the new values of position and momentum. It has a shape of (6,) and contains the following elements:
        :return: a tuple containing the values of `rpres[0]`, `rpres[1]`, `rpres[2]`, `rpres[3]`, `rpres[4]`, and `rpres[5]`.
        """
        if rpOld is None:
            rpOld = np.array([self.r[0], self.r[1], self.r[2], self.p[0], self.p[1], self.p[2]])
        rp_mid = 0.5*(np.array(rpNew)+rpOld)
        r_mid = rp_mid[0:3]
        rmdot = np.concatenate([rp_mid[3:6]/self.M, -self.alpha*r_mid/(np.dot(r_mid, r_mid)**(1.5)+1.0e-06)])

        rpres = rpOld-rpNew + self.dt*rmdot

        return (rpres[0], rpres[1], rpres[2], rpres[3], rpres[4], rpres[5]) 
    
    def m_new(self, solver_iterations=300, tol=1e-6): #return new r and p
        """
        The function `m_new` calculates new values for `r` and `p` using the `fsolve` function and returns
        the updated values.

        :return: The function `m_new` returns a tuple containing two tuples. The first tuple contains the values `(rx, ry, rz)` and the second tuple contains the values `(px, py, pz)`.
        """
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

class Particle2DIMR(object): #Implicit midpont rule
    def __init__(self, M, dt, alpha, init_rx,  init_ry, init_mx, init_my, zeta, device="cpu"):
        """
        The function initializes the variables M, dt, alpha, init_rx, init_ry, init_mx, init_my, and zeta.
        
        :param M: The parameter M represents the mass of the particle in the Hamiltonian
        :param dt: dt is the time step size. It determines the size of each time step in the simulation
        :param alpha: The parameter alpha represents the strength of the harmonic potential in the Hamiltonian. It determines how tightly the particle is confined to the potential well. A larger value of alpha corresponds to a stronger confinement
        :param init_rx: The initial x-coordinate of the position vector
        :param init_ry: The parameter `init_ry` represents the initial y-coordinate of the position vector. It is used to specify the initial position of the particle in the y-direction
        :param init_mx: The initial momentum in the x-direction
        :param init_my: The parameter `init_my` represents the initial momentum in the y-direction
        :param zeta: The parameter zeta represents the damping coefficient in the system. It determines the rate at which the system loses energy due to damping. A higher value of zeta leads to faster energy dissipation and damping of the system.
        """
        self.M = M #Hamiltonian = 1/2 p^2/M + 1/2 alpha r^2
        self.r = torch.stack([init_rx, init_ry], dim=1)
        self.p = torch.stack([init_mx, init_my], dim=1)
        self.alpha = alpha
        self.dt = dt
        self.zeta = zeta
        self.device = device

    def get_E(self, m):
        """
        The function `get_E` calculates the total energy `E` based on the input `m` and the class attributes `M` and `alpha`.
        
        :param m: The parameter `m` is a list or tuple containing four elements. The first two elements (`m[0]` and `m[1]`) represent the x and y components of the position vector, while the last two elements (`m[2]` and `m[3]`) represent the momentum.
        :return: the value of the expression 0.5*(m[2]**2 + m[3]**2)/self.M + 0.5 *self.alpha * (m[0]**2 + m[1]**2).
        """
        return 0.5*(m[:, 2]**2 + m[:, 3]**2)/self.M + 0.5 *self.alpha * (m[:, 0]**2 + m[:, 1]**2)

    def get_L(self, m):
        """
        The function `get_L` returns a 4x4 numpy array representing a transformation matrix.
        
        :param m: The parameter `m` is a tuple with three elements representing the x, y, and z coordinates respectively. The default value for `m` is (0.0, 0.0, 0.0)
        :return: a 4x4 numpy array called L.
        """
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

    def f(self, rpNew, rpOld=None):#defines the function f zero of which is sought
        """
        The function f calculates the residual of the difference between the old and new values of r and p,
        taking into account various factors such as mass, dissipation, and time step.
        
        :param rpNew: The parameter `rpNew` is a numpy array containing the new values of position and momentum. It has the form `[x_new, y_new, px_new, py_new]`
        :return: a tuple containing the values of `rpres[0]`, `rpres[1]`, `rpres[2]`, and `rpres[3]`.
        """
        if rpOld is None:
            rpOld = np.array([self.r[0], self.r[1], self.p[0], self.p[1]])
        rp_mid = 0.5*(np.array(rpNew)+rpOld)
        rmdot = np.concatenate([rp_mid[2:4]/self.M, -self.alpha*rp_mid[0:2]])
        rmdot += -self.zeta*np.array((0.0, 0.0, rp_mid[2], rp_mid[3])) #dissipation

        rpres = rpOld-rpNew + self.dt*rmdot
        return (rpres[0], rpres[1], rpres[2], rpres[3]) 

    def m_new(self, solver_iterations=200, tol=1e-6): #return new r and p
        """
        The function `m_new` returns new values for `r` and `p` by solving a system of equations using the `fsolve` function.
        :return: a tuple containing two tuples. The first tuple contains the values of `rx` and `ry`, and the second tuple contains the values of `px` and `py`.
        """
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
    def __init__(self, M, dt, alpha, init_rx,  init_ry, init_mx, init_my, zeta, device="cpu", method = "without", name = DEFAULT_folder_name):
        """
        The function initializes a Particle2DNeural object with specified parameters and loads a neural
        network based on the chosen method.
        
        :param M: The parameter M represents the mass of the particle
        :param dt: dt is the time step size for the simulation. It determines how much time elapses between each iteration of the simulation
        :param alpha: Provides the strength of external field.
        :param init_rx: The parameter `init_rx` represents the initial x-coordinate of the particle's position
        :param init_ry: The parameter `init_ry` represents the initial y-coordinate of the particle in a 2D space. It is used to specify the starting position of the particle along the y-axis
        :param init_mx: The parameter `init_mx` represents the initial x-component of the momentum of the particle
        :param init_my: The parameter `init_my` represents the initial y-coordinate of the momentum of the particle in a 2D system. It is used to initialize the momentum of the particle along the y-axis
        :param zeta: A friction coefficient in the equations.
        :param method: The "method" parameter is used to specify the method for solving the equations of motion in the Particle2DNeural class. It can take one of the following values:, defaults to without (optional)
        :param name: The `name` parameter is a string that represents the name of the folder where the saved models are located. It is used to load the pre-trained neural network models for energy and L (Lagrangian) calculations. 
        """
        super(Particle2DNeural, self).__init__(M, dt, alpha, init_rx, init_ry, init_mx, init_my, zeta, device=device)
        
        self.device = device
        # implicit not implemented - exception would jaev neem raised earlier
        self.energy_net, self.L_net, self.J_net, self.A = load_models(name = name, method = method, device = device)

    # Get gradient of energy from NN
    def neural_zdot(self, z):
        """
        The function `neural_zdot` calculates the Hamiltonian for a given input `z` using neural networks.
        
        :param z: The parameter `z` is a tensor representing the input to the neural network. It is of type `torch.Tensor` and has a shape determined by the specific neural network architecture being used.
        """
        z_tensor = torch.tensor(z, dtype=torch.float32, requires_grad=True, device=self.device)
        En = self.energy_net(z_tensor)

        E_z = torch.autograd.grad(En.sum(), z_tensor, only_inputs=True)[0]
        E_z = torch.flatten(E_z)

        if self.method == "soft" or self.method == "without":
            L = self.L_net(z_tensor).detach().cpu().numpy()[0]
            hamiltonian = np.matmul(L, E_z.detach().cpu().numpy())
        else:
            raise Exception("Implicit not implemented for P2D yet.")
            J, cass = self.J_net(z_tensor)
            J = J.detach().numpy()
            hamiltonian = np.cross(J, E_z.detach().numpy())
        return hamiltonian

    def f(self, rpNew, rpOld=None):#defines the function f zero of which is sought
        """
        The function `f` calculates the residual between the old and new values of `rp` and the time
        derivative of `z`.
        
        :param rpNew: The parameter `rpNew` represents the new values of `r` and `p` that are being passed to the function `f`
        :return: the value of the variable "res".
        """
        if rpOld is None:
            rpOld = np.concatenate([self.r, self.p])
        rp_mid = 0.5*(np.array(rpNew)+rpOld)

        zd = self.neural_zdot(rp_mid)
        res = np.array(rpOld) - np.array(rpNew) + self.dt*zd
        return res

    def get_cass(self, z):
        """
        The function `get_cass` takes in a parameter `z`, converts it to a tensor, passes it through a
        neural network `J_net`, and returns the output `cass` as a numpy array.
        
        :param z: The parameter `z` is a numerical input that is used as an input to the neural network. It is converted to a tensor using `torch.tensor` with a data type of `torch.float32`. The `requires_grad=True` argument indicates that gradients will be computed for this tensor during backprop
        :return: The function `get_cass` returns the value of `cass` as a NumPy array.
        """
        J, cass = self.J_net(z)
        return cass

    def get_L(self, z):
        """
        The function `get_L` takes a tensor `z`, passes it through a neural network `L_net`, and returns the
        output `L` as a numpy array.
        
        :param z: The parameter `z` is a numerical input that is used as an input to the `L_net` neural network. It is converted to a tensor using `torch.tensor` with a data type of `torch.float32`. 
        :return: the value of L, which is obtained by passing the input z through the L_net neural network and converting the result to a numpy array.
        """
        L = self.L_net(z)
        return L

    def get_E(self, z):
        """
        The function `get_E` takes a parameter `z`, converts it to a tensor, passes it through a neural
        network called `energy_net`, and returns the resulting energy value.
        
        :param z: The parameter `z` is a numerical input that is used as an input to the `energy_net` neural network. It is converted to a tensor using `torch.tensor` and is set to have a data type of `torch.float32`. The `requires_grad=True` argument indicates that gradients will
        :return: the value of E, which is the output of the energy_net model when given the input z.
        """
        E = self.energy_net(z)
        return E

    def m_new(self, solver_iterations=200, tol=1e-6): #return new r and p
        """
        The function `m_new` calculates new values for `r` and `p` using the `fsolve` function and returns
        the updated values.
        :return: a tuple containing two tuples. The first tuple contains the values of `self.r[0]` and `self.r[1]`, and the second tuple contains the values of `self.p[0]` and `self.p[1]`.
        """
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


class ShivamoggiIMR(object):
    def __init__(self, M, dt, alpha, init_rx,  init_ry, init_rz, init_u, device="cpu"):
        """
        The function initializes the variables M, dt, alpha, init_rx, init_ry, init_rz, and init_u.
        
        :param M: The parameter M represents the mass of the system. It is used in the Hamiltonian equation to calculate the kinetic energy term, which is given by 1/2 p^2/M, where p is the momentum of the system
        :param dt: dt is the time step size used in numerical integration methods to update the system's state. It determines the granularity of the simulation and affects the accuracy and stability of the calculations. Smaller values of dt result in more accurate but slower simulations, while larger values of dt can lead to faster but less accurate
        :param alpha: The parameter alpha represents the strength of the harmonic potential in the Hamiltonian. It determines how tightly the particle is confined in the potential well. A larger value of alpha corresponds to a stronger confinement
        :param init_rx: The initial x-coordinate of the particle's position
        :param init_ry: The parameter `init_ry` represents the initial value of the y-coordinate of the position vector
        :param init_rz: The parameter `init_rz` represents the initial value of the z-coordinate of the position vector
        :param init_u: The `init_u` parameter represents the initial momentum of the system. It is a vector that specifies the initial momentum in each direction (x, y, z)
        """
        self.M = M #Hamiltonian = 1/2 p^2/M + 1/2 alpha r^2
        self.u = init_u
        self.x = torch.stack((init_rx, init_ry, init_rz), dim=1)
        self.alpha = alpha
        self.dt = dt
        self.device = device

    def get_E(self, m):
        """
        The function `get_E` calculates the value of E using the formula E = m[3]**2 + m[0]**2 - m[2]**2.
        
        :param m: The parameter `m` is a list or tuple containing four elements
        :return: the value of m[3]**2 + m[0]**2 - m[2]**2.
        """
        return m[:, 3]**2 + m[:, 0]**2 - m[:, 2]**2

    def get_UV(self, m):
        """
        The function `get_UV` takes a list `m` as input and returns two tuples (U and V) based on the values in `m`.
        
        :param m: The parameter `m` is a list containing four elements: `u`, `x`, `y`, and `z`
        :return: The function `get_UV` returns a tuple of two tuples. The first tuple contains three values: 0.0, 2*u*(x+z), and 0.0. The second tuple contains three values: x, 0, and -z.
        """
        u = m[:, 0]
        x = m[:, 1]
        y = m[:, 2]
        z = m[:, 3]
        zeros = torch.zeros_like(u, dtype=m.dtype, device=m.device)
        #return (0.0, 2*u*(x+z), 0.0), (x, 0, -z)

        U = torch.cat([zeros.unsqueeze(-1), (2*u*(x+z)).unsqueeze(-1), zeros.unsqueeze(-1)], dim=1)
        V = torch.cat([x.unsqueeze(-1), zeros.unsqueeze(-1), (-z).unsqueeze(-1)], dim=1)
        return U, V

    def get_L(self, m = (0.0, 0.0, 0.0, 0.0)):
        """
        The function `get_L` calculates and returns a 4x4 matrix L based on the input parameter m.
        
        :param m: The parameter `m` is a tuple of four values `(m[0], m[1], m[2], m[3])`
        :return: a 4x4 numpy array called L.
        """
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

    def f(self, mNew, mOld=None):#defines the function f zero of which is sought
        """
        The function `f` calculates the residual of a given input `mNew` by performing a series of
        mathematical operations.
        
        :param mNew: The parameter `mNew` is a list or array containing four values
        :return: a tuple containing the values of mres[0], mres[1], mres[2], and mres[3].
        """
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
        # Jacobian: J = -I + (dt/2) * d(mdot)/d(m_mid)
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
        um_old = torch.cat([self.u.unsqueeze(-1), self.x], dim=1)
        um_new = self.solve(um_old, max_iter=solver_iterations, tol=tol)
        
        self.u = um_new[:, 0].detach()
        self.x = um_new[:, 1:4].detach()
        
        return um_new.detach()


class ShivamoggiNeural(ShivamoggiIMR):
    def __init__(self, M, dt, alpha, init_rx,  init_ry, init_rz, init_u, device="cpu", method = "without", name = DEFAULT_folder_name):
        """
        The function initializes a ShivamoggiNeural object with specified parameters and loads the
        appropriate neural network models based on the chosen method.
        
        :param M: The parameter M represents the mass of the system. It is a scalar value
        :param dt: The parameter `dt` represents the time step size in the simulation. It determines how much time elapses between each iteration of the simulation
        :param alpha: The parameter "alpha" is a value used in the ShivamoggiNeural class initialization. It
        is a scalar value that represents a coefficient used in the calculations performed by the class. The specific purpose and meaning of this parameter may depend on the context and implementation of the Shivamoggi equations
        :param init_rx: The parameter `init_rx` is the initial x-coordinate of the position vector. It represents the starting position of the object in the x-direction
        :param init_ry: The parameter `init_ry` represents the initial value for the y-coordinate of the position vector. It is used in the initialization of the `ShivamoggiNeural` class
        :param init_rz: The parameter `init_rz` represents the initial value for the z-coordinate of the position vector. It is used in the initialization of the `ShivamoggiNeural` class
        :param init_u: The parameter `init_u` represents the initial value of the variable `u` in the ShivamoggiNeural class. It is used to initialize the state of the system
        :param method: The "method" parameter in the code snippet refers to the method used for solving the Shivamoggi equations. There are three possible options:, defaults to without (optional)
        :param name: The `name` parameter is a string that represents the folder name where the saved models are located. It is used to load the pre-trained neural network models for energy and L (Lagrangian) calculations. The `name` parameter is used to construct the file paths for loading the models
        """
        super(ShivamoggiNeural, self).__init__(M, dt, alpha, init_rx,  init_ry, init_rz, init_u, device=device)
        
        self.device = device
        # implicit not implemented - exception would have been raised earlier
        self.energy_net, self.L_net, self.J_net, self.A = load_models(name = name, method = method, device = device)

    # Get gradient of energy from NN
    def neural_zdot(self, z):
        """
        The function `neural_zdot` calculates the Hamiltonian of a neural network given a set of input
        values.
        
        :param z: The parameter `z` is a tensor representing the input to the neural network. It is of type `torch.Tensor` and has a shape determined by the dimensions of the input data
        :return: The function `neural_zdot` returns the variable `hamiltonian`.
        """
        z_tensor = torch.tensor(z, dtype=torch.float32, requires_grad=True, device=self.device)
        En = self.energy_net(z_tensor)

        E_z = torch.autograd.grad(En.sum(), z_tensor, only_inputs=True)[0]
        E_z = torch.flatten(E_z)

        if self.method == "soft" or self.method == "without":
            L = self.L_net(z_tensor).detach().cpu().numpy()[0]
            hamiltonian = np.matmul(L, E_z.detach().cpu().numpy())
        else:
            raise Exception("Implicit not implemented for HT yet.")
            J, cass = self.J_net(z_tensor)
            J = J.detach().numpy()
            hamiltonian = np.cross(J, E_z.detach().numpy())
        return hamiltonian

    def f(self, mNew, mOld=None):#defines the function f zero of which is sought
        """
        The function `f` calculates the difference between the old and new values of `m` and adds the
        product of the time step and the derivative of `z` to it.
        
        :param mNew: The parameter `mNew` represents the new values of `m` that are passed to the function `f`
        :return: the difference between the old values and the new values, plus the product of the time step and the derivative of the neural network with respect to the midpoint of the old and new values.
        """
        if mOld is None:
            mOld = np.array([self.u, self.x[0], self.x[1], self.x[2]])
        m_mid = 0.5*(np.array(mNew)+mOld)

        zd = self.neural_zdot(m_mid)
        res = np.array(mOld) - np.array(mNew) + self.dt*zd
        return res

    def get_cass(self, z):
        """
        The function `get_cass` takes in a parameter `z`, converts it to a tensor, passes it through a
        neural network `J_net`, and returns the output `cass` as a numpy array.
        
        :param z: The parameter `z` is a numerical input that is used as an input to the neural network. It is converted to a tensor using `torch.tensor` with a data type of `torch.float32`. The `requires_grad=True` argument indicates that gradients will be computed for this tensor during backprop
        :return: the value of `cass` as a NumPy array.
        """
        #z_tensor = torch.tensor(z, dtype=torch.float32, requires_grad=True, device=self.device)
        #J, cass = self.J_net(z_tensor)
        #return cass.detach().cpu().numpy()
        J, cass = self.J_net(z)
        return cass

    def get_L(self, z):
        """
        The function `get_L` takes a tensor `z`, passes it through a neural network `L_net`, and returns the
        output `L` as a numpy array.
        
        :param z: The parameter `z` is a numerical input that is used as an input to the `L_net` neural network. It is converted to a tensor using `torch.tensor` with a data type of `torch.float32`. The `requires_grad=True` argument indicates that gradients will be computed with respect
        :return: the value of L, which is obtained by passing the input z through the L_net neural network and converting the result to a numpy array.
        """
        #z_tensor = torch.tensor(z, dtype=torch.float32, requires_grad=True, device=self.device)
        #L = self.L_net(z_tensor).detach().cpu().numpy()[0]
        L = self.L_net(z)
        return L

    def get_E(self, z):
        """
        The function `get_E` takes a parameter `z`, converts it to a tensor, passes it through a neural
        network called `energy_net`, and returns the resulting energy value.
        
        :param z: The parameter `z` is a numerical input that is used as an input to the `energy_net` neural network. It is converted to a tensor using `torch.tensor` and is set to have a data type of `torch.float32`. The `requires_grad=True` argument indicates that gradients will
        :return: the value of E, which is the output of the energy_net model when given the input z.
        """
        #z_tensor = torch.tensor(z, dtype=torch.float32, requires_grad=True, device=self.device)
        #E = self.energy_net(z_tensor).detach().cpu().numpy()[0]
        E = self.energy_net(z)
        return E

    def m_new(self, solver_iterations=300, tol=5e-6): #return new r and p
        """
        The function `m_new` returns the values of `u`, `x[0]`, `x[1]`, and `x[2]` after solving the
        equation `f` using the `fsolve` function.

        :return: a tuple containing the values of self.u, self.x[0], self.x[1], and self.x[2].
        """
        #calculate
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
        r = m[:, :self.D]
        p = m[:, self.D:]
        return 0.5 * (p.pow(2).sum(dim=1)) / self.M + 0.5 * self.alpha * (r.pow(2).sum(dim=1))

    def get_L(self, m):
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
        if rpOld is None:
            rpOld = np.concatenate([self.r[0].cpu().numpy(), self.p[0].cpu().numpy()])

        r_old, p_old = rpOld[: self.D], rpOld[self.D :]
        r_new, p_new = rpNew[: self.D], rpNew[self.D :]

        rmdot_old = np.concatenate([p_old / self.M, -self.alpha * r_old])
        rmdot_new = np.concatenate([p_new / self.M, -self.alpha * r_new])

        rpres = rpOld - rpNew + self.dt / 2 * (rmdot_old + rmdot_new)

        return rpres

    def m_new(self, solver_iterations=200, tol=1e-6):
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
    def __init__(self, D, M, dt, alpha, init_r=None, init_p=None, B=1,
                 device="cpu", method="without", name="models"):
        
        super().__init__(D, M, dt, alpha, init_r=init_r, init_p=init_p, B=B, device=device)

        self.device = device
        # implicit not implemented - exception would jaev neem raised earlier
        self.energy_net, self.L_net, self.J_net, self.A = load_models(name = name, method = method, device = device)

    def neural_zdot(self, z):
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
        if rpOld is None:
            rpOld = torch.cat([self.r, self.p], dim=1).detach().cpu().numpy()

        zdo = self.neural_zdot(rpOld)
        zd = self.neural_zdot(rpNew)

        return np.array(rpOld) - np.array(rpNew) + self.dt / 2 * (zdo + zd)

    def get_E(self, z):
        if isinstance(z, np.ndarray):
            z = torch.tensor(z, dtype=torch.float32, device=self.device)
        return self.energy_net(z)

    def get_L(self, z):
        if isinstance(z, np.ndarray):
            z = torch.tensor(z, dtype=torch.float32, device=self.device)
        return self.L_net(z)

    def m_new(self, solver_iterations=200, tol=1e-6):
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

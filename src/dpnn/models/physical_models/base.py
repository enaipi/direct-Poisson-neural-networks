#This file contains the base RigidBody class and model loading utilities

from scipy.optimize import fsolve
from math import *
import numpy as np

import torch

from dpnn.models.energy_nn import EnergyNet
from dpnn.models.tensor_nn import TensorNet, JacVectorNet
from dpnn.training import DEFAULT_folder_name


def load_models(name = DEFAULT_folder_name, method = "without", mx = torch.zeros((1,1)), device="cpu"):
    """Load neural network models for energy and L/J structures.
    
    :param name: Folder name where saved models are located
    :param method: Method type - "soft", "without", or "implicit"
    :param mx: Template tensor for device/dtype inference
    :param device: Device to load models onto
    :return: Tuple of (energy_net, L_net, J_net, A)
    """
    A, J_net = None, None
    if method == "soft":
        energy_net = torch.load(name+'/saved_models/soft_jacobi_energy', weights_only=False)
        energy_net.eval()   
        L_net = torch.load(name+'/saved_models/soft_jacobi_L', weights_only=False)
        L_net.eval()
    elif method == "without":
        energy_net = torch.load(name+'/saved_models/without_jacobi_energy', weights_only=False)
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
        energy_net = torch.load(name+'/saved_models/implicit_jacobi_energy', weights_only=False)
        energy_net.eval()
        J_net = torch.load(name+'/saved_models/implicit_jacobi_J', weights_only=False)
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
    """Base class for rigid body motion solvers."""
    
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
        """The function calculates the energy of an object in the x-direction."""
        return 0.5*self.mx*self.mx/self.Ix

    def energy_y(self):
        """The function calculates the energy of an object in the y-direction."""
        return 0.5*self.my*self.my/self.Iy

    def energy_z(self):
        """The function calculates the energy of an object rotating around the z-axis."""
        return 0.5*self.mz*self.mz/self.Iz

    def energy(self):#returns kinetic energy
        """The function calculates the kinetic energy of an object based on its mass and moments of inertia."""
        return 0.5*(self.mx*self.mx/self.Ix+self.my*self.my/self.Iy+self.mz*self.mz/self.Iz)

    def omega_x(self):
        """The function calculates the angular velocity around the x-axis."""
        return self.mx/self.Ix

    def omega_y(self):
        """The function calculates the omega_y value by dividing my by Iy."""
        return self.my/self.Iy

    def omega_z(self):
        """The function calculates the angular velocity around the z-axis."""
        return self.mz/self.Iz

    def m2(self):#returns m^2
        """The function calculates the square of the magnitude of a vector."""
        return self.mx*self.mx+self.my*self.my+self.mz*self.mz

    def mx2(self):#returns mx^2
        """The function mx2 returns the value of mx squared."""
        return self.mx*self.mx

    def my2(self):#returns my^2
        """The function `my2` returns the square of the value of `self.my`."""
        return self.my*self.my

    def mz2(self):#returns mz^2
        """The function mz2 returns the square of the value of mz."""
        return self.mz*self.mz

    def m_magnitude(self):#returns |m|
        """The function returns the magnitude of a vector."""
        return sqrt(self.m2())

    def Ein(self):#returns normalized internal energy
        """The function returns the normalized internal energy."""
        return self.Einconst*pow(self.sin,4.0/3)/self.Ein_init

    def Ein_s(self): #returns normalized derivative of internal energy with respect to entropy (inverse temperature)
        """The function returns the normalized derivative of internal energy with respect to entropy."""
        return self.Einconst*4.0/3*pow(self.sin, 1.0/3) / self.Ein_init

    def ST(self, T): #returns entropy of a Copper body with characteristic volume equal to one (Debye), [T] = K
        """The function calculates the entropy of a Copper body with a characteristic volume equal to one (Debye) at a given temperature."""
        return 2 * pi**2/15 * self.kB * (self.kB/self.hbar *T/self.umean)**3

    def Etot(self):#returns normalized total energy
        """The function `Etot` returns the sum of the energy and the input energy."""
        return self.energy() + self.Ein()

    def Sin(self): #returns normalized internal entorpy
        """The intenral entropy function returns the normalized internal entropy."""
        return self.sin/self.sin_init

    def S_x(self):#kinetic entropy for rotation around x, beta = 1/4Iz
        """The function calculates the kinetic entropy for rotation around the x-axis."""
        m2 = self.m2()
        return -m2/self.Ix - 0.5*0.25/self.Iz*(m2-self.mx0*self.mx0)**2

    def S_z(self):#kinetic entropy for rotation around z
        """The function calculates the kinetic entropy for rotation around the z-axis."""
        m2 = self.m2()
        return -m2/self.Iz - 0.5*0.25/self.Iz*(m2-self.mz0*self.mz0)**2

    def Phi_x(self): #Returns the Phi potential for rotation around the x-axis
        """The function Phi_x returns the sum of the energy and the S_x potential for rotation around the x-axis."""
        return self.energy() + self.S_x()

    def Phi_z(self):
        """The function Phi_z returns the sum of the energy and the S_z value."""
        return self.energy() + self.S_z()
        
    def get_L(self, m):
        """The function `get_L` returns a 3x3 matrix `L` (Poisson bivector) based on the input parameter `m`."""
        zeros = torch.zeros_like(self.mx)
        L = torch.stack([
            torch.stack([zeros, self.mz, -self.my], dim=1),
            torch.stack([-self.mz, zeros, self.mx], dim=1),
            torch.stack([self.my, -self.mx, zeros], dim=1)
        ], dim=1)
        return - L
        
    def get_E(self, m):
        """The function "get_E" returns the energy of an object."""
        return self.energy()

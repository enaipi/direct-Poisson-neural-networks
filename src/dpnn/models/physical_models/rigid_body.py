"""Rigid body dynamics solver with Poisson structure integration."""

from scipy.optimize import fsolve
from math import *
import numpy as np
import torch

from .base import GeneralSystem


class RigidBody(GeneralSystem):
    """
    Base class for rigid body motion solvers.
    
    Inherits from GeneralSystem to leverage generic integration methods.
    Maintains backward compatibility by providing mx, my, mz as properties.
    """
    
    def __init__(self, Ix, Iy, Iz, d2E, mx, my, mz, dt, alpha, T=100, verbose = False, device = "cpu", dtype=torch.float32):
        # Store RigidBody-specific parameters
        self.Ix = Ix
        self.Iy = Iy
        self.Iz = Iz
        self.d2E = d2E
        self._dtype = dtype  # Store dtype before parent init
        
        # Compute auxiliary parameters
        if Iz > 0 and Iy > 0 and Iz > 0:
            self.Jx = 1/Iz - 1/Iy
            self.Jy = 1/Ix - 1/Iz
            self.Jz = 1/Iy - 1/Ix

        # Store physics constants
        self.hbar = 1.0545718E-34  # Reduced Planck constant [SI]
        self.rho = 8.92E+03        # For copper
        self.myhbar = self.hbar * self.rho  # Due to rescaled mass
        self.kB = 1.38064852E-23   # Boltzmann constant
        self.umean = 4600          # Mean sound speed in copper [SI]
        self.Einconst = pi**2/10 * pow(15/(2* pi**2), 4.0/3) * self.hbar * self.umean * pow(self.kB, -4.0/3)
        
        if verbose:
            print("Internal energy prefactor = ", self.Einconst)

        # Compute initial entropy
        self.sin = self.ST(T)
        if verbose:
            print("Internal entropy set to Sin = ", self.sin, " at T=",T," K")

        # Initialize internal energy tracking
        self.Ein_init = 1
        
        # Convert initial conditions to tensors with proper device/dtype
        z_init = torch.stack([
            torch.as_tensor(mx, device=device, dtype=dtype),
            torch.as_tensor(my, device=device, dtype=dtype),
            torch.as_tensor(mz, device=device, dtype=dtype)
        ], dim=1)
        
        # Store initial values (for entropy calculations)
        self._mx0 = z_init[:, 0].clone()
        self._my0 = z_init[:, 1].clone()
        self._mz0 = z_init[:, 2].clone()
        
        # Define energy function for GeneralSystem
        def rb_energy(z):
            """Kinetic energy: 0.5 * (mx^2/Ix + my^2/Iy + mz^2/Iz)"""
            return 0.5 * (z[:, 0]**2 / Ix + z[:, 1]**2 / Iy + z[:, 2]**2 / Iz)
        
        # Define Poisson bivector function
        def rb_poisson(z):
            """Skew-symmetric Poisson matrix for rigid body (cross-product structure)."""
            batch_size = z.shape[0]
            device_z, dtype_z = z.device, z.dtype
            
            zeros = torch.zeros(batch_size, dtype=dtype_z, device=device_z)
            
            L = torch.stack([
                torch.stack([zeros, -z[:, 2], z[:, 1]], dim=1),
                torch.stack([z[:, 2], zeros, -z[:, 0]], dim=1),
                torch.stack([-z[:, 1], z[:, 0], zeros], dim=1)
            ], dim=1)
            
            return -L  # Negative sign for Poisson bracket convention
        
        # Define energy gradient function
        def rb_grad_energy(z):
            """Gradient: dE/dm = m / I"""
            I = torch.tensor([Ix, Iy, Iz], device=z.device, dtype=z.dtype)
            return z / I
        
        # Initialize parent GeneralSystem
        super().__init__(
            z_init=z_init,
            energy_fn=rb_energy,
            poisson_fn=rb_poisson,
            grad_energy_fn=rb_grad_energy,
            dt=dt,
            device=device,
            dtype=dtype,
            verbose=verbose
        )
        
        # Set tau (time scale parameter from alpha)
        self._tau = dt * alpha
        
        # Complete internal energy initialization now that parent is initialized
        self.Ein_init = self.Ein()
        self.sin_init = self.sin
        
        if verbose:
            print("Initial total energy = ", self.Etot())
            print("RB set up.")
    
    # ========================================================================
    # Backward compatibility: Access state as individual attributes
    # ========================================================================
    
    @property
    def mx(self):
        """Angular momentum x-component (property for backward compatibility)."""
        return self.z[:, 0]
    
    @mx.setter
    def mx(self, value):
        """Set angular momentum x-component."""
        self.z[:, 0] = torch.as_tensor(value, device=self.device, dtype=self._dtype)
    
    @property
    def my(self):
        """Angular momentum y-component (property for backward compatibility)."""
        return self.z[:, 1]
    
    @my.setter
    def my(self, value):
        """Set angular momentum y-component."""
        self.z[:, 1] = torch.as_tensor(value, device=self.device, dtype=self._dtype)
    
    @property
    def mz(self):
        """Angular momentum z-component (property for backward compatibility)."""
        return self.z[:, 2]
    
    @mz.setter
    def mz(self, value):
        """Set angular momentum z-component."""
        self.z[:, 2] = torch.as_tensor(value, device=self.device, dtype=self._dtype)
    
    @property
    def mx0(self):
        """Initial x-component (for entropy calculations)."""
        return self._mx0
    
    @mx0.setter
    def mx0(self, value):
        """Set initial x-component."""
        self._mx0 = value
    
    @property
    def my0(self):
        """Initial y-component (for entropy calculations)."""
        return self._my0
    
    @my0.setter
    def my0(self, value):
        """Set initial y-component."""
        self._my0 = value
    
    @property
    def mz0(self):
        """Initial z-component (for entropy calculations)."""
        return self._mz0
    
    @mz0.setter
    def mz0(self, value):
        """Set initial z-component."""
        self._mz0 = value
    
    @property
    def dtype(self):
        """Data type (for backward compatibility)."""
        return self._dtype
    
    @dtype.setter
    def dtype(self, value):
        """Set data type."""
        self._dtype = value
    
    @property
    def tau(self):
        """Time scale parameter (if set during init)."""
        if hasattr(self, '_tau'):
            return self._tau
        return None
    
    @tau.setter
    def tau(self, value):
        """Set time scale parameter."""
        self._tau = value


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

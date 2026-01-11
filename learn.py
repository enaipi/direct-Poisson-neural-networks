import argparse
import os
import datetime
import re
import time

import torch
from torch.utils.data import DataLoader
import torchmetrics
from torch import einsum

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from models.Model import EnergyNet, TensorNet, JacVectorNet 
from TrajectoryDataset import TrajectoryDataset

#declare default values of parameters
DEFAULT_dataset = "data/dataset.xyz"
DEFAULT_batch_size = 32
DEFAULT_dt = 0.1 
DEFAULT_learning_rate = 1.0e-05
DEFAULT_epochs = 10 
DEFAULT_prefactor = 1.0
DEFAULT_jacobi_prefactor = 1.0
DEFAULT_neurons = 64
DEFAULT_layers = 2
DEFAULT_folder_name = "."

class Learner(object):
    """
    This is the fundamental class that provides the capability to learn dynamical systems, 
    using various methods of learning (without Jacobi identity, with softly enforced Jacobi, and with implicitly valid Jacobi).
    """
    def __init__(self, model, batch_size = DEFAULT_batch_size, simulation_batch_size = DEFAULT_batch_size, dt = DEFAULT_dt, neurons = DEFAULT_neurons,
                 layers = DEFAULT_layers, name = DEFAULT_folder_name, device = "cpu", dissipative = False, dropout_rate=0.0,
                 quad_features=False, no_data_to_gpu=True, D=10, use_constant_L=False):
        """
        This function initializes a Learner object for a given model, with specified parameters and
        datasets.
        
        :param model: The "model" parameter specifies the type of model being used. It can take the following values:
        :param batch_size: The batch size is the number of samples that will be propagated through the neural network at once. It is used to control the number of samples processed in each iteration during training
        :param dt: The parameter "dt" represents the time step size used in the model. It determines the interval at which the model updates its internal state and makes predictions
        :param neurons: The "neurons" parameter represents the number of neurons in each hidden layer of the neural network models used in the code
        :param layers: The "layers" parameter specifies the number of layers in the neural network models used in the code. It determines the depth of the network and affects the complexity and capacity of the model
        :param name: The name parameter is used to specify the folder name where the dataset is located
        :param cuda: A boolean value indicating whether to use CUDA for GPU acceleration. If set to True, the model will be moved to the GPU device, defaults to False (optional)
        :param dissipative: A boolean parameter indicating whether the model is dissipative or not. If dissipative is set to True, it means that the model includes dissipative dynamics, defaults to False (NOT YET IMPLEMENTED)
        """
        self.model = model
        dim = 0
        if model == "RB":
            dim = 3 #input features
        elif model in ["HT", "P3D", "K3D"]:
            dim = 6
        elif model == "P2D" or model == "Sh":
            dim = 4
        elif model == "D":
            dim = 2*D
        else:
            raise Exception("Unknown model "+model)
        print("Generating Learner for model ", model, " with ", dim, " dimensions.")
        self.name = name

        self.dissipative = dissipative

        self.no_data_to_gpu = no_data_to_gpu

        #self.logdir = os.path.join("logs", "{}-{}-{}".format(
        #    os.path.basename(globals().get("__file__", "notebook")),
        #    datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
        #    ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", key), value) for key, value in sorted(vars(args).items())))
        #))

        #start_time = time.time()
        self.df = pd.read_csv(name+"/"+DEFAULT_dataset, dtype=np.float32)
        #end_time = time.time()
        #print(end_time-start_time)

        self.energy = EnergyNet(dim, neurons, layers, batch_size, dropout_rate=dropout_rate, quad_features=quad_features)  # added dropout and quad features parameters

        self.jac_vec = JacVectorNet(dim, neurons, layers, batch_size, dropout_rate=dropout_rate)  # added dropout parameter
        self.entropy = EnergyNet(dim, neurons, layers, batch_size, dropout_rate=dropout_rate, quad_features=quad_features)  # added dropout and quad features parameters

        self.device = device
        self.energy = self.energy.to(self.device)
        
        self.jac_vec = self.jac_vec.to(self.device)
        self.entropy = self.entropy.to(self.device)
        # self.dispot = self.dispot.to(self.device)

        self.use_constant_L = use_constant_L
        if self.use_constant_L:
            self.A = torch.nn.Parameter(torch.randn(dim, dim, device=self.device))
        else:
            self.L_tensor = TensorNet(dim, neurons, layers, max(batch_size, simulation_batch_size), dropout_rate=dropout_rate).to(self.device)
        
        self.train, self.test = train_test_split(self.df, test_size=0.4)
        self.train_dataset = TrajectoryDataset(self.train, model = model, device=self.device, no_data_to_gpu=no_data_to_gpu, dim=D)
        self.valid_dataset = TrajectoryDataset(self.test, model = model, device=self.device, no_data_to_gpu=no_data_to_gpu, dim=D)

        self.train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True)
        self.valid_loader = DataLoader(self.valid_dataset, batch_size=batch_size, shuffle=True)

        self.dt = dt
        #self.lam = 0.5

        #constant in M

        #for soft and without
        self.train_metric = torchmetrics.MeanSquaredError().to(self.device)
        self.val_metric = torchmetrics.MeanSquaredError().to(self.device)

        #for Jacobi
        self.train_metric_reg = torchmetrics.MeanSquaredError().to(self.device)
        self.val_metric_reg = torchmetrics.MeanSquaredError().to(self.device)

        self.loss_fn = torch.nn.MSELoss()

        self.train_errors = []
        self.validation_errors = []

        self.noise_sigma = -1 # add random noise to data

    def L_tensor_func(self, zn_tensor):
        L = self.A - self.A.t()
        return L.unsqueeze(0).repeat(zn_tensor.size(0), 1, 1)

    def forward_L_tensor(self, zn_tensor):
        if self.use_constant_L:
            return self.L_tensor_func(zn_tensor)
        else:
            return self.L_tensor(zn_tensor)

    def mov_loss_without(self, zn_tensor, zn2_tensor, mid_tensor):
        """
        The function calculates the movement loss using the "without" method.
        
        :param zn_tensor: The `zn_tensor` parameter represents the current state of the system. It is a tensor that contains the values of the variables in the system at a particular time
        :param zn2_tensor: The parameter `zn2_tensor` is a tensor representing the current state of the system at time `t+dt`. It is used to calculate the loss function for the movement of the system without considering any external forces or constraints
        :param mid_tensor: The `mid_tensor` parameter is not used in the `mov_loss_without` function. It is not necessary for the calculation and can be removed from the function signature
        :return: the result of the expression `(zn_tensor - zn2_tensor)/self.dt + 1.0/2.0*(torch.matmul(Lz, E_z.unsqueeze(2)).squeeze() + torch.matmul(Lz2, E_z2.unsqueeze(2)).squeeze())`.
        """
        En = self.energy(zn_tensor)
        En2 = self.energy(zn2_tensor)

        E_z = torch.autograd.grad(En.sum(), zn_tensor, only_inputs=True, create_graph=True)[0]
        E_z2 = torch.autograd.grad(En2.sum(), zn2_tensor, only_inputs=True, create_graph=True)[0]

        Lz = self.forward_L_tensor(zn_tensor)
        Lz2 = self.forward_L_tensor(zn2_tensor)

        term1 = torch.bmm(Lz, E_z.unsqueeze(2)).squeeze(2)
        term2 = torch.bmm(Lz2, E_z2.unsqueeze(2)).squeeze(2)

        return (zn_tensor - zn2_tensor)/self.dt + 0.5*(term1 + term2)

    def mov_loss_without_with_jacobi(self, zn_tensor, zn2_tensor, mid_tensor, reduced_L=None):
        """
        The function calculates the movement loss including Jacobi identity the for a given input tensor.
        
        :param zn_tensor: The zn_tensor is a tensor representing the current state of the system. It is used to calculate the energy and the Jacobian loss of the system
        :param zn2_tensor: The `zn2_tensor` parameter is a tensor representing the state at time `t+1`. It is used to calculate the loss for the movement of the system without using the Jacobian matrix
        :param mid_tensor: The `mid_tensor` parameter is a tensor representing the intermediate state between `zn_tensor` and `zn2_tensor`. It is used to calculate the loss function
        :param reduced_L: The parameter "reduced_L" is a reduced Laplacian matrix. It is used in the calculation of the Jacobi loss
        :return: two values: \\ 1. The difference between `zn_tensor` and `zn2_tensor` divided by `self.dt` plus half of the sum of the matrix multiplication of `Lz` and `E_z` and the matrix multiplication of `Lz2` and `E_z2`. \\ 2. The `jacobi_loss` calculated using `zn_tensor`, `
        """
        En = self.energy(zn_tensor)
        En2 = self.energy(zn2_tensor)

        E_z = torch.autograd.grad(En.sum(), zn_tensor, only_inputs=True, create_graph=True)[0]
        E_z2 = torch.autograd.grad(En2.sum(), zn2_tensor, only_inputs=True, create_graph=True)[0]

        Lz = self.forward_L_tensor(zn_tensor)
        Lz2 = self.forward_L_tensor(zn2_tensor)

        with torch.no_grad():
            jacobi_loss = self.jacobi_loss(zn_tensor, Lz, reduced_L)

        return (zn_tensor - zn2_tensor)/self.dt + 1.0/2.0*(torch.matmul(Lz, E_z.unsqueeze(2)).squeeze() \
                                                        + torch.matmul(Lz2, E_z2.unsqueeze(2)).squeeze()), jacobi_loss

    def jacobi_loss(self, zn_tensor, Lz, reduced_L):
        """
        The function `jacobi_loss` calculates the Jacobi loss (error in Jacobi identity) using the given inputs.
        
        :param zn_tensor: The `zn_tensor` parameter is a tensor representing the input to the function. It is used to compute the Jacobian loss.
        :param Lz: Lz is a tensor representing the Jacobian matrix of the output with respect to the input. It has shape (m, n, n), where m is the number of samples and n is the number of input variables.
        :param reduced_L: The parameter `reduced_L` is a tensor representing the reduced loss function
        :return: the sum of three terms: term1, term2, and term3.
        """
        #Lz_grad = torch.autograd.functional.jacobian(reduced_L, zn_tensor, create_graph=True).permute(2, 0, 1, 3)
        J = self.L_tensor.get_jacobian(zn_tensor)
        
        """B, dim, _ = Lz.shape
        
        tri_i0, tri_i1 = self.L_tensor.tri_i

        # p = dim*(2*dim - 1)
        Lz_flat = Lz[:, tri_i0, tri_i1]  # (B, p)

        term1_flat = Lz_flat.unsqueeze(2) * J  # (B, p, dim)

        term1 = torch.empty(B, dim, dim, dim, device=term1_flat.device)

        term1[:, tri_i0, tri_i1, :] = term1_flat
        # self.L_tensor.sym_sing = -1
        term1[:, tri_i1, tri_i0, :] = self.L_tensor.sym_sing * term1_flat  # (B, dim, dim, dim)
        # torch.mul(self.L_tensor.sym_sing, term1_flat, out=term1[:, tri_i1, tri_i0, :])

        out = term1.clone()
        out.add_(term1.permute(0,2,3,1))
        out.add_(term1.permute(0,3,1,2))"""

        term1 = einsum('mkl,mijk->mijl', Lz, J)
    
        term2 = term1.permute(0, 2, 3, 1)
        term3 = term1.permute(0, 3, 1, 2)
        
        jacobi_identity_error = term1 + term2 + term3
        del term1, term2, term3, J
        return jacobi_identity_error.pow(2).mean()
    
    def jacobi_loss_og(self, zn_tensor, Lz, reduced_L):
        """
        The function `jacobi_loss` calculates the Jacobi loss (error in Jacobi identity) using the given inputs.
        
        :param zn_tensor: The `zn_tensor` parameter is a tensor representing the input to the function. It is used to compute the Jacobian loss.
        :param Lz: Lz is a tensor representing the Jacobian matrix of the output with respect to the input. It has shape (m, n, n), where m is the number of samples and n is the number of input variables.
        :param reduced_L: The parameter `reduced_L` is a tensor representing the reduced loss function
        :return: the sum of three terms: term1, term2, and term3.
        """
        Lz_grad = torch.autograd.functional.jacobian(reduced_L, zn_tensor, create_graph=True).permute(2, 0, 1, 3)
        #print(torch.max(Lz_grad))
        term1 = einsum('mkl,mijk->mijl', Lz, Lz_grad)
        term2 = term1.permute(0,2,3,1)
        term3 = term1.permute(0,3,1,2)
        return (term1 + term2 + term3).pow(2).mean()

    def mov_loss_soft(self, zn_tensor, zn2_tensor, mid_tensor, reduced_L):
        """
        The function `mov_loss_soft` calculates the movement loss and Jacobi loss for a given input tensor, using the "soft" method.
        
        :param zn_tensor: The zn_tensor is a tensor representing the current state of the system. It is used to calculate the energy and gradient of the energy with respect to zn_tensor
        :param zn2_tensor: The `zn2_tensor` parameter is a tensor representing the state at time `t+1`. It is used to calculate the movement loss and Jacobi loss in the `mov_loss_soft` function
        :param mid_tensor: The `mid_tensor` parameter is not used in the `mov_loss_soft` function. It is not clear what its purpose is without further context.
        :param reduced_L: The parameter "reduced_L" is a tensor representing the reduced Laplacian matrix
        :return: two values: `mov_loss` and `jacobi_loss`.
        """
        En = self.energy(zn_tensor)
        En2 = self.energy(zn2_tensor)

        E_z = torch.autograd.grad(En.sum(), zn_tensor, only_inputs=True, create_graph=True)[0]
        E_z2 = torch.autograd.grad(En2.sum(), zn2_tensor, only_inputs=True, create_graph=True)[0]

        Lz = self.forward_L_tensor(zn_tensor)
        Lz2 = self.forward_L_tensor(zn2_tensor)
        mov_loss = (zn_tensor - zn2_tensor)/self.dt + 1.0/2.0*(torch.matmul(Lz, E_z.unsqueeze(2)).squeeze() \
                                                        + torch.matmul(Lz2, E_z2.unsqueeze(2)).squeeze())
        #Jacobi
        jacobi_loss = self.jacobi_loss(zn_tensor, Lz, reduced_L)
        return mov_loss, jacobi_loss

        
    def mov_loss_implicit(self, zn_tensor, zn2_tensor, mid_tensor):
        """
        The function `mov_loss_implicit` calculates the loss for a motion model using the "implicit" method.
        
        :param zn_tensor: The `zn_tensor` parameter is a tensor representing the current state of the system. It is used to calculate the energy and Jacobian vectors for the system.
        :param zn2_tensor: The `zn2_tensor` parameter is a tensor representing the state of the system at time `t + dt`, where `t` is the current time and `dt` is the time step.
        :param mid_tensor: The `mid_tensor` parameter is not used in the `mov_loss_implicit` function. It is not necessary for the calculation and can be removed from the function signature.
        :return: The function `mov_loss_implicit` returns the result of the expression `(zn_tensor - zn2_tensor)/self.dt + 1.0/2.0*(torch.cross(Jz, E_z, dim=1) + torch.cross(Jz2, E_z2, dim=1))`.
        """
        En = self.energy(zn_tensor)
        En2 = self.energy(zn2_tensor)

        E_z = torch.autograd.grad(En.sum(), zn_tensor, only_inputs=True, create_graph=True)[0]
        E_z2 = torch.autograd.grad(En2.sum(), zn2_tensor, only_inputs=True, create_graph=True)[0]
 
        Jz, cass = self.jac_vec(zn_tensor)
        Jz2, cass2 = self.jac_vec(zn2_tensor)
        return (zn_tensor - zn2_tensor)/self.dt + 1.0/2.0*(torch.cross(Jz, E_z, dim=1) + torch.cross(Jz2, E_z2, dim=1))

    def learn(self, method = "without", learning_rate = DEFAULT_learning_rate, epochs = DEFAULT_epochs, prefactor = DEFAULT_prefactor, jac_prefactor = DEFAULT_jacobi_prefactor, scheme="IMR"):
        """
        The `learn` function is used to train a model using different methods and parameters, and it saves
        the trained models and error metrics.
        
        :param method: The method parameter determines the learning method to be used. It can take one of three values: "without", "soft", or "implicit", defaults to without (optional)
        :param learning_rate: The learning rate determines the step size at which the optimizer adjusts the model's parameters during training. It controls how quickly or slowly the model learns from the training data
        :param epochs: The "epochs" parameter determines the number of times the model will iterate over the entire training dataset during the learning process. Each iteration is called an epoch
        :param prefactor: The `prefactor` parameter is a scaling factor that is applied to the movement loss during training. It allows you to control the relative importance of the movement loss compared to other losses or metrics. By adjusting the value of `prefactor`, you can increase or decrease the impact of the movement loss on
        :param jac_prefactor: The `jac_prefactor` parameter is used as a scaling factor for the regularization term in the loss function. It determines the relative importance of the regularization term compared to the movement term in the loss function. A higher value of `jac_prefactor` will give more weight to the regularization term, while
        :param scheme: The "scheme" parameter is a string that specifies the numerical scheme used for solving the equations. It can take one of the following values:, defaults to IMR (optional)
        """
        if method not in ["without", "soft", "implicit"]:
            raise Exception("Unknown method "+method)
        print("Learning from folder "+self.name) 
        print("Method = "+method)
        print("Epochs = "+str(epochs))
        print("Dissipative NOT IMPLEMENTED" if self.dissipative else "Non-dissipative")
        optimizer, scheduler = None, None

        if method in ["without", "soft"]:
            if self.use_constant_L:
                optimizer = torch.optim.Adam(list(self.energy.parameters()) + [self.A], lr=learning_rate)
            else:
                optimizer = torch.optim.Adam(list(self.energy.parameters()) + list(self.L_tensor.parameters()), lr = learning_rate)
        elif method == "implicit":
            optimizer = torch.optim.Adam(list(self.energy.parameters())
                             + list(self.jac_vec.parameters()), lr = learning_rate)
        #scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.98, last_epoch= -1)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

        mov_loss, jacobi_loss, Lz, Lz2, Jz, Jz2, cass, cass2, reduced_L = None, None, None, None, None, None, None, None, None
        for epoch in range(epochs):
            print("\nStart of epoch %d" % (epoch,))
            start_time_train = time.time()
            #start_time = time.time()
            reduced_L = lambda z: torch.sum(self.forward_L_tensor(z), axis=0)

            # Iterate over the batches of the dataset.
            for step, (zn_tensor, zn2_tensor, mid_tensor) in enumerate(self.train_loader):
                #print("zn = ", zn_tensor)
                #print("zn2 = ", zn2_tensor)
                # zero the parameter gradients

                if self.no_data_to_gpu == False:
                    zn_tensor = zn_tensor.to(self.device)
                    zn2_tensor = zn2_tensor.to(self.device)
                    mid_tensor = mid_tensor.to(self.device)

                if self.noise_sigma > 0.0:
                    zn_tensor = zn_tensor + self.noise_sigma * torch.randn_like(zn_tensor)
                    zn2_tensor = zn2_tensor + self.noise_sigma * torch.randn_like(zn2_tensor)
                    mid_tensor = mid_tensor + self.noise_sigma * torch.randn_like(mid_tensor)
                    
                optimizer.zero_grad()
                zn_tensor.requires_grad = True
                zn2_tensor.requires_grad = True
                mid_tensor.requires_grad = True

                if method == "without":
                    mov_loss = self.mov_loss_without(zn_tensor, zn2_tensor, mid_tensor)
                elif method == "implicit":
                    mov_loss = self.mov_loss_implicit(zn_tensor, zn2_tensor, mid_tensor)
                elif method == "soft":
                    mov_loss, jacobi_loss = self.mov_loss_soft(zn_tensor, zn2_tensor, mid_tensor, reduced_L)
                
                mov_value = self.loss_fn(torch.zeros_like(mov_loss), prefactor*mov_loss)
                loss = mov_value
                self.train_metric(torch.zeros_like(mov_value), mov_value)
                if method == "soft":
                    reg_value = self.loss_fn(torch.zeros_like(jacobi_loss), jac_prefactor*jacobi_loss)
                    loss += reg_value
                    self.train_metric_reg(torch.zeros_like(reg_value), reg_value)

                loss.backward()
                optimizer.step()

                # Run one step of gradient descent by updating
                # the value of the variables to minimize the loss.

                # Log every 200 batches.
                """if step % 200 == 0:
                    if method == "soft":
                        print(
                        "Training loss (for one batch) at step %d: movement %.4f and reg %.4f"
                        % (step, float(mov_value), float(reg_value))
                        )
                    else:
                        print(
                            "Training loss (for one batch) at step %d: movement %.4f "
                            % (step, float(mov_value))
                        )"""
                    #print("Seen so far: %s samples" % ((step + 1) * 64))


            # Display metrics at the end of each epoch.
            train_acc = self.train_metric.compute()
            self.train_metric.reset()
            if method == "soft":
                train_acc_reg = self.train_metric_reg.compute()
                self.train_metric_reg.reset()
                print("Training err over epoch: %.4f reg %.4f" % (float(train_acc), float(train_acc_reg)))
                self.train_errors.append([float(train_acc), float(train_acc_reg)])
            else:
                print("Training err over epoch: %.4f " % (float(train_acc)))
                self.train_errors.append([float(train_acc), 0.0])
            
            end_time_train = time.time()
            print("Time taken for training: %.2fs" % (end_time_train - start_time_train))

            start_time_val = time.time()
            # Run a validation loop at the end of each epoch.
            jacobi_loss = None
            for step, (zn_tensor, zn2_tensor, mid_tensor) in enumerate(self.valid_loader):

                if self.no_data_to_gpu == False:
                    zn_tensor = zn_tensor.to(self.device)
                    zn2_tensor = zn2_tensor.to(self.device)
                    mid_tensor = mid_tensor.to(self.device)

                zn_tensor.requires_grad = True
                zn2_tensor.requires_grad = True
                mid_tensor.requires_grad = True

                if method == "without":
                    if self.use_constant_L:
                        mov_loss = self.mov_loss_without(zn_tensor, zn2_tensor, mid_tensor)
                    else:
                        mov_loss, jacobi_loss = self.mov_loss_without_with_jacobi(zn_tensor, zn2_tensor, mid_tensor, reduced_L)
                elif method == "implicit":
                    mov_loss = self.mov_loss_implicit(zn_tensor, zn2_tensor, mid_tensor) #Jacobi identity automatically satisfied
                elif method == "soft":
                    mov_loss, jacobi_loss = self.mov_loss_soft(zn_tensor, zn2_tensor, mid_tensor, reduced_L)

                #mov_value = self.loss_fn(torch.zeros_like(mov_loss), prefactor*mov_loss)
                mov_value = mov_loss.pow(2).mean() * prefactor**2
                self.val_metric(torch.zeros_like(mov_value), mov_value)
                if jacobi_loss != None:
                    reg_value = self.loss_fn(torch.zeros_like(jacobi_loss), jac_prefactor*jacobi_loss)
                    #reg_value = jacobi_loss.pow(2).mean() * jac_prefactor**2
                    self.val_metric_reg(torch.zeros_like(reg_value), reg_value)

            val_acc_val = self.val_metric.compute()
            self.val_metric.reset()
            if jacobi_loss != None:
                val_acc_reg = self.val_metric_reg.compute()
                self.val_metric_reg.reset()
                self.validation_errors.append([float(val_acc_val), float(val_acc_reg)])
                print("Validation error: value %.4f" % float(val_acc_val), float(val_acc_reg))
            else: #implicit
                self.validation_errors.append([float(val_acc_val), 0.0])
                print("Validation error: value %.4f" % (float(val_acc_val)))

            end_time_val = time.time()
            print("Time taken for validation: %.2fs" % (end_time_val - start_time_val))
            #print("Time taken: %.2fs" % (time.time() - start_time))
            scheduler.step()
            #print(optimizer.param_groups[0]['lr'])

        # The above code is saving various variables and dataframes to files based on the value of the
        # "method" parameter. If the "method" is "without", it saves the "energy", "L_tensor", "entropy", and
        # "dispot" variables to separate files, and saves the "errors_df" dataframe to a CSV file. If the
        # "method" is "implicit", it saves the "energy", "jac_vec", "entropy", and "dispot" variables to
        # separate files, and saves the "errors_df" dataframe to a CSV file. If the "method" is "
        errors = np.hstack((self.train_errors, self.validation_errors))
        errors_df = pd.DataFrame(errors, columns = ["train_mov", "train_reg", "val_mov", "val_reg"]) 
        if method == "without":
            torch.save(self.energy, self.name+'/saved_models/without_jacobi_energy')
            if self.use_constant_L:
                torch.save({'L_type': 'constant', 'A': self.A}, self.name+'/saved_models/without_jacobi_L')
            else:
                torch.save({'L_type': 'module', 'L_tensor': self.L_tensor}, self.name+'/saved_models/without_jacobi_L')
            errors_df.to_csv(self.name+"/data/errors_without.csv")
        elif method == "implicit":
            torch.save(self.energy, self.name+'/saved_models/implicit_jacobi_energy')
            torch.save(self.jac_vec, self.name+'/saved_models/implicit_jacobi_J')
            errors_df.to_csv(self.name+"/data/errors_implicit.csv")
        elif method == "soft":
            torch.save(self.energy, self.name+'/saved_models/soft_jacobi_energy')
            torch.save(self.L_tensor, self.name+'/saved_models/soft_jacobi_L')
            errors_df.to_csv(self.name+"/data/errors_soft.csv")

# The class `LearnerIMR` is a subclass of `Learner` and contains methods for calculating different
# types of loss functions for a neural network model.
class LearnerIMR(Learner):
    def mov_loss_without(self, zn_tensor, zn2_tensor, mid_tensor):
        """
        The function calculates the loss for a given input tensor by computing the energy, and then combining them with other tensors.
        
        :param zn_tensor: The `zn_tensor` parameter is a tensor representing the current state of the system
        :param zn2_tensor: The `zn2_tensor` parameter is a tensor representing the state at time `t-2*dt`
        :param mid_tensor: The `mid_tensor` parameter is a tensor representing the intermediate state of the system. It is used to compute various quantities such as energy (`En`), and the gradient of energy (`E_z`) with respect to `mid_tensor`. These quantities
        :return: returns `(zn_tensor - zn2_tensor)/self.dt + ham`.
        """
        En = self.energy(mid_tensor)
        E_z = torch.autograd.grad(En.sum(), mid_tensor, only_inputs=True, create_graph=True)[0]
        Lz = self.forward_L_tensor(mid_tensor)
        ham = torch.matmul(Lz, E_z.unsqueeze(2)).squeeze()
        return (zn_tensor - zn2_tensor)/self.dt + ham
                                                        
    def mov_loss_without_with_jacobi(self, zn_tensor, zn2_tensor, mid_tensor, reduced_L):
        """
        The function calculates the moving loss and Jacobi loss for given tensors.
        
        :param zn_tensor: The zn_tensor is a tensor representing the current state of the system. It is used as input to calculate the mov_loss and jacobi_loss
        :param zn2_tensor: The `zn2_tensor` parameter is a tensor representing the second frame of a video sequence
        :param mid_tensor: The `mid_tensor` parameter is a tensor that represents the intermediate state of the model during training. It is typically used to calculate losses or perform other operations
        :param reduced_L: The parameter "reduced_L" is a reduced version of the L tensor. It is used in the calculation of the Jacobi loss
        :return: two values: mov_loss and jacobi_loss.
        """
        mov_loss = self.mov_loss_without(zn_tensor, zn2_tensor, mid_tensor)
        Lz = self.forward_L_tensor(mid_tensor)
        jacobi_loss = self.jacobi_loss(mid_tensor, Lz, reduced_L) 
        return mov_loss, jacobi_loss

    def mov_loss_soft(self, zn_tensor, zn2_tensor, mid_tensor, reduced_L):
        """
        The function calculates the moving loss and Jacobi loss for a given set of tensors.
        
        :param zn_tensor: The zn_tensor is a tensor representing the first set of input data for the mov_loss_soft function
        :param zn2_tensor: The `zn2_tensor` parameter is a tensor representing the second zero-normalized tensor
        :param mid_tensor: The `mid_tensor` parameter is a tensor that represents the intermediate output of a neural network model. It is used as input to calculate the moving loss and Jacobi loss
        :param reduced_L: The parameter "reduced_L" is a reduced version of the L tensor. It is used in the calculation of the Jacobi loss
        :return: two values: mov_loss and jacobi_loss.
        """
        mov_loss = self.mov_loss_without(zn_tensor, zn2_tensor, mid_tensor)
        Lz = self.forward_L_tensor(mid_tensor)
        #Jacobi
        jacobi_loss = self.jacobi_loss(mid_tensor, Lz, reduced_L) 
        return mov_loss, jacobi_loss

    def mov_loss_implicit(self, zn_tensor, zn2_tensor, mid_tensor):
        """
        The function calculates the implicit loss for a given input tensor.
        
        :param zn_tensor: The `zn_tensor` parameter represents the current state of the system at time `n`
        :param zn2_tensor: The `zn2_tensor` parameter is a tensor representing the state at time `t - dt`
        :param mid_tensor: The `mid_tensor` parameter represents the input tensor for which the energy and Jacobian vectors are calculated
        :return: the result of the expression `(zn_tensor - zn2_tensor)/self.dt + torch.cross(Jz, E_z, dim=1)`.
        """
        if self.dissipative:
            raise Exception("Not yet implemented")
        En = self.energy(mid_tensor)
        E_z = torch.autograd.grad(En.sum(), mid_tensor, only_inputs=True, create_graph=True)[0]
 
        Jz, cass = self.jac_vec(mid_tensor)
        Jz2, cass2 = self.jac_vec(mid_tensor)
        return (zn_tensor - zn2_tensor)/self.dt + torch.cross(Jz, E_z, dim=1) 


class LearnerRK4(Learner):
    def z_dot(self, zn_tensor):
        En = self.energy(zn_tensor)
        E_z = torch.autograd.grad(En.sum(), zn_tensor, only_inputs=True, create_graph=True)[0]
        Lz = self.forward_L_tensor(zn_tensor)
        return torch.matmul(Lz, E_z.unsqueeze(2)).squeeze()

    def mov_loss_without(self, zn_tensor, zn2_tensor, mid_tensor):
        k1 = self.dt * self.z_dot(zn_tensor)
        k2 = self.dt * self.z_dot(zn_tensor + k1/2)
        k3 = self.dt * self.z_dot(zn_tensor + k2/2)
        k4 = self.dt * self.z_dot(zn_tensor+ k3)
        return (zn_tensor-zn2_tensor)/self.dt + 1/6 * (k1 + 2*k2 + 2*k3 +k4)/self.dt
    
    def mov_loss_without_with_jacobi(self, zn_tensor, zn2_tensor, mid_tensor, reduced_L):
        mov_loss = self.mov_loss_without(zn_tensor, zn2_tensor, mid_tensor)
        Lz = self.forward_L_tensor(mid_tensor)
        jacobi_loss = self.jacobi_loss(mid_tensor, Lz, reduced_L) 
        return mov_loss, jacobi_loss
    
    def mov_loss_soft(self, zn_tensor, zn2_tensor, mid_tensor, reduced_L):
        mov_loss = self.mov_loss_without(zn_tensor, zn2_tensor, mid_tensor)
        Lz = self.forward_L_tensor(mid_tensor)
        jacobi_loss = self.jacobi_loss(mid_tensor, Lz, reduced_L) 
        return mov_loss, jacobi_loss
    
    def mov_loss_implicit(self, zn_tensor, zn2_tensor, mid_tensor):
        En = self.energy(mid_tensor)
        E_z = torch.autograd.grad(En.sum(), mid_tensor, only_inputs=True, create_graph=True)[0]
 
        Jz, cass = self.jac_vec(mid_tensor)
        Jz2, cass2 = self.jac_vec(mid_tensor)
        return (zn_tensor - zn2_tensor)/self.dt + torch.cross(Jz, E_z, dim=1) 

def check_folder(name):
    """
    The function `check_folder` checks if the specified folder exists, and if not, creates it along with
    two subfolders named "data" and "saved_models".
    
    :param name: The `name` parameter is the name of the folder that you want to check and create if it doesn't exist
    """
    print("Checking folder ", name)
    name = os.getcwd()+"/"+name
    data_name = name+"/data"
    models_name = name+"/saved_models"
    if not os.path.exists(data_name):
        print("Making folder: "+data_name)
        os.makedirs(data_name)
    if not os.path.exists(models_name):
        print("Making folder: "+models_name)
        os.makedirs(models_name)

# The above code is a Python script that defines a command-line interface using the `argparse` module.
# It allows the user to specify various parameters and options when running the script.
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", default=DEFAULT_epochs, type=int, help="Number of epochs")
    parser.add_argument("--neurons", default=DEFAULT_neurons, type=int, help="Number of hidden neurons")
    parser.add_argument("--layers", default=DEFAULT_layers, type=int, help="Number of layers")
    parser.add_argument("--batch_size", default=DEFAULT_batch_size, type=int, help="Batch size")
    parser.add_argument("--dt", default=DEFAULT_dt, type=float, help="Step size")
    parser.add_argument("--prefactor", default=DEFAULT_prefactor, type=float, help="Prefator in the loss")
    parser.add_argument("--learning_rate", default=DEFAULT_learning_rate, type=float, help="Learning rate")
    parser.add_argument("--model", type=str, help="Model = RB, HT, or P3D.", required = True)
    parser.add_argument("--name", default = DEFAULT_folder_name, type=str, help="Folder name")
    parser.add_argument("--method", default = "without", type=str, help="Method: without, implicit, or soft")
    #parser.parse_args(['-h'])

    args = parser.parse_args([] if "__file__" not in globals() else None)

    check_folder(args.folder_name) #check whether the folders data and saved_models exist, or create them

    learner = Learner(args.model, neurons = args.neurons, layers = args.layers, batch_size = args.batch_size, dt = args.dt, name = args.folder_name)
    learner.learn(method = args.method, learning_rate = args.learning_rate, epochs = args.epochs, prefactor = args.prefactor)



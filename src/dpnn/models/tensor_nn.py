"""Neural network models for Jacobian tensors and related structures."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TensorNet(nn.Module):
    """Neural network that outputs an antisymmetric tensor (Poisson structure)."""
    
    def __init__(self, dim, neurons, layers, batch_size, dropout_rate=0.0):
        """
        Initialize the TensorNet model.
        
        :param dim: Dimensionality of the input data
        :param neurons: Number of neurons in each hidden layer
        :param layers: Number of hidden layers
        :param batch_size: Batch size for processing
        :param dropout_rate: Dropout rate for regularization
        """
        super(TensorNet, self).__init__()
        self.dim = dim
        self.neurons = neurons
        self.layers = layers
        self.batch_size = batch_size

        self.inputDense = nn.Linear(dim, neurons)
        self.hidden = [nn.Linear(neurons, neurons) for _ in range(layers-1)]
        self.hidden = nn.ModuleList(self.hidden)
        self.outputSize = int(dim*(dim-1)/2)
        self.outputDense = nn.Linear(neurons, self.outputSize)
        self.sym_sing = -1.0
        
        self.register_buffer('tri_i', torch.triu_indices(dim, dim, 1))
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        """
        Forward pass through the network.
        
        :param x: Input tensor
        :return: Antisymmetric tensor L (batch, dim, dim)
        """
        x = self.inputDense(x)
        x = F.softplus(x)
        x = self.dropout(x)
        for layer in self.hidden:
            x = layer(x)
            x = F.softplus(x)
            x = self.dropout(x)

        data = self.outputDense(x)
        b_n = data.size(0) if data.dim() > 1 else 1

        z = torch.zeros(b_n, self.dim, self.dim, device=data.device, dtype=data.dtype)        
        tri_i0, tri_i1 = self.tri_i
        z[:, tri_i0, tri_i1] = data

        output = z + self.sym_sing * z.transpose(1, 2)
        return output

    def get_jacobian(self, x):
        """
        Compute the Jacobian of the output with respect to the input.
        
        :param x: Input tensor
        :return: Jacobian tensor (batch, dim, dim, dim)
        """
        B = x.size(0)
        preactivations = []
        x_processed = x
        for layer in [self.inputDense] + list(self.hidden):
            x_processed = layer(x_processed)
            preactivations.append(x_processed)
            x_processed = F.softplus(x_processed)

        J_mlp = self.outputDense.weight.unsqueeze(0).expand(B, -1, -1)

        for i in reversed(range(len(self.hidden))):
            s = torch.sigmoid(preactivations[i + 1])
            J_mlp = J_mlp * s.unsqueeze(1)
            W_prev = self.hidden[i].weight
            J_mlp = torch.matmul(J_mlp, W_prev)

        s = torch.sigmoid(preactivations[0])
        J_mlp = J_mlp * s.unsqueeze(1)
        J_mlp = torch.matmul(J_mlp, self.inputDense.weight)
        
        final_J = torch.zeros(B, self.dim, self.dim, self.dim, device=x.device, dtype=x.dtype)
        
        row_indices, col_indices = self.tri_i
        final_J[:, row_indices, col_indices, :] = J_mlp
        final_J[:, col_indices, row_indices, :] = self.sym_sing * J_mlp

        return final_J


class JacVectorNet(nn.Module):
    """Neural network for computing Jacobian vectors (implicit Jacobi case)."""
    
    def __init__(self, dim, neurons, layers, batch_size, dropout_rate=0.0):
        """
        Initialize the JacVectorNet model.
        
        :param dim: Dimensionality of the input data
        :param neurons: Number of neurons in each hidden layer
        :param layers: Number of hidden layers
        :param batch_size: Batch size for processing
        :param dropout_rate: Dropout rate for regularization
        """
        super(JacVectorNet, self).__init__()
        self.dim = dim
        self.neurons = neurons
        self.layers = layers
        self.batch_size = batch_size

        self.inputDense = nn.Linear(3, neurons)
        self.hidden = [nn.Linear(neurons, neurons) for i in range(layers-1)]
        self.hidden = nn.ModuleList(self.hidden)
        self.multiplier = nn.Linear(neurons, 1)
        self.cassimir = nn.Linear(neurons, 1)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, inp):
        """
        Forward pass through the network.
        
        :param inp: Input tensor
        :return: Tuple of (multiplied gradient, Casimir value)
        """
        x = self.inputDense(inp)
        x = F.softplus(x)
        x = self.dropout(x)
        for i in range(self.layers-1):
            x = self.hidden[i](x)
            x = F.softplus(x)
            x = self.dropout(x)
        multi = self.multiplier(x)
        cass = self.cassimir(x)
        cass_grad = torch.autograd.grad(torch.sum(cass), inp, only_inputs=True, create_graph=True)[0]

        return multi * cass_grad, cass

"""Neural network model for energy functions."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class EnergyNet(nn.Module):
    """Neural network that maps states to energy values."""
    
    def __init__(self, dim, neurons, layers, batch_size, dropout_rate=0.0, quad_features=False):
        """
        Initialize the EnergyNet model.
        
        :param dim: Dimensionality of the input data
        :param neurons: Number of neurons in each hidden layer
        :param layers: Number of hidden layers
        :param batch_size: Batch size for processing
        :param dropout_rate: Dropout rate for regularization
        :param quad_features: Whether to include quadratic features
        """
        super(EnergyNet, self).__init__()
        self.dim = dim
        self.neurons = neurons
        self.layers = layers
        self.batch_size = batch_size
        self.quad_features = quad_features
        self.input_dim = dim + (dim * (dim + 1)) // 2 if quad_features else dim

        self.inputDense = nn.Linear(self.input_dim, neurons)
        self.hidden = [nn.Linear(neurons, neurons) for i in range(layers-1)]
        self.hidden = nn.ModuleList(self.hidden)
        self.outputDense = nn.Linear(neurons, 1)
        self.dropout = nn.Dropout(dropout_rate)

        if quad_features:
            self.register_buffer('quad_indices', torch.triu_indices(dim, dim))

    def forward(self, x):
        """
        Forward pass through the network.
        
        :param x: Input tensor
        :return: Energy value (scalar or batch of scalars)
        """
        if self.quad_features:
            if x.dim() == 1:
                quadratic_features = torch.outer(x, x)[self.quad_indices[0], self.quad_indices[1]]
                x = torch.cat([x, quadratic_features], dim=0)
            else:
                outer_product = x.unsqueeze(2) * x.unsqueeze(1)
                quadratic_features = outer_product[:, self.quad_indices[0], self.quad_indices[1]]
                x = torch.cat([x, quadratic_features], dim=1)

        x = self.inputDense(x)
        x = F.softplus(x)
        x = self.dropout(x)
        for i in range(self.layers-1):
            x = self.hidden[i](x)
            x = F.softplus(x)
            x = self.dropout(x)
        output = self.outputDense(x)
        return output

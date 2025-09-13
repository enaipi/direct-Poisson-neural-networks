import torch
import torch.nn as nn
import torch.nn.functional as F


# The `EnergyNet` class is a neural network model that takes in input data and outputs a single value,
# using a specified number of layers and neurons per layer.
class EnergyNet(nn.Module):
    def __init__(self, dim, neurons, layers, batch_size, dropout_rate=0.0, quad_features=False):  # added dropout and quad features parameter
        """
        The function initializes a neural network with a specified number of dimensions, neurons,
        layers, and batch size.
        
        :param dim: The `dim` parameter represents the dimensionality of the input data. It specifies the number of features or variables in the input data
        :param neurons: The "neurons" parameter represents the number of neurons in each hidden layer of the neural network
        :param layers: The "layers" parameter represents the number of hidden layers in the neural network
        :param batch_size: The batch size is the number of samples that will be propagated through the network at once. It is used to divide the dataset into smaller batches for efficient training
        """
        super(EnergyNet, self).__init__()
        self.dim = dim
        self.neurons = neurons
        self.layers = layers
        self.batch_size = batch_size

        self.quad_features = quad_features
        self.input_dim = dim + (dim * (dim + 1)) // 2 if quad_features else dim  # input dim suitable for quad features

        self.inputDense = nn.Linear(self.input_dim, neurons)
        self.hidden = [nn.Linear(neurons, neurons)
                       for i in range(layers-1)]
        self.hidden = nn.ModuleList(self.hidden)
        self.outputDense = nn.Linear(neurons, 1)

        self.dropout = nn.Dropout(dropout_rate)

        if quad_features:
            self.register_buffer('quad_indices', torch.triu_indices(dim, dim))

    # x represents our data
    def forward(self, x):
        """
        The forward function takes an input x, applies a series of dense and activation layers, and
        returns the output.
        
        :param x: The parameter `x` represents the input to the neural network. It is passed through the input dense layer, followed by a softplus activation function. Then, it is passed through a series of hidden layers, each followed by a softplus activation function. Finally, the output is obtained by passing through the output layer.
        :return: The output of the forward pass through the neural network model.
        """
        if self.quad_features:  # calculation of quadratic features
            if x.dim() == 1:  # for 1 dimensional vectors
                quadratic_features = torch.outer(x, x)[self.quad_indices[0], self.quad_indices[1]]
                x = torch.cat([x, quadratic_features], dim=0)
            else:
                outer_product = x.unsqueeze(2) * x.unsqueeze(1)
                quadratic_features = outer_product[:, self.quad_indices[0], self.quad_indices[1]]
                x = torch.cat([x, quadratic_features], dim=1)

        x = self.inputDense(x)
        x = F.softplus(x)
        x = self.dropout(x)  # added dropout
        for i in range(self.layers-1):
            x = self.hidden[i](x)
            x = F.softplus(x)
            x = self.dropout(x)  # added dropout
        output = self.outputDense(x)
        return output


#antisymmetry
class TensorNet(nn.Module):
    def __init__(self, dim, neurons, layers, batch_size, dropout_rate=0.0):  # added dropout parameter
        """
        The function initializes a neural network with a specified number of dimensions, neurons, layers,
        and batch size, and sets up the necessary linear layers and indices for the network.
        
        :param dim: The "dim" parameter represents the dimensionality of the input data. It specifies the number of features or variables in the input data
        :param neurons: The "neurons" parameter represents the number of neurons in each hidden layer of the neural network
        :param layers: The "layers" parameter represents the number of hidden layers in the neural network
        :param batch_size: The batch_size parameter determines the number of samples that will be processed in each forward pass of the neural network. It represents the number of input data points that will be fed into the network simultaneously
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
        self.sym_sing = -1
        
        """# This locks the maximum bactch size
        # for variable maximum batch size, this would need to be performed every forward pass
        tri_i = torch.triu_indices(dim, dim, 1) 
        batch_i = torch.tensor([i for i in range(batch_size) for _ in range(tri_i.size(1))])
        tri_rep = tri_i.repeat(1, batch_size) 
        self.indices = torch.stack((batch_i, tri_rep[0], tri_rep[1]))"""

        self.register_buffer('tri_i', torch.triu_indices(dim, dim, 1))

        self.dropout = nn.Dropout(dropout_rate)

    # x represents our data
    def forward(self, x):
        """
        The forward function takes an input tensor, applies a series of operations including dense layers
        and activation functions, and returns an output tensor.
        
        :param x: The parameter `x` is the input to the forward function. 
        :return: the variable "output".
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

        z = torch.zeros(b_n, self.dim, self.dim, device=data.device)        
        tri_i0, tri_i1 = self.tri_i
        z[:, tri_i0, tri_i1] = data

        """z[self.indices[0, :b_n*self.outputSize], 
        self.indices[1, :b_n*self.outputSize],
        self.indices[2, :b_n*self.outputSize]] = data.ravel()"""

        output = z + self.sym_sing*z.transpose(1, 2)
        return output
        
#antisymmetry
class JacVectorNet(nn.Module):
    def __init__(self, dim, neurons, layers, batch_size, dropout_rate=0.0):  # added dropout parameter
        """
        The above function is the initialization method for a neural network model called JacVectorNet,
        which takes in parameters for the dimensions, number of neurons, number of layers, and batch size,
        and sets up the layers and linear transformations for the model.
        
        :param dim: The "dim" parameter represents the dimensionality of the input data. In this case, it is not explicitly used in the code snippet provided. 
        :param neurons: The "neurons" parameter represents the number of neurons in each hidden layer of the neural network
        :param layers: The "layers" parameter represents the number of hidden layers in the neural network
        :param batch_size: The batch_size parameter determines the number of samples that will be processed in each forward and backward pass during training.
        """
        super(JacVectorNet, self).__init__()
        self.dim = dim
        self.neurons = neurons
        self.layers = layers
        self.batch_size = batch_size

        self.inputDense = nn.Linear(3, neurons)
        self.hidden = [nn.Linear(neurons, neurons)
                       for i in range(layers-1)]
        self.hidden = nn.ModuleList(self.hidden)
        self.multiplier = nn.Linear(neurons, 1)
        self.cassimir   = nn.Linear(neurons, 1)

        self.dropout = nn.Dropout(dropout_rate)  # defined the dropout module

    # x represents our data
    def forward(self, inp):
        """
        The forward function takes an input, applies a series of operations to it, and returns the
        product of the output and the gradient of the Cassimir term with respect to the input, as well as
        the Cassimir term itself.
        
        :param inp: The `inp` parameter represents the input to the forward method. It is the input data that will be passed through the neural network
        :return: two values: `multi * cass_grad` and `cass`.
        """

        x = self.inputDense(inp)
        x = F.softplus(x)
        x = self.dropout(x)  # added dropout
        for i in range(self.layers-1):
            x = self.hidden[i](x)
            x = F.softplus(x)
            x = self.dropout(x)  # added dropout
        multi = self.multiplier(x)
        cass = self.cassimir(x)
        cass_grad = torch.autograd.grad(torch.sum(cass), inp, only_inputs=True, create_graph=True)[0]

        return multi * cass_grad, cass

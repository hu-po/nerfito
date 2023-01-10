""" NeRF Model."""

import torch
import torch.nn as nn

class NeRF(nn.Module):
    
    def __init__(self,
        num_layers: int = 3,
        num_hidden: int = 256,
        input_size: int = 5,
        output_size: int = 4,
        use_leaky_relu: bool = False,
        ):
        """Initialize the NeRF model.

        Args:
            num_layers (int): Number of layers in the model.
            num_hidden (int): Number of hidden units in each layer.
        """
        super(NeRF, self).__init__()
        self.num_layers = num_layers
        self.num_hidden = num_hidden
        self.use_leaky_relu = use_leaky_relu
        self.layers = nn.ModuleList()
        self.layers.extend([nn.Linear(input_size, num_hidden)])
        self.layers.extend([nn.Linear(num_hidden, num_hidden) for _ in range(num_layers - 2)])
        self.layers.extend([nn.Linear(num_hidden, output_size)])

    def forward(self, x):
        """Forward pass of the NeRF model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 5).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, 4).
        """
        for i in range(self.num_layers - 1):
            if self.use_leaky_relu:
                x = torch.nn.LeakyReLU(self.layers[i](x))
            else:
                x = torch.relu(self.layers[i](x))
        x = self.layers[-1](x)
        return x


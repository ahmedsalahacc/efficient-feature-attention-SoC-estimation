"""Utility functions for models."""

import torch

from torch import nn


class FeatureWiseAttention(nn.Module):
    """Feature wise attention module."""

    def __init__(self, num_features: int, is_residual=False) -> None:
        """Feature wise attention module.

        Parameters
        ----------
        num_features : int
            Number of features in the input tensor.
        is_residual : bool, optional
            Whether to use residual connection, by default False
        """
        super(FeatureWiseAttention, self).__init__()
        self.attention = nn.Linear(num_features, num_features)
        self.is_residual = is_residual

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Output tensor.
        """
        attention = self.attention(x)
        attention = nn.functional.softmax(attention, dim=-1)
        return x * attention + x if self.is_residual else x * attention


class LinearResidualBlock(nn.Module):
    """Linear residual block."""

    def __init__(self, input_size: int, output_size: int) -> None:
        """Linear residual block.

        Parameters
        ----------
        input_size : int
            Input size of the linear layer.
        output_size : int
            Output size of the linear layer.
        """
        super(LinearResidualBlock, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, output_size),
            nn.ReLU(),
            nn.BatchNorm1d(output_size),
            nn.Linear(output_size, output_size),
        )
        self.input_mapping = nn.Linear(input_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Output tensor.
        """
        return self.model(x) + self.input_mapping(x)

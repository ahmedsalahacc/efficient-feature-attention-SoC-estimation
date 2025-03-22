"""SoC Estimator model definition."""

from typing import Tuple, Dict, Any

import torch

from torch import nn

import pytorch_lightning as pl

from models.model_utils import FeatureWiseAttention, LinearResidualBlock


class Model(nn.Module):
    def __init__(
        self,
        input_size,
        num_layers=4,
        hidden_size=128,
        is_residual=False,
    ) -> None:
        """SoC Estimator model.

        Parameters
        ----------
        input_size : int
            Input size.
        num_layers : int
            Number of layers.
        hidden_size : int
            Hidden size.
        is_residual : bool
            Whether to use residual connections.
        """
        super(Model, self).__init__()

        layers = [LinearResidualBlock(input_size, hidden_size)]
        for _ in range(num_layers):
            layers.append(FeatureWiseAttention(hidden_size, is_residual=is_residual))
            layers.append(LinearResidualBlock(hidden_size, hidden_size))
        layers += [nn.Linear(hidden_size, 1), nn.Sigmoid()]

        self.model = nn.Sequential(
            *layers,
        )

        self._kaiming_init()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
            torch.Tensor
            Output tensor."""

        return self.model(x).squeeze()

    @torch.jit.ignore
    def _kaiming_init(self):
        for layer in self.model:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight)
                nn.init.zeros_(layer.bias)


class SoCEstimator(pl.LightningModule):
    def __init__(
        self,
        input_size: int,
        num_layers: int = 4,
        hidden_size: int = 128,
        is_residual: bool = False,
    ) -> None:
        """SoC Estimator model.

        Parameters
        ----------
        input_size : int
            Input size.
        num_layers : int
            Number of layers.
        hidden_size : int
            Hidden size.
        is_residual : bool
            Whether to use residual connections.
        uncertainty : bool
            Whether to use uncertainty.
        """
        super(SoCEstimator, self).__init__()
        self.model = Model(
            input_size=input_size,
            num_layers=num_layers,
            hidden_size=hidden_size,
            is_residual=is_residual,
        )

        self.criterion = nn.MSELoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Output
        """
        return self.model(x)

    def training_step(self, batch: Tuple[torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Training step.

        Parameters
        ----------
        batch : tuple[torch.Tensor]
            Batch of input and target tensors.
        batch_idx : int
            Batch index.

        Returns
        -------
        torch.Tensor
            Loss.
        """
        x, y = batch
        y_pred = self(x)
        loss = self.criterion(y_pred, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(
        self, batch: Tuple[torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Validation step.

        Parameters
        ----------
        batch : tuple[torch.Tensor]
            Batch of input and target tensors.
        batch_idx : int
            Batch index.

        Returns
        -------
        torch.Tensor
            Loss.
        """
        x, y = batch
        y_pred = self(x)
        loss = self.criterion(y_pred, y)
        self.log("val_loss", loss)
        return loss

    def test_step(self, batch: Tuple[torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Test step.

        Parameters
        ----------
        batch : tuple[torch.Tensor]
            Batch of input and target tensors.
        batch_idx : int
            Batch index.

        Returns
        -------
        torch.Tensor
            Loss.
        """
        x, y = batch
        y_pred = self(x)
        loss = self.criterion(y_pred, y)
        self.log("test_loss", loss)
        return loss

    def configure_optimizers(self) -> Dict[str, Any]:
        """Configure optimizer."""
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-3)

        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer, gamma=0.98, verbose=True
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "interval": "epoch",
                "frequency": 1,
            },
        }

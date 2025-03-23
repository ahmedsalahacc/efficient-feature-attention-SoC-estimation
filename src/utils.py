"""Helper functions frequently in the project."""

import os

import glob

import numpy as np

import torch

from typing import List, Protocol, Iterable, Union, Type

from datasets.LGDataset import LGDataset, get_dataloader


class TorchModel(Protocol):
    def parameters(self) -> Iterable[float]:
        """Returns the parameters of the torch model"""


def count_parameters(model: TorchModel) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def fetch_data_files(path: str, mode="train") -> Union[List[str], str]:
    """Fetches the csv files in the path.

    Parameters
    ----------
    path: str
        path to get the files from
    mode: str
        must be either train|valid|test

    Returns
    -------
    List[str]
        List of files
    """
    assert mode in ("train", "valid", "test"), "Mode must be one of train|valid|test."
    path = os.path.join(path, mode, "*.csv")
    files = glob.glob(os.path.abspath(path))

    if mode in ("train", "valid"):
        if files:
            return files[0]
        raise FileNotFoundError(f"{path} has no csv files.")
    else:
        if files:
            return files
        raise FileNotFoundError(f"{path} has no csv files.")


def test_model_on_multiple_temps(model, batch_size, test_files: List[str]) -> None:
    """
    Evaluates the performance of a model on multiple temperature-based datasets.

    This function takes a model and evaluates it on a list of temperature-specific datasets.
    For each dataset, it calculates and prints the performance metrics (MSE, RMSE, MAE, MAXE).
    After evaluating all files, it computes the overall performance across all datasets.

    Parameters
    ----------
    model : torch.nn.Module
        The model to be evaluated. It is assumed that the model has a `.cuda()` method for moving to GPU.

    batch_size : int
        The batch size used for loading the dataset during evaluation.

    test_files : List[str]
        A list of file paths, each representing a temperature-specific dataset in CSV format.
        The model will be evaluated on each file sequentially.

    Returns
    -------
    None
        This function does not return any value. It prints evaluation results for each dataset and overall performance.

    Notes
    -----
    The function only works with `LGDataset` class to load data from the provided file paths.
    The performance metrics are printed in percentage terms.
    """
    # predefine the collective preds and labels placeholder
    y_preds_all = []
    y_trues_all = []

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model.to(device)

    # evaluate each file
    for file in test_files:
        test_dataset = LGDataset(file)
        test_loader = get_dataloader(test_dataset, batch_size=batch_size, shuffle=False)

        y_preds = []
        y_trues = []
        # evaluate preds
        for x, y in test_loader:
            y_preds.append(model(x.to(device)).detach().cpu().numpy())
            y_trues.append(y.numpy())

        # concatenate all batches
        y_preds = np.concatenate(y_preds)
        y_trues = np.concatenate(y_trues)

        y_preds_all += y_preds.tolist()
        y_trues_all += y_trues.tolist()

        # evaluate model performance
        mse = np.mean((y_preds - y_trues) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(y_preds - y_trues))
        maxe = np.max(np.abs(y_preds - y_trues))
        print("File:", file)
        print(
            f"\t-MSE%: {mse*100:.3f}, RMSE%: {rmse*100:.3f}, MAE%: {mae*100:.3f}, MAXE%: {maxe*100:.3f}"
        )
        print()

    # Evaluate overall performance
    print("<Overall>")
    # evaluate model performance
    y_preds_all = np.array(y_preds_all)
    y_trues_all = np.array(y_trues_all)

    mse = np.mean((y_preds_all - y_trues_all) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_preds_all - y_trues_all))
    maxe = np.max(np.abs(y_preds_all - y_trues_all))
    print(
        f"MSE%: {mse*100:.3f}, RMSE%: {rmse*100:.3f}, MAE%: {mae*100:.3f}, MAXE%: {maxe*100:.3f}"
    )

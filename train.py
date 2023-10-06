from typing import Callable, Tuple

import numpy as np
import torch
from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from torch.optim import Optimizer
from tqdm import trange

from mlops_toy_tools.model import DenseNet


def training_loop(
    n_epochs: int,
    network: torch.nn.Module,
    loss_fn: Callable,
    optimizer: Optimizer,
    ds_train: Tuple[torch.Tensor, torch.Tensor],
    ds_test: Tuple[torch.Tensor, torch.Tensor],
    device: torch.device,
    verbose: bool = True,
):
    x_train, y_train = ds_train
    x_test, y_test = ds_test

    train_losses, test_losses, train_accuracies, test_accuracies = [], [], [], []
    for epoch in trange(n_epochs):
        network.train()

        def closure():
            optimizer.zero_grad()
            pred_train = network(x_train.to(device))
            loss = loss_fn(pred_train, y_train.to(device))
            loss.backward()

            return loss.item()

        optimizer.step(closure)
        network.eval()

        with torch.no_grad():
            pred_train = network(x_train.to(device))

            loss = loss_fn(pred_train, y_train.to(device))
            train_losses.append(loss.item())

            score = accuracy_score(
                y_train, torch.argmax(pred_train, dim=-1).detach().cpu().numpy()
            )
            train_accuracies.append(score * 100)

            pred_test = network(x_test.to(device))
            loss = loss_fn(pred_test, y_test.to(device))
            test_losses.append(loss.item())

            score = accuracy_score(
                y_test, torch.argmax(pred_test, dim=-1).detach().cpu().numpy()
            )
            test_accuracies.append(score * 100)

            if epoch % 20 == 0 and verbose:
                msg = 'Loss (Train/Test): {0:.3f}/{1:.3f}.'
                msg += 'Accuracy, % (Train/Test): {2:.2f}/{3:.2f}'
                print(
                    msg.format(
                        train_losses[-1],
                        test_losses[-1],
                        train_accuracies[-1],
                        test_accuracies[-1],
                    )
                )

    return train_losses, test_losses, train_accuracies, test_accuracies


def tensor_convert(np_array: np.ndarray) -> torch.Tensor:
    return torch.tensor(np_array, device=torch.device('cpu'))


if __name__ == '__main__':
    # Loading dataset
    X, y = load_digits(return_X_y=True)
    X_train, X_test, y_train, y_test = map(
        tensor_convert, train_test_split(X, y, train_size=1 / 4, test_size=3 / 4)
    )

    model = DenseNet(
        in_features=64, hidden_size=32, n_classes=10, n_layers=3, activation=torch.nn.ReLU
    )

    # Initializing optimizer
    optimizer = torch.optim.LBFGS(model.parameters(), max_iter=1)

    # Setting loss function
    loss_fn = torch.nn.CrossEntropyLoss()

    # Setting device
    if torch.cuda.is_available():
        device = torch.device('cuda', 0)
    else:
        device = torch.device('cpu')
    model.to(device)

    # Starting training
    train_losses, test_losses, train_accs, test_accs = training_loop(
        n_epochs=200,
        network=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        ds_train=(X_train, y_train),
        ds_test=(X_test, y_test),
        device=device,
    )

    print(f'Cross Entropy (Train/Test): {train_losses[-1]:.2f}/{test_losses[-1]:.2f}')
    print(f'Accuracy (Train/Test): {train_accs[-1]:.2f}/{test_accs[-1]:.2f}')

    torch.save(model, 'checkpoints/dense.pt')

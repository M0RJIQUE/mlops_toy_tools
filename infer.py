import csv

import torch
from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from train import tensor_convert


def write_csv(results, file_name):
    with open(file_name, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['Ground truth', 'Prediction'])
        writer.writerows(results)


if __name__ == '__main__':
    model = torch.load('checkpoints/dense.pt')
    X, y = load_digits(return_X_y=True)
    X_train, X_test, y_train, y_test = map(
        tensor_convert, train_test_split(X, y, train_size=1 / 4, test_size=3 / 4)
    )
    y_pred = torch.argmax(model(X_test), dim=-1).detach().cpu()

    accuracy = accuracy_score(y_pred.numpy(), y_test.numpy())
    print(f'TEST ACCURACY = {accuracy:0.3}')

    write_csv(results=zip(y_test.tolist(), y_pred.tolist()), file_name='results.csv')

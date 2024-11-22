'''This script provides definition and function of the most common
metrics for evaluating the performance of a regression model.'''

import torch


def pcc(pred, true):
    '''Pearson correlation coefficient between the pred and true.'''

    if pred.shape != true.shape:
        raise ValueError('pred and true must have the same shape')

    pred = pred - torch.mean(pred)
    true = true - torch.mean(true)

    numerator = torch.sum(pred * true)
    denominator = torch.sqrt(torch.sum(pred**2) * torch.sum(true**2))

    return (numerator / denominator).item()


def rmse(pred, true):
    '''Root mean squared error between the pred and true.'''

    if pred.shape != true.shape:
        raise ValueError('pred and true must have the same shape')

    return (torch.sqrt(torch.mean((pred - true) ** 2))).item()


def mae(pred, true):
    '''Mean absolute error between the pred and true.'''

    if pred.shape != true.shape:
        raise ValueError('pred and true must have the same shape')

    return (torch.mean(torch.abs(pred - true))).item()


def mse(pred, true):
    '''Mean squared error between the pred and true.'''

    if pred.shape != true.shape:
        raise ValueError('pred and true must have the same shape')

    return (torch.mean((pred - true) ** 2)).item()


def r2(pred, true):
    '''R^2 score between the pred and true.'''

    if pred.shape != true.shape:
        raise ValueError('pred and true must have the same shape')

    numerator = torch.sum((pred - true) ** 2)
    denominator = torch.sum((true - torch.mean(true)) ** 2)

    return (1 - numerator / denominator).item()


if __name__ == '__main__':

    from time import time
    import sklearn.metrics as m

    pred = torch.tensor([1, 2, 3, 4, 5.5], dtype=torch.float)
    true = torch.tensor([9, 8, 7, 6, 57], dtype=torch.float)

    print('r2')
    start = time()
    val = m.r2_score(true, pred)
    print(f'sklearn : {val:.4f} ({time() - start:.8f} sec)')
    start = time()
    val = r2(pred, true)
    print(f'mine    : {val:.4f} ({time() - start:.8f} sec)')
    print()

    print('MSE')
    start = time()
    val = m.mean_squared_error(true, pred)
    print(f'sklearn : {val:.4f} ({time() - start:.8f} sec)')
    start = time()
    val = mse(pred, true)
    print(f'mine    : {val:.4f} ({time() - start:.8f} sec)')
    print()

    print('RMSE')
    start = time()
    val = m.mean_squared_error(true, pred, squared=False)
    print(f'sklearn : {val:.4f} ({time() - start:.8f} sec)')
    start = time()
    val = rmse(pred, true)
    print(f'mine    : {val:.4f} ({time() - start:.8f} sec)')
    print()

    print('MAE')
    start = time()
    val = m.mean_absolute_error(true, pred)
    print(f'sklearn : {val:.4f} ({time() - start:.8f} sec)')
    start = time()
    val = mae(pred, true)
    print(f'mine    : {val:.4f} ({time() - start:.8f} sec)')

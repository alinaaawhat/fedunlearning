import numpy as np
import torch
from torch.utils.data import DataLoader
from dataset.data_utils import data_set, separate_data, split_proxy


def data_init(FL_params):
    kwargs = {'pin_memory': True} if FL_params.device == 'cuda' else {}
    dataset_x = []
    dataset_y = []
    trainset, testset = data_set(FL_params.data_name)
    test_loader = DataLoader(
        testset,
        batch_size=FL_params.test_batch_size,
        shuffle=True,
        num_workers=min(32, 4),
        **kwargs,
    )
    train_loader = DataLoader(
        trainset,
        batch_size=FL_params.local_batch_size,
        shuffle=True,
        num_workers=min(32, 4),
        **kwargs,
    )
    for x_train, y_train in train_loader:
        dataset_x.extend(x_train.cpu().detach().numpy())
        dataset_y.extend(y_train.cpu().detach().numpy())
    if FL_params.forget_paradigm == 'client':
        for x_test, y_test in test_loader:
            dataset_x.extend(x_test.cpu().detach().numpy())
            dataset_y.extend(y_test.cpu().detach().numpy())
    dataset_x = np.array(dataset_x)
    dataset_y = np.array(dataset_y)
    X, y, statistic = separate_data(
        (dataset_x, dataset_y),
        FL_params.num_user,
        FL_params.num_classes,
        FL_params,
        FL_params.niid,
        FL_params.balance,
        FL_params.partition,
        class_per_client=2,
    )
    client_loaders, test_loaders, proxy_client_loaders, proxy_test_loaders = split_proxy(
        X, y, FL_params
    )
    FL_params.datasize_ls = [len(k) for k in X]
    return client_loaders, test_loaders, proxy_client_loaders, proxy_test_loaders

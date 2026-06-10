import pickle
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import copy
def load_data(file_path="Know_regression.pkl"):
    with open(file_path, "rb") as f:
        data = pickle.load(f)
    return data

def preprocess_data(data):
    # Reshaping and preprocessing as per your instructions
    temp = np.swapaxes(data, 0, 1)
    P_W_data = np.swapaxes(temp, 1, 2)
    com_P_W_data = np.reshape(P_W_data, (731*96, 13))

    # Adding time and date labels
    time_label = torch.ones(70176).unsqueeze(1).numpy()
    for i in range(731):
        time_label[96*i:96*(i+1)] = np.arange(96)

    days = np.concatenate((np.arange(365), np.arange(366)))
    n_days = np.repeat(days, 96)[:, np.newaxis]

    com_P_W_data = np.concatenate((com_P_W_data, time_label, n_days), axis=1)

    # Monthly label
    months = np.concatenate([np.arange(12), np.arange(12)])
    month_label = months.repeat(30*96)
    month_label = np.pad(month_label, (0, len(com_P_W_data)-len(month_label)), 'constant')[:, np.newaxis]
    com_P_W_data = np.concatenate((com_P_W_data, month_label), axis=1)

    # Select specific features for P_W_data_inverterA
    # Assuming the order and the inclusion of time, day, and month labels are correct, adjust indices as needed
    feature_indices = [0, 1, 2, 5, 7, 8, 9, 10, 11, 12, 13, 14] 
    P_W_data_inverterA = com_P_W_data[:, feature_indices]

    return P_W_data_inverterA

def normalize_data(data):
    for d in range(data.shape[-1]):
        data[..., d] = (data[..., d] - np.min(data[..., d])) / (np.max(data[..., d]) - np.min(data[..., d]) + 1e-6)
    return data

def split_data(data, ratio=0.7, valid_ratio=0.15):
    # Assuming the last dimension is the feature dimension
    train_idx = round(ratio * len(data))
    valid_idx = train_idx + round(valid_ratio * len(data))
    train_data = data[:train_idx]
    valid_data = data[train_idx:valid_idx]
    test_data = data[valid_idx:]
    return train_data, valid_data, test_data

def create_dataloaders(train_data, valid_data, test_data, batch_size=32, device="cuda:0"):
    def data_to_tensor(data):
        X = torch.tensor(data[..., :-1], dtype=torch.float32).to(device)
        Y = torch.tensor(data[..., -1], dtype=torch.float32).to(device)
        return TensorDataset(X, Y)

    train_dataset = data_to_tensor(train_data)
    valid_dataset = data_to_tensor(valid_data)
    test_dataset = data_to_tensor(test_data)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, valid_loader, test_loader

def cons_data(traj_train, device='cuda:0', con_batchsize=12):
    """
    Generate constraint data loaders for derate and off constraints.

    Parameters:
    - traj_train: The training dataset array.
    - device: The device to use ('cuda:0' for GPU or 'cpu').
    - con_batchsize: The batch size for the constraint data loaders.

    Returns:
    - derate_loader: DataLoader for the derate constraints.
    - off_loader: DataLoader for the off constraints.
    """
    rng = np.random.default_rng()

    # IGBT-related constraints setup
    output_1_index = np.where(traj_train[:, 0] > 0.2)[0]
    output_1 = traj_train[:, 1:][output_1_index][:18432]
    output_1[:, 6] = 0  # Assuming index 6 is correct, adjust as needed.

    # Derate constraints
    derate_1 = copy.deepcopy(output_1)
    derate_1[:, 4] = rng.integers(1, 4) / 9  # Adjust index as per your dataset
    derate_0 = copy.deepcopy(output_1)
    derate_0[:, 4] = 0

    # Off constraints
    off_1 = copy.deepcopy(output_1)
    off_1[:, 5] = rng.integers(1, 2) / 9  # Adjust index as per your dataset
    off_0 = copy.deepcopy(output_1)
    off_0[:, 5] = 0

    # Helper function to convert numpy arrays to DataLoader
    def numpy_to_loader(array0, array1):
        tensor0 = torch.from_numpy(array0).type(torch.Tensor).to(device)
        tensor1 = torch.from_numpy(array1).type(torch.Tensor).to(device)
        return DataLoader(TensorDataset(tensor0, tensor1), batch_size=con_batchsize, shuffle=True)

    # Creating DataLoaders
    derate_loader = numpy_to_loader(derate_0, derate_1)
    off_loader = numpy_to_loader(off_0, off_1)

    return derate_loader, off_loader
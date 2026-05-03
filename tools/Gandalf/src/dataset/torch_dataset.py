from torch.utils.data import Dataset

import glob


class TorchDataset(Dataset):
    dim_convert = [(0), (1, 0), (2, 0, 1), (3, 0, 1, 2)]

    def __init__(self, pattern, method):
        self.data = []
        self.convert_method = method
        for path in glob.glob(pattern):
            self.data.append(path)

    def __getitem__(self, index):
        data_path = self.data[index]
        x, y = self.convert_method(data_path)
        return x.transpose(TorchDataset.dim_convert[x.ndim - 1]), y

    def __len__(self):
        return len(self.data)


class TorchMemDataset(Dataset):
    dim_convert = [(0), (1, 0), (2, 0, 1), (3, 0, 1, 2)]

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __getitem__(self, index):
        x, y = self.x[index], self.y[index]
        return x.transpose(TorchMemDataset.dim_convert[x.ndim - 1]), y

    def __len__(self):
        return self.x.shape[0]
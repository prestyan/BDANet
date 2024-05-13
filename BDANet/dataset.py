import torch
from torch.utils.data import Dataset


class EegDataset(Dataset):
    def __init__(self, data, labels, source_or_target, transform=None):
        self.transform = transform
        if source_or_target == 's':
            self.eeg_data = torch.from_numpy(data).float()
            # self.eeg_data = self.eeg_data.permute(2,0,1)
            self.labels = torch.from_numpy(labels).long()
        elif source_or_target == 't':
            self.eeg_data = torch.from_numpy(data).float()
            # self.eeg_data = self.eeg_data.permute(2,0,1)
            self.labels = torch.from_numpy(labels).long()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        x = self.eeg_data[index]
        if self.transform is not None:
            x = self.transform(x)
        y = self.labels[index]
        return x, y
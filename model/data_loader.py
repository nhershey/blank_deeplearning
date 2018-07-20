import os
import pandas as pd
import numpy as np
import h5py
import torch

from tqdm import tqdm

from torch.utils.data import Dataset, DataLoader

from random import shuffle # used to shuffle initial file names
from random import randint # used to randomly pull nonSeizures

class HeadacheDataset(Dataset):
    """
    A standard PyTorch definition of Dataset which defines the functions __len__ and __getitem__.
    """

    def __init__(self, data_dir, split_type):
        """
        Store the filenames of the seizures to use.

        Args:
            data_dir: (string) directory containing the dataset
            split_type: (string) whether train, val, or test set
        """
        all_df = pd.read_csv(data_dir)
        for c in all_df:
            if all_df[c].dtype == 'object':
                all_df = all_df.drop(c, axis=1)

        total = all_df.shape[0]
        if split_type == 'train':
            self.df = all_df[: int(total * 0.9)]
        elif split_type == 'val':
            self.df = all_df[int(total * 0.9) : int(total * 0.95)]
        elif split_type == 'test':
            self.df = all_df[int(total * 0.95) : int(total)]

        self.data_dir = data_dir

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        """
        Fetch index idx seizure and label from dataset.

        Args:
            idx: (int) index in [0, 1, ..., size_of_dataset-1]

        Returns:
            image: (Tensor) transformed image
            label: (int) corresponding label of image
        """
        pd_row = self.df.iloc[[idx]]
        np_row = pd_row.as_matrix()
        pt_row = torch.FloatTensor(np_row)
        pt_col = torch.t(pt_row)
        tup = torch.split(pt_col, pt_col.shape[0]-1)
        return (tup[0], int(tup[1][0][0])) # tup[1][0][0] has the raw value of the tensor


def fetch_dataloader(types, data_dir, file_dir, params):
    """
    Fetches the DataLoader object for each type in types from data_dir.

    Args:
        types: (list) has one or more of 'train', 'val', 'test' depending on which data is required
        data_dir: (string) directory containing the dataset
        file_dir: (string) directory containing the list of file names to pull in
        params: (Params) hyperparameters

    Returns:
        data: (dict) contains the DataLoader object for each type in types
    """
    dataloaders = {}

    for split in ['train', 'val', 'test']:
        if split in types:
            dl = DataLoader(HeadacheDataset(data_dir, split), batch_size=params.batch_size, shuffle=True,
                                        num_workers=params.num_workers,
                                        pin_memory=params.cuda)
            dataloaders[split] = dl

    return dataloaders

if __name__ == '__main__':
    """ used for testing"""
    PATH_TO_DATA = "../../Dad/NickFinalData.csv"
    sd = HeadacheDataset(PATH_TO_DATA, 'train')
    print sd[5]
    # for i, (train_batch, labels_batch) in enumerate(sd):
    #     print(i, train_batch.shape)

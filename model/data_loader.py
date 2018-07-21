import os
import logging
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

        self.y_df = all_df.iloc[:,-1]
        x_df = all_df.iloc[:,:-1]

        for c in x_df:
            # remove non-numeric with more than 10 categories (likely errors)
            if ( (x_df[c].dtype != 'float64' and len(x_df[c].value_counts())) > 10):
                logging.info("dropping column {}.".format(c))
                x_df = x_df.drop(c, axis=1)

            # remove columns with just one value
            elif (len(x_df[c].value_counts()) == 1):
                logging.info("dropping column {}.".format(c))
                x_df = x_df.drop(c, axis=1)

        # turn categorical into dummy
        x_df = pd.get_dummies(x_df)

        # standardize
        x_df = (x_df - x_df.mean()) / x_df.std()

        # split
        total = x_df.shape[0]
        if split_type == 'train':
            self.x_df = x_df[: int(total * 0.9)]
        elif split_type == 'val':
            self.x_df = x_df[int(total * 0.9) : int(total * 0.95)]
        elif split_type == 'test':
            self.x_df = x_df[int(total * 0.95) : int(total)]

        self.num_input_features = self.x_df.shape[1]
        self.data_dir = data_dir

    def __len__(self):
        return self.x_df.shape[0]

    def __getitem__(self, idx):
        """
        Fetch index idx seizure and label from dataset.

        Args:
            idx: (int) index in [0, 1, ..., size_of_dataset-1]

        Returns:
            image: (Tensor) transformed image
            label: (int) corresponding label of image
        """
        pd_row = self.x_df.iloc[[idx]]
        np_row = pd_row.as_matrix()
        pt_row = torch.FloatTensor(np_row)
        pt_col = torch.t(pt_row)
        return (pt_col.squeeze(), self.y_df[idx]) # tup[1][0][0] has the raw value of the tensor

def get_data_path(data_dir):
    file_names = os.listdir(data_dir)
    csv_files = [d for d in file_names if d[-4:] == ".csv"]
    if len(csv_files) == 0:
        logging.info("No .csv files in the put_data_here/ directory")
        exit()
    elif len(csv_files) > 1:
        logging.info("Two or more .csv files in the put_data_here/ directory")
        exit()
    data_file = csv_files[0]
    logging.info("Using {} as the data set.".format(data_file))
    return os.path.join(data_dir, data_file)

def fetch_dataloader(types, data_dir, params):
    """
    Fetches the DataLoader object for each type in types from data_dir.

    Args:
        types: (list) has one or more of 'train', 'val', 'test' depending on which data is required
        data_dir: (string) directory containing the dataset
        params: (Params) hyperparameters

    Returns:
        data: (dict) contains the DataLoader object for each type in types
    """

    # Make sure there is one csv file
    data_path = get_data_path(data_dir)

    dataloaders = {}

    for split in ['train', 'val', 'test']:
        logging.info("CREATING {} DATA SET.".format(split))
        if split in types:
            dl = DataLoader(HeadacheDataset(data_path, split), batch_size=params.batch_size, shuffle=True,
                                        num_workers=params.num_workers,
                                        pin_memory=params.cuda)
            dataloaders[split] = dl

    return dataloaders

if __name__ == '__main__':
    """ used for testing"""
    PATH_TO_DATA = "../../Dad/NickFinalData.csv"
    sd = HeadacheDataset(PATH_TO_DATA, 'train')
    # for i, (train_batch, labels_batch) in enumerate(sd):
    #     logging.info(i, train_batch.shape)

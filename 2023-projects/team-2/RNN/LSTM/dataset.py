import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data import random_split
import lightning.pytorch as pl
from getOneHot import getOneHot

import numpy as np


class DM(pl.LightningDataModule):
    def __init__(self, data_dir, filename, batch_size, num_workers, params):
        super().__init__()
        self.data_dir = data_dir
        self.filename = filename
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.params = params

        print("batchsize: ", self.batch_size)

    def setup(self, stage):
        restricted = [
        'euc1', 'e1', 'x1', 'y1', 'z1',
        'euc2', 'e2', 'x2', 'y2', 'z2',
        'euc3', 'e3', 'x3', 'y3', 'z3',
        ]
        x, y = getOneHot("{}/{}".format(self.data_dir, self.filename), restricted=restricted, **self.params) 
        

        X_tensor = torch.tensor(x).to(torch.float16)
        Y_tensor = torch.tensor(y)
        

        # Create a TensorDataset
        data = TensorDataset(X_tensor, Y_tensor) 


        total_size = len(data)
        val_size = int(self.params['validation'] * total_size)
        train_size = total_size - val_size

        self.train_set, self.val_set = random_split(data, [train_size, val_size])


    def train_dataloader(self):
        return DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_set,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )    

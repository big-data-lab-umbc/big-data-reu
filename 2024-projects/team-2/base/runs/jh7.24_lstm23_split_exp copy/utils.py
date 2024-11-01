import torch
import model as mdl
import sklearn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset, random_split
import lightning as pl
import numpy as np
import torch.nn as nn


def get_model(model_key:str, params=None, pred=False):
    models = {
        'FCN': mdl.FCN,
        'LSTM23': mdl.LSTM2023,
        'impr_fcn': mdl.ImprovedFCN,
        'deep_impr_fcn': mdl.DeepImprovedFCN,
        'impr_lstm23': mdl.ImprovedLSTM2023,
        'impr_lstm24': mdl.ImprovedLSTM2024
    }
    if model_key not in models:
        raise Exception("model key not valid")
    if not pred:
        return models[model_key](**params)
    else:
        return models[model_key]


def get_activation(activation_key: str):
    activations = {
        'relu': nn.ReLU, 
        'leakyrelu': nn.LeakyReLU, 
        'prelu': nn.PReLU,
        'swish': nn.SiLU,
    }
    return activations[activation_key]()


def get_optimizer(optim_key: str):
    optimizers = {
        'adam': torch.optim.Adam, 
    }
    return optimizers[optim_key]


class PGMLDataModule(pl.LightningDataModule):
    def __init__(self, train_data_path='./', test_data_path='./', batch_size:int=32, val_split:float=0.1, split_func = 'pytorch'):
        super().__init__()
        self.train_data_path = train_data_path
        self.test_data_path = test_data_path  # test OR predict
        self.batch_size = batch_size
        self.val_split = val_split
        self.y_truth = torch.empty(1)  # not sure if this is pythonic
        
        if split_func =='pytorch':
            self.split_func = 'pytorch'
        elif split_func =='sklearn':
            self.split_func  = 'sklearn'
        else:
            raise Exception("train test split function not initialized")

    # def prepare_data(self):  # lightning recommended method: not useful for pgml


    def setup(self, stage: str):
        # stage is either 'fit' or 'test'
        # you should prob add predict, or yeah ig you could just use test?
        
        # Assign train/val datasets for use in dataloaders
        if stage == "fit":
            X_path = str(self.train_data_path) + 'X.npy'
            y_path = str(self.train_data_path) + 'y.npy'


           
            if self.split_func == 'pytorch':
                data_full = TensorDataset(torch.tensor(np.load(X_path).astype(np.float32)), torch.tensor(np.load(y_path)))
                self.train, self.val = random_split(data_full, [1-self.val_split, self.val_split])
            else:
                X_np = np.load(X_path).astype(np.float32)
                y_np = np.load(y_path)

                X_train, X_val, y_train, y_val = train_test_split(X_np, y_np, test_size=self.val_split, stratify=y_np)

                self.train =  TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
                self.val  = TensorDataset(torch.tensor(X_val), torch.tensor(y_val))

                



        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage == 'predict':
            print("FLAG: reached test/predict conditional")
            X_path = str(self.test_data_path) + 'X.npy'
            y_path = str(self.test_data_path) + 'y.npy'
            self.test = TensorDataset(torch.tensor(np.load(X_path).astype(np.float32)), torch.tensor(np.load(y_path)))
            self.y_truth = torch.tensor(np.load(y_path))


    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size)
    

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size)


    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size)


    def predict_dataloader(self):
        return DataLoader(self.test)
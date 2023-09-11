import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import torchvision
import lightning.pytorch as pl
import torchmetrics 

class NN(pl.LightningModule):
    def __init__(self, params):
        super().__init__()

        self.indim = params['indim']
        self.num_layers = params['num_layers']
        self.neurons = params['neurons']
        self.layer_type = params['layer_type']
        self.inter_activation = params['inter_activation']
        self.outdim = params['outdim']
        self.dropout = params['dropout']
        self.batch_norm = params['batch_normalization']          
        self.momentum = params['momentum'] if params['withmomentum'] else .1
        self.learning_rate = params['learning_rate']
        self.optimizer = params['optimizer']
        self.weight_decay = params['weight_decay']

        
        ####################### Defining layers for Model ###
        # 4 LSTM layers
        self.lstm = nn.LSTM(input_size=self.indim, hidden_size=self.neurons, num_layers=self.num_layers, batch_first=True)
        
        # Two fully connected layers
        self.fc1 = nn.Linear(self.neurons, 64)  # Adjust the output size as per requirements
        self.fc2 = nn.Linear(64, 32)  # Adjust the output size as per requirements
        self.fc_out = nn.Linear(32, self.outdim)

        ################### end of defining layers ##### 

        self.loss_fn = nn.CrossEntropyLoss()
        self.accuracy = torchmetrics.Accuracy(
            task="multiclass", num_classes=self.outdim
        )
        self.f1_score = torchmetrics.F1Score(task="multiclass", num_classes=self.outdim)
        
    def forward(self, x):
        
         # Initializing hidden state for first input with zeros
        h0 = torch.zeros(4, x.size(0), self.neurons).to(x)

        c0 = torch.zeros(4, x.size(0), self.neurons).to(x)

        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        
        # We need to detach as we are doing truncated backpropagation through time (BPTT)
        # If we don't, we'll backprop all the way to the start even after going through another batch
        # Forward propagation by passing in the input, hidden state, and cell state into the model
        out = out[:, -1, :]
        
        
        # Passing through the fully connected layers
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc_out(out)
        return out

    def get_activation(self):
        name = self.inter_activation
        if name == 'leakyrelu':
            activator = nn.LeakyReLU()
        elif name == 'prelu': #hacky solution to make sure prelu is on the right device
            activator = nn.PReLU().to(next(self.parameters()).device)
        elif name == 'relu':
            activator = nn.ReLU()
        elif name == 'sigmoid':
            activator = nn.Sigmoid()
        return activator 
   
    def training_step(self, batch, batch_idx):
        x, y = batch
        outputs = self(x)
        loss = self.loss_fn(outputs, y)
        self.log("train_loss", loss, on_epoch=True, on_step=False, sync_dist=True)
        with torch.no_grad():
            logits = outputs
            predicted_labels = torch.argmax(logits, 1)
            train_acc = self.accuracy(predicted_labels, y)
            self.log("train_acc", train_acc, on_epoch=True, on_step=False, sync_dist=True)
        return loss  # this is passed to the optimizer for training
 

    def validation_step(self, batch, batch_idx):
        x, y = batch
        outputs = self(x)
        loss = self.loss_fn(outputs, y)
        self.log("val_loss", loss, on_epoch=True, on_step=False, sync_dist=True)

        logits = outputs
        predicted_labels = torch.argmax(logits, 1)
        val_acc = self.accuracy(predicted_labels, y)
        self.log("val_acc", val_acc, on_epoch=True, on_step=False, sync_dist=True)
    

    def configure_optimizers(self):
        print(self.optimizer)

        if self.optimizer == 'adam':
            return optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)

        elif self.optimizer == 'nadam' or self.optimizer == 'adamw':
            # PyTorch does not have an exact equivalent for Nadam,
            # But you can use AdamW which is Adam with weight decay fix
            return optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)

        elif self.optimizer == 'sgd': # weight decay here is the same as l2 regularization
            return optim.SGD(self.parameters(), lr=self.learning_rate, \
                            momentum=self.momentum, weight_decay=self.weight_decay)

        else:
            print("{} optimizer not an acceptable variant.".format(self.optimizer))
            print("Try: Adam, Nadam, or SGD.")
            return None


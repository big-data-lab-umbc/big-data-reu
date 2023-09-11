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
        
        ###### to display the intermediate input- and output sizes of all the layers
        self.example_input_array = torch.Tensor(1443992, 15)

        
        ####################### Defining layers for Model ####
        layers = []

        ###### Input layer####
        layers.append(nn.Linear(self.indim , self.neurons))
        layers.append(nn.ReLU())
        
        # Dynamic hidden layers
        for layer in range(self.num_layers - 1):
            layers.append(nn.Linear(self.neurons , self.neurons))
            activator = get_activation()
            layers.append(activator)
            layers.append(nn.Dropout(self.dropout))

        ###### Output layer ####
        layers.append(nn.Linear(self.neurons , self.outdim))

        self.all_layers = nn.Sequential(*layers) 

        ################### end of defining layers ##### 

        self.loss_fn = nn.CrossEntropyLoss()
        self.accuracy = torchmetrics.Accuracy(
            task="multiclass", num_classes=self.outdim
        )
        self.f1_score = torchmetrics.F1Score(task="multiclass", num_classes=self.outdim)
        
    def forward(self, x):
        logits = self.all_layers(x)
        return logits 

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


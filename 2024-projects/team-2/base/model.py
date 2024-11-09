import lightning as pl
import torch
import torch.nn as nn
import time
import numpy as np
import utils

from torchmetrics import Accuracy


# lightning implementation of reu 2023's golden boy model; some is hard coded
class LSTM2023(pl.LightningModule):
    def __init__(self, indim, outdim, num_layers, neurons, lr, lr_step, lr_gam, dropout, activation, optimizer, l2=0, log_test=False):
        super().__init__()
        self.save_hyperparameters()

        # PL attributes:
        self.train_acc = Accuracy(task='multiclass', num_classes=13)
        self.valid_acc = Accuracy(task='multiclass', num_classes=13)
        self.test_acc = Accuracy(task='multiclass', num_classes=13)

        self.indim = indim
        self.outdim = outdim
        self.num_layers = num_layers
        self.neurons = neurons
        self.lr = lr
        self.lr_step = lr_step
        self.lr_gam = lr_gam
        self.l2=l2

        self.input_layer = nn.Linear(self.indim, self.neurons)
        self.lstm = nn.LSTM(input_size=self.neurons, hidden_size=self.neurons, num_layers=4, batch_first=True)
        self.rnn_layers = self.get_layers()  # dense layers
        self.dropout = nn.Dropout(dropout)
        self.activation = utils.get_activation(activation)
        self.output_layer = nn.Linear(self.neurons, self.outdim)

        self.optimizer = utils.get_optimizer(optimizer)(self.parameters(), lr=lr, weight_decay=self.l2)
        self.start_time = time.time()
        self.log_test = log_test


    def forward(self, x, tgt, dropout_on=True):
        # x, tgt = batch
        x = x.unsqueeze(1)  # adds a sequence length of 1
        x = self.input_layer(x)
        
        # Initializing hidden state for first input with zeros
        h0 = torch.zeros(4, x.size(0), self.neurons).to(x.device).requires_grad_()
        # Initializing cell state for first input with zeros
        c0 = torch.zeros(4, x.size(0), self.neurons).to(x.device).requires_grad_()

        # We need to detach as we are doing truncated backpropagation through time (BPTT)
        # If we don't, we'll backprop all the way to the start even after going through another batch
        # Forward propagation by passing in the input, hidden state, and cell state into the model
        x, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))

        # Reshaping the outputs in the shape of (batch_size, seq_length, hidden_size)
        # so that it can fit into the fully connected layer
        x = x[:, -1, :]

        # RNN layer
        for layer in self.rnn_layers: 
            if isinstance(layer, nn.Linear):
                x = layer(x)
                # dropout -- CHANGED ON 7/19 moved to after activation by mc, changed on 7/30 so it's after every layer by mc
                if dropout_on:
                    x = self.dropout(x) # only want the final timestep output , if it's batch_first
            else:
                x, tgt = layer(x, tgt)
        
        
        x = self.activation(x) #activation function after fully connected layer

        x = self.output_layer(x)
    
        return x


    def get_layers(self):
        layers = nn.ModuleList()
        layer = nn.Linear(self.neurons, self.neurons, bias=True)
        for _ in range(self.num_layers - 1):
            layers.append(layer)
        layers.append(layer)  # this is clearly wrong but it's how they did the run
        return layers


    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x, y)
        loss = nn.functional.cross_entropy(self(x, y), y.view(-1))
        preds = torch.argmax(logits, dim=1)
        self.train_acc(preds, y)
        self.log('train_acc', self.train_acc, on_step=True, on_epoch=True)
        self.log("train_loss", loss, prog_bar=True)
        return loss
    

    def on_train_epoch_end(self):
        self.log('lr', self.optimizer.state_dict()['param_groups'][0]['lr'], sync_dist=True)  # hacky, possibly damaging
        self.log("time", time.time() - self.start_time, sync_dist=True)

    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x, y, dropout_on=False)
        loss = nn.functional.cross_entropy(self(x, y), y.view(-1))
        preds = torch.argmax(logits, dim=1)
        self.valid_acc(preds, y)
        self.log("valid_acc", self.valid_acc, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("valid_loss", loss, prog_bar=True, sync_dist=True)
        return loss
    

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x, y, dropout_on=False)
        loss = nn.functional.cross_entropy(self(x, y), y.view(-1))
        preds = torch.argmax(logits, dim=1)
        self.test_acc(preds, y)
        self.log("test_acc", self.test_acc, prog_bar=True, sync_dist=True)  # let the hook determine the mode
        self.log("test_loss", loss, prog_bar=True, sync_dist=True)
        return loss
    

    def configure_optimizers(self):
        optimizer = self.optimizer
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.lr_step, gamma=self.lr_gam)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
            },
        }
    

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x, _ = batch  # see predict.py, was running into an Attribute no Flatten error [resolved]
        logits = self(x, torch.empty(x.size(dim=0)), dropout_on=False)  # SUPER HACKY, CHECK THIS
        preds = torch.argmax(logits, dim=1)
        return preds


# FCN experimentation
class ImprovedFCN(pl.LightningModule):    
    def __init__(self, input_size, num_classes, hidden_layers, activation, lr, lr_step, lr_gam, penalty, dropout=0, l2=0):
        super().__init__()
        self.save_hyperparameters()

        # PL attributes:
        self.train_acc = Accuracy(task='multiclass', num_classes=13)
        self.valid_acc = Accuracy(task='multiclass', num_classes=13)
        self.test_acc = Accuracy(task='multiclass', num_classes=13)

        # params
        self.input_size = input_size
        self.num_classes = num_classes
        
        self.hidden_layers = hidden_layers
        self.activation = utils.get_activation(activation)
        # old dropout
        # self.dropout = dropout
        # if dropout is not None: self.dropout_layer = nn.Dropout(dropout)
        # self.dropout = dropout
        # new dropout
        self.dropout_layer = nn.Dropout(dropout)

        self.lr = lr
        self.lr_step = lr_step
        self.lr_gam = lr_gam
        self.l2 = l2

        self.penalty = penalty
        self.optimizer_pointer = None
        self.start_time = time.time()

        # # FCN model
        # model_layers = [nn.Flatten()]
        # for num in self.hidden_layers:
        #     layer = nn.Linear(input_size, num)
        #     model_layers.append(layer)
        #     model_layers.append(self.activation)
        #     input_size = num
        # self.model = nn.Sequential(*model_layers)
        # self.output_layer = nn.Linear(self.hidden_layers[-1], num_classes)

        # # FCN model
        # self.model = self.make_model()  # no dropout, for val/test
        # self.dropout_model = self.make_model(dropout_on=True)  # dropout if turned on; if off, identical to self.model
        # self.output_layer = nn.Linear(self.hidden_layers[-1], num_classes)

        # # FCN model
        # self.output_layer = nn.Linear(self.hidden_layers[-1], num_classes)
        # # regular model
        # in_layer = input_size
        # model_layers = [nn.Flatten()]
        # for num in self.hidden_layers:
        #     layer = nn.Linear(in_layer, num)
        #     model_layers.append(layer)
        #     model_layers.append(self.activation)
        #     in_layer = num
        # self.model = nn.Sequential(*model_layers)
        # # self.dropout_model = self.model

        # # dropout model
        # in_layer = input_size
        # model_layers = [nn.Flatten()]
        # for num in self.hidden_layers:
        #     layer = nn.Linear(in_layer, num)
        #     model_layers.append(layer)
        #     model_layers.append(self.activation)
        #     in_layer = num
        # self.dropout_model = nn.Sequential(*model_layers)

        # FCN model
        self.output_layer = nn.Linear(self.hidden_layers[-1], num_classes)
        # regular model
        in_layer = input_size
        model_layers = [nn.Flatten()]
        for num in self.hidden_layers:
            layer = nn.Linear(in_layer, num)
            model_layers.append(layer)
            model_layers.append(self.activation)
            model_layers.append(self.dropout_layer)
            in_layer = num
        self.model = nn.Sequential(*model_layers)


    # def make_model(self, dropout_on=False):
        # input_size = self.input_size
        # model_layers = [nn.Flatten()]
        # for num in self.hidden_layers:
        #     layer = nn.Linear(input_size, num)
        #     model_layers.append(layer)
        #     model_layers.append(self.activation)
        #     # dropout
        #     if self.dropout is not None and dropout_on:
        #         model_layers.append(self.dropout_layer)
        #     input_size = num
        # return nn.Sequential(*model_layers)


    def forward(self, x):  # , dropout_on=True
        x = self.model(x)  
        x = self.output_layer(x)
        return x
        # x = self.model(x)    
        # # dropout
        # if self.dropout is not None and dropout_on:
        #     x = self.dropout_layer(x)
        # x = self.output_layer(x)
        # return x
    

    def loss_fn(self, x, y):
        # return (nn.functional.cross_entropy(X, y)) + self.misclassB(X, y)
        cross_entropy = nn.functional.cross_entropy(self(x), y) 

        arr = torch.argmax(self(x), dim=1)
        encoded_arr = torch.zeros((arr.size(dim=0), 13), dtype=torch.int).to(x.device)
        encoded_arr[torch.arange(arr.size(dim=0)), arr] = 1
        pair_wise_x = encoded_arr

        #pair_wise_x = nn.Softmax(dim=1)(self(x))  # def figure out above softmax mode, not argmax but hard mode

        pair_wise = (1 + self.loss_penalty(pair_wise_x, y)) ** self.penalty  # put in actual predictions for custom loss
        return cross_entropy * pair_wise


    # give incorrect predictions a logarithmic divergence
    # (note, it would be better to work with log-probalities than to use clamp)
    def misclassB (self, pred, targ):   # pred[nBatch, nClass], targ[nBatch], D[nClass, nClass]
        D = [ [1, 2, 2, 2, 2, 2, 8, 8, 8, 8, 8, 8, 20],
        [2, 1, 2, 2, 2, 2, 8, 8, 8, 8, 8, 8, 20],
        [2, 2, 1, 2, 2, 2, 8, 8, 8, 8, 8, 8, 20],
        [2, 2, 2, 1, 2, 2, 8, 8, 8, 8, 8, 8, 20],
        [2, 2, 2, 2, 1, 2, 8, 8, 8, 8, 8, 8, 20],
        [2, 2, 2, 2, 2, 1, 8, 8, 8, 8, 8, 8, 20],
        [8, 8, 8, 8, 8, 8, 1, 2, 2, 2, 2, 2, 20], 
        [8, 8, 8, 8, 8, 8, 2, 1, 2, 2, 2, 2, 20],
        [8, 8, 8, 8, 8, 8, 2, 2, 1, 2, 2, 2, 20],
        [8, 8, 8, 8, 8, 8, 2, 2, 2, 1, 2, 2, 20],
        [8, 8, 8, 8, 8, 8, 2, 2, 2, 2, 1, 2, 20],
        [8, 8, 8, 8, 8, 8, 2, 2, 2, 2, 2, 1, 20],
        [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 1] ]
        D = torch.from_numpy(np.asarray(D).astype(np.float32)).to(pred.device)
        D -= D * torch.eye(13).to(pred.device)
        return  (D[targ] * (-(pred - 1.0).clamp (min = 1.e-7).log())).mean()
    

    # experimenting with loss penalty
    def loss_penalty (self, pred, targ): 
        D = [ [1, 2, 2, 2, 2, 2, 8, 8, 8, 8, 8, 8, 20],
        [2, 1, 2, 2, 2, 2, 8, 8, 8, 8, 8, 8, 20],
        [2, 2, 1, 2, 2, 2, 8, 8, 8, 8, 8, 8, 20],
        [2, 2, 2, 1, 2, 2, 8, 8, 8, 8, 8, 8, 20],
        [2, 2, 2, 2, 1, 2, 8, 8, 8, 8, 8, 8, 20],
        [2, 2, 2, 2, 2, 1, 8, 8, 8, 8, 8, 8, 20],
        [8, 8, 8, 8, 8, 8, 1, 2, 2, 2, 2, 2, 20], 
        [8, 8, 8, 8, 8, 8, 2, 1, 2, 2, 2, 2, 20],
        [8, 8, 8, 8, 8, 8, 2, 2, 1, 2, 2, 2, 20],
        [8, 8, 8, 8, 8, 8, 2, 2, 2, 1, 2, 2, 20],
        [8, 8, 8, 8, 8, 8, 2, 2, 2, 2, 1, 2, 20],
        [8, 8, 8, 8, 8, 8, 2, 2, 2, 2, 2, 1, 20],
        [20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 1] ]
        D = torch.from_numpy(np.asarray(D).astype(np.float32)).to(pred.device)
        D -= D * torch.eye(13).to(pred.device)
        # return  ((D[targ] * (-(pred - 1.0).clamp (min = 1.e-7).log())).mean()) ** self.penalty
        # print("num: ", (D[targ] * pred).mean())
        return  (D[targ] * pred).mean()


    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(x, y)
        preds = torch.argmax(logits, dim=1)
        self.train_acc(preds, y)
        self.log('train_acc', self.train_acc, on_step=True, on_epoch=True)
        self.log("train_loss", loss, prog_bar=True)
        return loss
    
  
    def on_train_epoch_end(self):
        self.log('lr', self.optimizer_pointer.state_dict()['param_groups'][0]['lr'], sync_dist=True)  # hacky, possibly damaging
        self.log("time", time.time() - self.start_time, sync_dist=True)


    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)  # , dropout_on=False
        loss = self.loss_fn(x, y)
        preds = torch.argmax(logits, dim=1)
        self.valid_acc(preds, y)
        self.log("valid_acc", self.valid_acc, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("valid_loss", loss, prog_bar=True, sync_dist=True)
        return loss
    

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)  #, dropout_on=False
        loss = self.loss_fn(x, y)
        preds = torch.argmax(logits, dim=1)
        self.test_acc(preds, y)
        self.log("test_acc", self.test_acc, prog_bar=True, sync_dist=True)  # let the hook determine the mode
        self.log("test_loss", loss, prog_bar=True, sync_dist=True)
        return loss
    

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.l2)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.lr_step, gamma=self.lr_gam)
        self.optimizer_pointer = optimizer
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
            },
        }
    

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x, _ = batch  # see predict.py, was running into an Attribute no Flatten error [resolved]
        logits = self(x)  # , dropout_on=False
        preds = torch.argmax(logits, dim=1)
        return preds
    

#deep FCN experimentation
class DeepImprovedFCN(pl.LightningModule):    
    def __init__(self, input_size, num_classes, num_layers, hidden_layers, lr, lr_step, lr_gam, activation, penalty, dropout=None):
        super().__init__()
        self.save_hyperparameters()

        # PL attributes:
        self.train_acc = Accuracy(task='multiclass', num_classes=13)
        self.valid_acc = Accuracy(task='multiclass', num_classes=13)
        self.test_acc = Accuracy(task='multiclass', num_classes=13)

        # params
        self.num_classes = num_classes
        self.num_layers = num_layers
        #self.hidden_layers = hidden_layers
        self.neurons_per_layer = hidden_layers
        self.lr = lr
        self.lr_step = lr_step
        self.lr_gam = lr_gam
        self.penalty = penalty
        self.dropout = dropout
        if dropout is not None: self.dropout_layer = nn.Dropout(dropout)
        self.activation = utils.get_activation(activation)

        # misc attributes
        self.start_time = time.time()
        self.optimizer_pointer = None

        #deep fcn architecture allows for blocks with arbitrary hidden unit neuron sizes between blocks, the number of hidden layers
        #aka the length of hidden layers array parameter  must divide the total number of hidden layers num_layers
        self.num_blocks = int(self.num_layers / len(self.neurons_per_layer))
        #################### defining model layers ###################################

        list_blocks = [nn.Flatten()]
        input = input_size
        #we have 
        for i in range(self.num_blocks):
            """IMPORTANt note on Architecture design, every block of hidden layers is designed to be identical
             since they are each using the same neurons_per_layer list of neuron dims
             i.e if neurons_per_layer = [1024,512,256,128,64], every block of hidden layers will
             have these same neuron dims, this is a naive choice of architecture, and it may be better to 
             have each hidden block have a different number of neurons per hidden layer
            """

            #for every block but the last block we design it so that the output layer of the last layer
            #of the block is the same as the input for next hidden block
            if i < self.num_blocks - 1:        
                first_size = self.neurons_per_layer[0]
                block_hidden, next_input = self.make_block(self.neurons_per_layer, input, first_size)

                #append both to the ModuleLists
                list_blocks.append(block_hidden)

                #the input size of the next block is the next output size of the previous block, so we update it here
                #if this is a middle block, that means it is self.neurons_per_layer[0], 
                input = next_input
            else:
                #for the last block we make sure we feed output dim as output dim of the last layer of last block 
                # use make block function to return a normal block of hidden layers as an (nn.Sequential obj)
                block_hidden, next_input = self.make_block(self.neurons_per_layer, input, self.num_classes)
                list_blocks.append(block_hidden)
        self.model = nn.Sequential(*list_blocks)


    #this method makes creates nn.Sequential objects, one of normal hidden layers with activation
    def make_block(self, neurons_per_layer, input, output):
         ####### Defining layers for Model ####
        normal_layers = [nn.Flatten()]
        #projection = [nn.Flatten()]
     
        ####loop to append linear layers
        inp = input
        # Dynamic hidden ==layers
        for num_neurons in neurons_per_layer:
            #each layer has is a matrix in R^[input x width]
            normal_layers.append(nn.Linear(inp, num_neurons))
            #projection.append(nn.Linear(inp, num_neurons))

            #normal_layers.append(nn.BatchNorm1d(num_neurons)):
            activator = self.activation
            normal_layers.append(activator)
            #append a dropout layer after each linear layer (see barajas thesis p36, i think was done this way)
            normal_layers.append(nn.Dropout(self.dropout))
            #projection.append(nn.Dropout(self.dropout_rate))
            #input for next layer is number of neurons in current layeer
            inp = num_neurons

        ###### Output layer to next block ####

        # Note that the outdim of the last layer is the same 
        normal_layers.append(nn.Linear(neurons_per_layer[-1] , output))
        #projection.append(nn.Linear(neurons_per_layer[-1] , output))

        block_layers =  nn.Sequential(*normal_layers) 
        #projection block is our skip or shorcut connection of input that are passed through the layers without any activation function
        #projection_block = nn.Sequential(*projection) 

        # pass output as a parameter so that output layer of this block of neurons can
        # connect to the next input layer of the next block of hidden neurons
        return block_layers, output


    def forward(self, x):
        x = self.model(x)
        return x


    # give incorrect predictions a logarithmic divergence
    # (note, it would be better to work with log-probalities than to use clamp)
    def misclassB (self, pred, targ):   # pred[nBatch, nClass], targ[nBatch], D[nClass, nClass]
        D = [ [1, 2, 2, 2, 2, 2, 8, 8, 8, 8, 8, 8, 20],
        [2, 1, 2, 2, 2, 2, 8, 8, 8, 8, 8, 8, 20],
        [2, 2, 1, 2, 2, 2, 8, 8, 8, 8, 8, 8, 20],
        [2, 2, 2, 1, 2, 2, 8, 8, 8, 8, 8, 8, 20],
        [2, 2, 2, 2, 1, 2, 8, 8, 8, 8, 8, 8, 20],
        [2, 2, 2, 2, 2, 1, 8, 8, 8, 8, 8, 8, 20],
        [8, 8, 8, 8, 8, 8, 1, 2, 2, 2, 2, 2, 20], 
        [8, 8, 8, 8, 8, 8, 2, 1, 2, 2, 2, 2, 20],
        [8, 8, 8, 8, 8, 8, 2, 2, 1, 2, 2, 2, 20],
        [8, 8, 8, 8, 8, 8, 2, 2, 2, 1, 2, 2, 20],
        [8, 8, 8, 8, 8, 8, 2, 2, 2, 2, 1, 2, 20],
        [8, 8, 8, 8, 8, 8, 2, 2, 2, 2, 2, 1, 20],
        [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 1] ]
        D = torch.from_numpy(np.asarray(D).astype(np.float32)).to(pred.device)
        D -= D * torch.eye(13).to(pred.device)
        return  ((D[targ] * (-(pred - 1.0).clamp (min = 1.e-7).log())).mean())  * self.penalty


    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = (nn.functional.cross_entropy(self(x), y)) + self.misclassB(self(x), y)
        preds = torch.argmax(logits, dim=1)
        self.train_acc.update(preds, y)
        self.log("train_loss", loss, prog_bar=True)
        return loss
    

    def on_train_epoch_end(self):
        self.log("train_acc", self.train_acc.compute(), sync_dist=True)
        self.log('lr', self.optimizer_pointer.state_dict()['param_groups'][0]['lr'], sync_dist=True)  # hacky, possibly damaging
        self.log("time", time.time() - self.start_time, sync_dist=True)


    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = (nn.functional.cross_entropy(self(x), y)) + self.misclassB(self(x), y)
        preds = torch.argmax(logits, dim=1)
        self.valid_acc.update(preds, y)
        self.log("valid_loss", loss, prog_bar=True, sync_dist=True)
        self.log("valid_acc", self.valid_acc.compute(), prog_bar=True, sync_dist=True)
        return loss
    

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = (nn.functional.cross_entropy(self(x), y)) + self.misclassB(self(x), y)
        preds = torch.argmax(logits, dim=1)
        self.test_acc.update(preds, y)
        self.log("test_loss", loss, prog_bar=True, sync_dist=True)
        self.log("test_acc", self.test_acc.compute(), prog_bar=True, sync_dist=True)
        return loss
    

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=450, gamma=0.95)
        self.optimizer_pointer = optimizer
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
            },
        }
    

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x, _ = batch  # see predict.py, was running into an Attribute no Flatten error [resolved]
        logits = self(x)
        preds = torch.argmax(logits, dim=1)
        return preds
    

#lstm 2023 with custom loss function inputs 
class ImprovedLSTM2023(pl.LightningModule):
    def __init__(self, indim, outdim, num_layers, neurons, lr, lr_step, lr_gam, dropout, activation, penalty, optimizer):
        super().__init__()
        self.save_hyperparameters()

        # PL attributes:
        self.train_acc = Accuracy(task='multiclass', num_classes=13)
        self.valid_acc = Accuracy(task='multiclass', num_classes=13)
        self.test_acc = Accuracy(task='multiclass', num_classes=13)

        self.indim = indim
        self.outdim = outdim
        self.num_layers = num_layers
        self.neurons = neurons
        self.lr = lr
        self.lr_step = lr_step
        self.lr_gam = lr_gam

        self.penalty = penalty
        self.input_layer = nn.Linear(self.indim, self.neurons)
        self.lstm = nn.LSTM(input_size=self.neurons, hidden_size=self.neurons, num_layers=4, batch_first=True)
        self.rnn_layers = self.get_layers()  # dense layers
        self.dropout = nn.Dropout(dropout)
        self.activation = utils.get_activation(activation)
        self.output_layer = nn.Linear(self.neurons, self.outdim)

        self.optimizer = utils.get_optimizer(optimizer)(self.parameters(), lr=lr)
        self.start_time = time.time()


    def forward(self, x, tgt, dropout_on=True):
        # x, tgt = batch
        x = x.unsqueeze(1)  # adds a sequence length of 1
        x = self.input_layer(x)
        
        # Initializing hidden state for first input with zeros
        h0 = torch.zeros(4, x.size(0), self.neurons).to(x.device).requires_grad_()
        # Initializing cell state for first input with zeros
        c0 = torch.zeros(4, x.size(0), self.neurons).to(x.device).requires_grad_()

        # We need to detach as we are doing truncated backpropagation through time (BPTT)
        # If we don't, we'll backprop all the way to the start even after going through another batch
        # Forward propagation by passing in the input, hidden state, and cell state into the model
        x, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))

        # Reshaping the outputs in the shape of (batch_size, seq_length, hidden_size)
        # so that it can fit into the fully connected layer
        x = x[:, -1, :]

        # RNN layer
        for layer in self.rnn_layers: 
            if isinstance(layer, nn.Linear):
                x = layer(x)
            else:
                x, tgt = layer(x, tgt)
        
        
        x = self.activation(x) #activation function after fully connected layer

        # dropout -- CHANGED ON 7/19 moved to after activation by mc
        if dropout_on:
            x = self.dropout(x) # only want the final timestep output , if it's batch_first

        x = self.output_layer(x)

        return x


    def get_layers(self):
        layers = nn.ModuleList()
        layer = nn.Linear(self.neurons, self.neurons, bias=True)
        for _ in range(self.num_layers - 1):
            layers.append(layer)
        layers.append(layer)  # this is clearly wrong but it's how they did the run
        return layers
    

    # give incorrect predictions a logarithmic divergence
    # (note, it would be better to work with log-probalities than to use clamp)
    def misclassB (self, pred, targ):   # pred[nBatch, nClass], targ[nBatch], D[nClass, nClass]
        D = [ [1, 2, 2, 2, 2, 2, 8, 8, 8, 8, 8, 8, 20],
        [2, 1, 2, 2, 2, 2, 8, 8, 8, 8, 8, 8, 20],
        [2, 2, 1, 2, 2, 2, 8, 8, 8, 8, 8, 8, 20],
        [2, 2, 2, 1, 2, 2, 8, 8, 8, 8, 8, 8, 20],
        [2, 2, 2, 2, 1, 2, 8, 8, 8, 8, 8, 8, 20],
        [2, 2, 2, 2, 2, 1, 8, 8, 8, 8, 8, 8, 20],
        [8, 8, 8, 8, 8, 8, 1, 2, 2, 2, 2, 2, 20], 
        [8, 8, 8, 8, 8, 8, 2, 1, 2, 2, 2, 2, 20],
        [8, 8, 8, 8, 8, 8, 2, 2, 1, 2, 2, 2, 20],
        [8, 8, 8, 8, 8, 8, 2, 2, 2, 1, 2, 2, 20],
        [8, 8, 8, 8, 8, 8, 2, 2, 2, 2, 1, 2, 20],
        [8, 8, 8, 8, 8, 8, 2, 2, 2, 2, 2, 1, 20],
        [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 1] ]
        D = torch.from_numpy(np.asarray(D).astype(np.float32)).to(pred.device)
        D -= D * torch.eye(13).to(pred.device)
        return  ((D[targ] * (-(pred - 1.0).clamp (min = 1.e-7).log())).mean())  * self.penalty


    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x,y)
        loss = (nn.functional.cross_entropy(self(x,y), y)) + self.misclassB(self(x,y), y)
        preds = torch.argmax(logits, dim=1)
        self.train_acc.update(preds, y)
        self.log("train_loss", loss, prog_bar=True)
        return loss
    

    def on_train_epoch_end(self):
        self.log("train_acc", self.train_acc.compute(), sync_dist=True)
        self.log('lr', self.optimizer_pointer.state_dict()['param_groups'][0]['lr'], sync_dist=True)  # hacky, possibly damaging
        self.log("time", time.time() - self.start_time, sync_dist=True)


    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x,y, dropout_on=False)
        loss = (nn.functional.cross_entropy(self(x,y), y)) + self.misclassB(self(x,y), y)
        preds = torch.argmax(logits, dim=1)
        self.valid_acc.update(preds, y)
        self.log("valid_loss", loss, prog_bar=True, sync_dist=True)
        self.log("valid_acc", self.valid_acc.compute(), prog_bar=True, sync_dist=True)
        return loss
    

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x,y, dropout_on=False)
        loss = (nn.functional.cross_entropy(self(x,y), y)) + self.misclassB(self(x,y), y)
        preds = torch.argmax(logits, dim=1)
        self.test_acc.update(preds, y)
        self.log("test_loss", loss, prog_bar=True, sync_dist=True)
        self.log("test_acc", self.test_acc.compute(), prog_bar=True, sync_dist=True)
        return loss
    

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=450, gamma=0.95)
        self.optimizer_pointer = optimizer
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
            },
        }
    

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x, _ = batch  # see predict.py, was running into an Attribute no Flatten error [resolved]
        # x, _ = batch  # see predict.py, was running into an Attribute no Flatten error [resolved]
        logits = self(x, torch.empty(x.size(dim=0)), dropout_on=False)  # SUPER HACKY, CHECK THIS
        
        preds = torch.argmax(logits, dim=1)
        return preds
    

#lstm 2024 with custom loss function inputs and better model principles
class ImprovedLSTM2024(pl.LightningModule):
    def __init__(self, 
                indim, outdim, 
                num_linears, neurons_per_hidden,
                input_neurons,  num_lstm_layers, hidden_state_size,
                lr, lr_step, lr_gam, 
                dropout, activation, 
                penalty, optimizer, l2=0.0, one_activation=False,
                custom_loss=True, bias=True):
        super().__init__()
        self.save_hyperparameters()

        # PL attributes:
        self.train_acc = Accuracy(task='multiclass', num_classes=13)
        self.valid_acc = Accuracy(task='multiclass', num_classes=13)
        self.test_acc = Accuracy(task='multiclass', num_classes=13)
        self.validation_step_outputs = []

        #architecture attrs
        self.indim = indim
        self.outdim = outdim
        self.num_linears = num_linears
        self.num_blocks = int(num_linears/len(neurons_per_hidden))
        #neurons is the number of neurons of input linear layer to lstm layers
        self.input_neurons = input_neurons
        self.lr = lr
        self.lr_step = lr_step
        self.lr_gam = lr_gam
        self.l2 = l2
        self.penalty = penalty
        self.neurons_per_hidden = neurons_per_hidden
        self.hidden_state_size = hidden_state_size

        self.one_activation = one_activation
        #bias is a boolean flag that will turn bias on for linear layers, this should really be initialized as a parameter
        self.bias = bias
        self.activation = utils.get_activation(activation)
        self.dropout = nn.Dropout(dropout)
        #defining model layers
        self.input_layer = nn.Linear(self.indim, self.input_neurons)
        self.lstm = nn.LSTM(input_size=self.input_neurons, hidden_size=self.hidden_state_size, num_layers=num_lstm_layers, batch_first=True) #lstm layers
        self.fc_layers = self.get_layers(self.input_neurons)  # dense layers
        
        self.output_layer #= nn.Linear(self.neurons, self.outdim) I am leaving this uninit here and init in get_layers

        self.optimizer = utils.get_optimizer(optimizer)(self.parameters(), lr=lr, weight_decay=self.l2)
        self.start_time = time.time()

        #custom loss is a boolean flag that will use custom loss in all steps, by default is on
        self.custom_loss = custom_loss


    def forward(self, x):
        # x, tgt = batch
        x = x.unsqueeze(1)  # adds a sequence length of 1
        x = self.input_layer(x)
        
        # Initializing hidden state for first input with zeros
        h0 = torch.zeros(4, x.size(0), self.hidden_state_size).to(x.device).requires_grad_()
        # Initializing cell state for first input with zeros
        c0 = torch.zeros(4, x.size(0), self.hidden_state_size).to(x.device).requires_grad_()

        # We need to detach as we are doing truncated backpropagation through time (BPTT)
        # If we don't, we'll backprop all the way to the start even after going through another batch
        # Forward propagation by passing in the input, hidden state, and cell state into the model
        x, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))

        # Reshaping the outputs in the shape of (batch_size, seq_length, hidden_size)
        # so that it can fit into the fully connected layer
        x = x[:, -1, :]

        #note, for loops in the forward function are veryy bad for speed due to complexity reasons,
        #during each forward call, and there are 100s of thousands during backprop, this for loop
        #will be unpacked in the call stack, versus, it going through the sequential fc_layers object
        #which has its own wrapped forward call
        """
        # fc layers
        for layer in self.fc_layers: 
            #if isinstance(layer, nn.Linear):
            x = layer(x)
            x = self.activation(x)
            # dropout -am adding dropout after activation function for each linear layer
            if dropout_on:
                x = self.dropout(x) # only want the final timestep output , if it's batch_first

            #else:
            #    x, tgt = layer(x, tgt)
        """
        x = self.fc_layers(x)
        x = self.output_layer(x)

        return x


    def get_layers(self, input):
        #we are flattening all the linear layers to pass them into a nn.Sequential obj,
        layers = [nn.Flatten()]
        

        #input is the input size of the number of neurons in the (last) layer before this method is called
        indim = input
        #there are num_layers total hidden layers, we expect this to divide the neurons_per_hidden array of neurons
        for i in range(self.num_blocks):
            #IMPORTANT ARCHITECTURE DESIGN note, if for example, neurons_per_layer = [256,128,64,32], num_linears = 8
            #then the architecture of the hidden layers should be 
            # [Linear(256,128), Linear(128,64), Linear(64,32), 
            #  Linear(32,256) <- where this last layer feeds into the next 'block' of hidden layers
            #  Linear(256,128), Linear(128,64), Linear(64,32), Linear(32,num_classes) <-output layer ]
            #iterate through neurons_per_layer and create connections of hidden layers with
            #specified dimensions
            for num in self.neurons_per_hidden:
                #outdim of current layer is next item in neurons_per_hidden
                outdim = num
                #also intializing with bias
                layer = nn.Linear(indim, outdim, bias=self.bias)
                 
                layers.append(layer)

                # dropout -am adding dropout after activation function for each linear layer, if you dont want dropout, 
                # set it to zero in params, internally to pytorch this should just be identity
                layers.append(self.dropout)
                if self.one_activation == False:
                    layers.append(self.activation)
                #layers.append(self.activation)
                 
                #reinitialze indim of next hidden layer as outdim of current layer
                indim = outdim
        #using this flag attribute to test whether having the activation function between the layers
        #or after the layers is best
        if self.one_activation == True:
            layers.append(self.activation)
        #IMPORTANT, output layer is initialized here, im not going to pack it into the sequential object for arbitrary reasons
        self.output_layer = nn.Linear(outdim, self.outdim)
        # no activation function on output layer (if we werent using cross_entropy_loss, the outdim should have softmax activ after for probability inference
        hidden_layers = nn.Sequential(*layers)  
        return hidden_layers
    

    def loss_fn(self, x, y):
        # return (nn.functional.cross_entropy(X, y)) + self.misclassB(X, y)
        cross_entropy = nn.functional.cross_entropy(self(x), y) 

        arr = torch.argmax(self(x), dim=1)
        encoded_arr = torch.zeros((arr.size(dim=0), 13), dtype=torch.int).to(x.device)
        encoded_arr[torch.arange(arr.size(dim=0)), arr] = 1
        pair_wise_x = encoded_arr

        #pair_wise_x = nn.Softmax(dim=1)(self(x))  # def figure out above softmax mode, not argmax but hard mode

        pair_wise = (1 + self.loss_penalty(pair_wise_x, y)) ** self.penalty  # put in actual predictions for custom loss
        return cross_entropy * pair_wise


    # give incorrect predictions a logarithmic divergence
    # (note, it would be better to work with log-probalities than to use clamp)
    def misclassB (self, pred, targ):   # pred[nBatch, nClass], targ[nBatch], D[nClass, nClass]
        D = [ [1, 2, 2, 2, 2, 2, 10, 10, 10, 10, 10, 10, 20],
        [2, 1, 2, 2, 2, 2, 10, 10, 10, 10, 10, 10, 20],
        [2, 2, 1, 2, 2, 2, 10, 10, 10, 10, 10, 10, 20],
        [2, 2, 2, 1, 2, 2, 10, 10, 10, 10, 10, 10, 20],
        [2, 2, 2, 2, 1, 2, 10, 10, 10, 10, 10, 10, 20],
        [2, 2, 2, 2, 2, 1, 10, 10, 10, 10, 10, 10, 20],
        [10, 10, 10, 10, 10, 10, 1, 2, 2, 2, 2, 2, 20], 
        [10, 10, 10, 10, 10, 10, 2, 1, 2, 2, 2, 2, 20],
        [10, 10, 10, 10, 10, 10, 2, 2, 1, 2, 2, 2, 20],
        [10, 10, 10, 10, 10, 10, 2, 2, 2, 1, 2, 2, 20],
        [10, 10, 10, 10, 10, 10, 2, 2, 2, 2, 1, 2, 20],
        [10, 10, 10, 10, 10, 10, 2, 2, 2, 2, 2, 1, 20],
        [10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 1] ]
        D = torch.from_numpy(np.asarray(D).astype(np.float32)).to(pred.device)
        D -= D * torch.eye(13).to(pred.device)
        return  ((D[targ] * (-(pred - 1.0).clamp (min = 1.e-7).log())).mean()) * self.penalty
    

    # experimenting with loss penalty
    def loss_penalty (self, pred, targ): 
        D = [ [1, 2, 2, 2, 2, 2, 10, 10, 10, 10, 10, 10, 20],
        [2, 1, 2, 2, 2, 2, 10, 10, 10, 10, 10, 10, 20],
        [2, 2, 1, 2, 2, 2, 10, 10, 10, 10, 10, 10, 20],
        [2, 2, 2, 1, 2, 2, 10, 10, 10, 10, 10, 10, 20],
        [2, 2, 2, 2, 1, 2, 10, 10, 10, 10, 10, 10, 20],
        [2, 2, 2, 2, 2, 1, 10, 10, 10, 10, 10, 10, 20],
        [10, 10, 10, 10, 10, 10, 1, 8, 2, 2, 2, 2, 20], 
        [10, 10, 10, 10, 10, 10, 10, 1, 8, 2, 2, 2, 20],
        [10, 10, 10, 10, 10, 8, 2, 8, 1, 8, 2, 2, 20],
        [10, 10, 10, 10, 10, 10, 2, 2, 8, 1, 8, 2, 20],
        [10, 10, 10, 10, 10, 10, 2, 2, 2, 8, 1, 8, 20],
        [10, 10, 10, 10, 10, 10, 2, 2, 2, 2, 8, 1, 20],
        [20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 1] ]
        D = torch.from_numpy(np.asarray(D).astype(np.float32)).to(pred.device)
        D -= D * torch.eye(13).to(pred.device)
        # return  ((D[targ] * (-(pred - 1.0).clamp (min = 1.e-7).log())).mean()) ** self.penalty
        # print("num: ", (D[targ] * pred).mean())
        return  (D[targ] * pred).mean()


    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        if self.custom_loss:
            loss = self.loss_fn(x, y)
        else:
            loss = (nn.functional.cross_entropy(self(x), y))
        preds = torch.argmax(logits, dim=1)
        self.train_acc(preds, y)
        self.log('train_acc', self.train_acc, on_step=True, on_epoch=True)
        self.log("train_loss", loss, prog_bar=True)
        return loss


    def on_train_epoch_end(self):
        self.log('lr', self.optimizer_pointer.state_dict()['param_groups'][0]['lr'], sync_dist=True)  # hacky, possibly damaging
        self.log("time", time.time() - self.start_time, sync_dist=True)


    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        if self.custom_loss:
            loss = self.loss_fn(x, y)
        else:
            loss = (nn.functional.cross_entropy(self(x), y))
        preds = torch.argmax(logits, dim=1)
        self.valid_acc(preds, y)
        self.log("valid_acc", self.valid_acc, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("valid_loss", loss, prog_bar=True, sync_dist=True)
        #self.log("valid_acc", self.valid_acc.compute(), prog_bar=True, sync_dist=True)
        #self.validation_step_outputs.append(loss)
        return loss
    

    # mc commented this to bring it up to date with rest of it
    # def on_validation_epoch_end(self):  
    #     #self.log('valid_acc_epoch', self.valid_acc.compute())
    #     self.log("valid_acc", self.valid_acc.compute(), prog_bar=True, sync_dist=True)
    #     self.valid_acc.reset()


    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        if self.custom_loss:
            loss = self.loss_fn(x, y)
        else:
            loss = (nn.functional.cross_entropy(self(x), y))
        preds = torch.argmax(logits, dim=1)
        self.test_acc(preds, y)
        self.log("test_acc", self.test_acc, prog_bar=True, sync_dist=True)  # let the hook determine the mode
        self.log("test_loss", loss, prog_bar=True, sync_dist=True)
        return loss
    

    def configure_optimizers(self):
        #optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        optimizer = self.optimizer
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=450, gamma=0.95)
        self.optimizer_pointer = optimizer
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
            },
        }
    

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x, _ = batch  # see predict.py, was running into an Attribute no Flatten error [resolved]
        # x, _ = batch  # see predict.py, was running into an Attribute no Flatten error [resolved]
        #logits = self(x, torch.empty(x.size(dim=0)), dropout_on=False)  # SUPER HACKY, CHECK THIS
        #x, y = batch
        logits = self(x)
        preds = torch.argmax(logits, dim=1)
        return preds


# control by deleting LSTM layers
class LSTM2023Control(pl.LightningModule):

    def __init__(self, indim, outdim, num_layers, neurons, lr, lr_step, lr_gam, dropout, activation, optimizer, l2=0, log_test=False):
        super().__init__()
        self.save_hyperparameters()

        # PL attributes:
        self.train_acc = Accuracy(task='multiclass', num_classes=13)
        self.valid_acc = Accuracy(task='multiclass', num_classes=13)
        self.test_acc = Accuracy(task='multiclass', num_classes=13)

        self.indim = indim
        self.outdim = outdim
        self.num_layers = num_layers
        self.neurons = neurons
        self.lr = lr
        self.lr_step = lr_step
        self.lr_gam = lr_gam
        self.l2=l2

        self.input_layer = nn.Linear(self.indim, self.neurons)

        model_layers = []
        input_size = self.neurons
        for num in [512, 512, 376, 256]:  # hacky af
            layer = nn.Linear(input_size, num)
            model_layers.append(layer)
            # model_layers.append(nn.Tanh())  # activation
            model_layers.append(nn.ReLU())  # for ctrl run mc7.27_LSTM23Cv2
            print("\n\n\n\n\n\n\n\n\n\n\n\nFLAG NOTE: tanh changed!\n\n\n\n\n\n\n\n\n\n\n\n")
            input_size = num
        self.lstm = nn.Sequential(*model_layers)
        #self.lstm = nn.LSTM(input_size=self.neurons, hidden_size=self.neurons, num_layers=4, batch_first=True)
        self.activation = utils.get_activation(activation)  # had to move this up 2 lines for the ctrl test, should have no effect
        self.rnn_layers = self.get_layers()  # dense layers
        self.dropout = nn.Dropout(dropout)
        self.output_layer = nn.Linear(self.neurons, self.outdim)

        self.optimizer = utils.get_optimizer(optimizer)(self.parameters(), lr=lr, weight_decay=self.l2)
        self.start_time = time.time()
        self.log_test = log_test


    def forward(self, x, tgt, dropout_on=True):
        # x, tgt = batch
        x = x.unsqueeze(1)  # adds a sequence length of 1
        x = self.input_layer(x)
        
        # Initializing hidden state for first input with zeros
        h0 = torch.zeros(4, x.size(0), self.neurons).to(x.device).requires_grad_()
        # Initializing cell state for first input with zeros
        c0 = torch.zeros(4, x.size(0), self.neurons).to(x.device).requires_grad_()

        # We need to detach as we are doing truncated backpropagation through time (BPTT)
        # If we don't, we'll backprop all the way to the start even after going through another batch
        # Forward propagation by passing in the input, hidden state, and cell state into the model
        # x, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        x = self.lstm(x)

        # Reshaping the outputs in the shape of (batch_size, seq_length, hidden_size)
        # so that it can fit into the fully connected layer
        x = x[:, -1, :]

        # RNN layer
        for layer in self.rnn_layers: 
            if isinstance(layer, nn.Linear) or isinstance(layer, nn.ReLU):  # hacky or added for ctrl run
                x = layer(x)
            else:
                x, tgt = layer(x, tgt)
        
        
        x = self.activation(x) #activation function after fully connected layer

        # dropout -- CHANGED ON 7/19 moved to after activation by mc
        if dropout_on:
            x = self.dropout(x) # only want the final timestep output , if it's batch_first

        x = self.output_layer(x)
    
        return x


    def get_layers(self):
        layers = nn.ModuleList()
        layer = nn.Linear(self.neurons, self.neurons, bias=True)
        for _ in range(self.num_layers - 1):
            layers.append(layer)
            # layers.append(self.activation)  # this line and below is for a control run 7.27_LSTM23Cv1
        layers.append(layer)  # this is clearly wrong but it's how they did the run
        # layers.append(self.activation)  # this line
        # print("\n\n\n\n\n\n\n\n\n\n\n\nFLAG NOTE: activations added!\n\n\n\n\n\n\n\n\n\n\n\n")
        return layers


    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x, y)
        loss = nn.functional.cross_entropy(self(x, y), y.view(-1))
        preds = torch.argmax(logits, dim=1)
        self.train_acc(preds, y)
        self.log('train_acc', self.train_acc, on_step=True, on_epoch=True)
        self.log("train_loss", loss, prog_bar=True)
        return loss
    

    def on_train_epoch_end(self):
        self.log('lr', self.optimizer.state_dict()['param_groups'][0]['lr'], sync_dist=True)  # hacky, possibly damaging
        self.log("time", time.time() - self.start_time, sync_dist=True)


    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x, y, dropout_on=False)
        loss = nn.functional.cross_entropy(self(x, y), y.view(-1))
        preds = torch.argmax(logits, dim=1)
        self.valid_acc(preds, y)
        self.log("valid_acc", self.valid_acc, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("valid_loss", loss, prog_bar=True, sync_dist=True)
        return loss
    
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x, y, dropout_on=False)
        loss = nn.functional.cross_entropy(self(x, y), y.view(-1))
        preds = torch.argmax(logits, dim=1)
        self.test_acc(preds, y)
        self.log("test_acc", self.test_acc, prog_bar=True, sync_dist=True)  # let the hook determine the mode
        self.log("test_loss", loss, prog_bar=True, sync_dist=True)
        return loss
    

    def configure_optimizers(self):
        optimizer = self.optimizer
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.lr_step, gamma=self.lr_gam)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
            },
        }
    

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x, _ = batch  # see predict.py, was running into an Attribute no Flatten error [resolved]
        logits = self(x, torch.empty(x.size(dim=0)), dropout_on=False)  # SUPER HACKY, CHECK THIS
        preds = torch.argmax(logits, dim=1)
        return preds


# 1D CNN experimentation
class CNN1D(pl.LightningModule):    
    def __init__(self, input_size, num_classes, hidden_layers, activation, lr, lr_step, lr_gam, penalty, dropout=None, l2=0):
        super().__init__()
        self.save_hyperparameters()

        # PL attributes:
        self.train_acc = Accuracy(task='multiclass', num_classes=13)
        self.valid_acc = Accuracy(task='multiclass', num_classes=13)
        self.test_acc = Accuracy(task='multiclass', num_classes=13)

        # params
        self.input_size = input_size
        self.num_classes = num_classes
        
        self.hidden_layers = hidden_layers
        self.activation = utils.get_activation(activation)
        self.dropout = dropout
        if dropout is not None: self.dropout_layer = nn.Dropout(dropout)

        self.lr = lr
        self.lr_step = lr_step
        self.lr_gam = lr_gam
        self.l2 = l2

        self.penalty = penalty
        self.optimizer_pointer = None
        self.start_time = time.time()

        # CNN
        model_layers = [nn.Conv1d(1, 10, 1, stride=5), 
                        nn.Conv1d(10, 50, 1, stride=5), 
                        nn.Conv1d(50, 250, 1, stride=5),]
        self.cnn_model = nn.Sequential(*model_layers)
        input_size = 250

        # FCN
        model_layers = [nn.Flatten()]
        for num in self.hidden_layers:
            layer = nn.Linear(input_size, num)
            model_layers.append(layer)
            model_layers.append(self.activation)
            input_size = num
        self.fcn_model = nn.Sequential(*model_layers)
        self.output_layer = nn.Linear(self.hidden_layers[-1], num_classes)


    def to_img(self, x):
        img_x = torch.unsqueeze(x, 1)
        return img_x
    

    def forward(self, x, dropout_on=True):
        img_x = self.to_img(x)
        x = self.cnn_model(img_x)

        x = self.fcn_model(x)
        # dropout
        if self.dropout is not None and dropout_on:
            x = self.dropout_layer(x)       
        x = self.output_layer(x)
        return x
    

    def loss_fn(self, x, y):
        # return (nn.functional.cross_entropy(X, y)) + self.misclassB(X, y)
        cross_entropy = nn.functional.cross_entropy(self(x), y) 

        arr = torch.argmax(self(x), dim=1)
        encoded_arr = torch.zeros((arr.size(dim=0), 13), dtype=torch.int).to(x.device)
        encoded_arr[torch.arange(arr.size(dim=0)), arr] = 1
        pair_wise_x = encoded_arr

        #pair_wise_x = nn.Softmax(dim=1)(self(x))  # def figure out above softmax mode, not argmax but hard mode

        pair_wise = (1 + self.loss_penalty(pair_wise_x, y)) ** self.penalty  # put in actual predictions for custom loss
        return cross_entropy * pair_wise


    # give incorrect predictions a logarithmic divergence
    # (note, it would be better to work with log-probalities than to use clamp)
    def misclassB (self, pred, targ):   # pred[nBatch, nClass], targ[nBatch], D[nClass, nClass]
        D = [ [1, 2, 2, 2, 2, 2, 8, 8, 8, 8, 8, 8, 20],
        [2, 1, 2, 2, 2, 2, 8, 8, 8, 8, 8, 8, 20],
        [2, 2, 1, 2, 2, 2, 8, 8, 8, 8, 8, 8, 20],
        [2, 2, 2, 1, 2, 2, 8, 8, 8, 8, 8, 8, 20],
        [2, 2, 2, 2, 1, 2, 8, 8, 8, 8, 8, 8, 20],
        [2, 2, 2, 2, 2, 1, 8, 8, 8, 8, 8, 8, 20],
        [8, 8, 8, 8, 8, 8, 1, 2, 2, 2, 2, 2, 20], 
        [8, 8, 8, 8, 8, 8, 2, 1, 2, 2, 2, 2, 20],
        [8, 8, 8, 8, 8, 8, 2, 2, 1, 2, 2, 2, 20],
        [8, 8, 8, 8, 8, 8, 2, 2, 2, 1, 2, 2, 20],
        [8, 8, 8, 8, 8, 8, 2, 2, 2, 2, 1, 2, 20],
        [8, 8, 8, 8, 8, 8, 2, 2, 2, 2, 2, 1, 20],
        [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 1] ]
        D = torch.from_numpy(np.asarray(D).astype(np.float32)).to(pred.device)
        D -= D * torch.eye(13).to(pred.device)
        return  ((D[targ] * (-(pred - 1.0).clamp (min = 1.e-7).log())).mean()) * self.penalty
    

    # experimenting with loss penalty
    def loss_penalty (self, pred, targ): 
        D = [ [1, 2, 2, 2, 2, 2, 8, 8, 8, 8, 8, 8, 20],
        [2, 1, 2, 2, 2, 2, 8, 8, 8, 8, 8, 8, 20],
        [2, 2, 1, 2, 2, 2, 8, 8, 8, 8, 8, 8, 20],
        [2, 2, 2, 1, 2, 2, 8, 8, 8, 8, 8, 8, 20],
        [2, 2, 2, 2, 1, 2, 8, 8, 8, 8, 8, 8, 20],
        [2, 2, 2, 2, 2, 1, 8, 8, 8, 8, 8, 8, 20],
        [8, 8, 8, 8, 8, 8, 1, 2, 2, 2, 2, 2, 20], 
        [8, 8, 8, 8, 8, 8, 2, 1, 2, 2, 2, 2, 20],
        [8, 8, 8, 8, 8, 8, 2, 2, 1, 2, 2, 2, 20],
        [8, 8, 8, 8, 8, 8, 2, 2, 2, 1, 2, 2, 20],
        [8, 8, 8, 8, 8, 8, 2, 2, 2, 2, 1, 2, 20],
        [8, 8, 8, 8, 8, 8, 2, 2, 2, 2, 2, 1, 20],
        [20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 1] ]
        D = torch.from_numpy(np.asarray(D).astype(np.float32)).to(pred.device)
        D -= D * torch.eye(13).to(pred.device)
        # return  ((D[targ] * (-(pred - 1.0).clamp (min = 1.e-7).log())).mean()) ** self.penalty
        # print("num: ", (D[targ] * pred).mean())
        return  (D[targ] * pred).mean()


    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(x, y)
        preds = torch.argmax(logits, dim=1)
        self.train_acc(preds, y)
        self.log('train_acc', self.train_acc, on_step=True, on_epoch=True)
        self.log("train_loss", loss, prog_bar=True)
        return loss
    

    def on_train_epoch_end(self):
        self.log('lr', self.optimizer_pointer.state_dict()['param_groups'][0]['lr'], sync_dist=True)  # hacky, possibly damaging
        self.log("time", time.time() - self.start_time, sync_dist=True)


    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x, dropout_on=False)
        loss = self.loss_fn(x, y)
        preds = torch.argmax(logits, dim=1)
        self.valid_acc(preds, y)
        self.log("valid_acc", self.valid_acc, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("valid_loss", loss, prog_bar=True, sync_dist=True)
        return loss
    

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x, dropout_on=False)
        loss = self.loss_fn(x, y)
        preds = torch.argmax(logits, dim=1)
        self.test_acc(preds, y)
        self.log("test_acc", self.test_acc, prog_bar=True, sync_dist=True)  # let the hook determine the mode
        self.log("test_loss", loss, prog_bar=True, sync_dist=True)
        return loss
    

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.l2)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.lr_step, gamma=self.lr_gam)
        self.optimizer_pointer = optimizer
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
            },
        }
    

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x, _ = batch  # see predict.py, was running into an Attribute no Flatten error [resolved]
        logits = self(x, dropout_on=False)
        preds = torch.argmax(logits, dim=1)
        return preds
 
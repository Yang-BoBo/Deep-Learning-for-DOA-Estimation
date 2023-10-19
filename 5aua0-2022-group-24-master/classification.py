#%%
#%%
import os
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import TensorDataset,random_split
import torch.nn.functional as F
import torchvision
import PIL
import matplotlib.pyplot as plt
import pandas as pd
import zipfile
from tqdm.auto import tqdm
import math
import torch.nn as nn
#%%
device = torch.device('cuda')
dtype = torch.float32
#%%
Full_dataset = torch.load('Complete_dataset.pt')
#%%
Train_set,Val_set = random_split(Full_dataset,[9175040,2293760])
#%%
batch_size = 2048
Train_loader = torch.utils.data.DataLoader(
       dataset=Train_set,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True)

Val_loader = torch.utils.data.DataLoader(
       dataset=Val_set,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True)




#%%
import torch.nn.functional as F
class Classification(nn.Module): 
    def __init__(self, n_channels=16,channel1=64,channel2=128,channel3=256,channel4=512, channel5=1024, channel6=2048, n_classes=14):
        super().__init__()
        #####################Encoder layers##############################
        self.encoder_1 = nn.Sequential(
            nn.Linear(n_channels, channel1),
            nn.BatchNorm1d(channel1),
            nn.ReLU(inplace = True),
        )

        self.encoder_2 = nn.Sequential(
            nn.Linear(channel1, channel1),
            nn.BatchNorm1d(channel1),
            nn.ReLU(inplace = True),
        )

        self.encoder_3 = nn.Sequential(
            nn.Linear(channel1, channel2),
            nn.BatchNorm1d(channel2),
            nn.ReLU(inplace = True),
        )
        self.encoder_4 = nn.Sequential(
            nn.Linear(channel2, channel3),
            nn.BatchNorm1d(channel3),
            nn.ReLU(inplace = True),
        )
        self.encoder_5 = nn.Sequential(
            nn.Linear(channel3, channel4),
            nn.BatchNorm1d(channel4),
            nn.ReLU(inplace = True),
        )
        self.encoder_6 = nn.Sequential(
            nn.Linear(channel4, channel5),
            nn.BatchNorm1d(channel5),
            nn.ReLU(inplace = True),
        )
        self.encoder_7 = nn.Sequential(
            nn.Linear(channel5, channel6),
            nn.BatchNorm1d(channel6),
            nn.ReLU(inplace = True),
        )
        self.encoder_8 = nn.Sequential(
            nn.Linear(channel6, channel3),
            nn.BatchNorm1d(channel3),
            nn.ReLU(inplace = True),
        )
        self.classifier = nn.Sequential(
            nn.Linear(channel3, 4),
        )
    def forward(self,x):
        
        x = self.encoder_1(x)

        x = self.encoder_2(x)

        x = self.encoder_3(x)

        x = self.encoder_4(x)

        x = self.encoder_5(x)

        x = self.encoder_6(x)

        x = self.encoder_7(x)

        x = self.encoder_8(x)

        number = self.classifier(x)
        
        return number
#%%
model = Classification()
weight_decay = 1e-6
optimizer = optim.Adam(model.parameters(), lr=0.00001, weight_decay=weight_decay)
model = model.to(device=device)  # move the model parameters to CPU/GPU
epochs = 10
#%%
model.load_state_dict(torch.load("20230622_classification_e70.pth"))
#%%
for e in range(epochs):
    for t, (x, y) in enumerate(tqdm(Train_loader)):
        x = x.to(device=device, dtype=dtype)  # move to device, e.g. GPU
        y = y.to(device=device, dtype=dtype)
        model.train()  # put model to training mode
        optimizer.zero_grad()

        num = model(x)

        _,classification = num.max(1)
        ind = y[:,-1]
        index = ind.to(torch.int64)
        loss_classification = F.cross_entropy(num.float(),index)
        # loss_regression = F.mse_loss(scores[:,ind:ind+N_source+1], y[:,0:N_source+1])
        # loss_regression = torch.sqrt(loss_regression)

        # Zero out all of the gradients for the variables which the optimizer
        # will update.

        # This is the backwards pass: compute the gradient of the loss with
        # respect to each  parameter of the model.
        # loss_regression.backward()
        # else:

        # optimizer.zero_grad()
        loss_classification.backward()
        # Actually update the parameters of the model using the gradients
        # computed by the backwards pass.
        optimizer.step()
        # print(f'prediction {doa1} \n truth {y[0]}')
        # print(f'classification {classification[0]+1}')
    if e%2 == 0:
        print('Epoch %d, Classification loss = %.4f' % (e, loss_classification.item()))
            # print()
            # print(scores[0,:])
            # print(y[0,:])
    with torch.no_grad():
        model.eval()
        for idx,(x,y) in enumerate(tqdm(Val_loader)):
                x = x.to(device=device, dtype=dtype)  # move to device, e.g. GPU
                y = y.to(device=device, dtype=dtype)
                scores = model(x)
                _,classification = scores.max(1)
                ind = y[:,-1]
                index = ind.to(torch.int64)
                loss_classification = F.cross_entropy(scores.float(),index)
                loss = loss_classification
        if e%2 == 0:
            print('Epoch %d, Classification loss = %.4f' % (e, loss_classification))
            print(f'Epoch {e}, Truth ={y[0,:]}')
            print(f'Epoch {e}, sources ={classification[0]}')

# %%

#%%
torch.save(model.state_dict(),'20230622_classification_e70.pth')
# %%
with torch.no_grad():
    model.eval()
    correct = 0
    for idx,(x,y) in enumerate(tqdm(Val_loader)):
            x = x.to(device=device, dtype=dtype)  # move to device, e.g. GPU
            y = y.to(device=device, dtype=dtype)
            scores = model(x)
            _,classification = scores.max(1)
            ind = y[:,-1]
            result = torch.sum(torch.eq(classification,ind))
            correct = result+correct
    accuarcy = correct/len(Val_set)
#%%
print(accuarcy)
# %%

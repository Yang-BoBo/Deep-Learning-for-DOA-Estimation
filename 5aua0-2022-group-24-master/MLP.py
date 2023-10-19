#%%
#%%
import os
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import TensorDataset,random_split,ConcatDataset
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
import data_generation as d
#%%
Max_sources = 4
N_angle = 2
Data_Size = 4096
#%%
SNR = [0,5,10,15,20,25,30]
Dataset_list = []
for ratio in SNR:
    x,aoa = d.create_i_o(ratio,n_angle=N_angle,Data_Size=Data_Size,max_gen_ang=70,
                    return_angle=True,return_Y=False)
    aoa = torch.tensor(aoa)
    x = torch.tensor(x)
    x = torch.unsqueeze(x,-2)
    hermitian_x = torch.conj(torch.transpose(x,-2,-1))
    cov_x = torch.matmul(hermitian_x,x)
    upperCov = torch.triu(cov_x)
    upperCov = upperCov.view(upperCov.size(0),-1)



    concat = torch.cat((real_cov,imag_cov),1)
    DataSet_WithSource = TensorDataset(concat,aoa)
    Dataset_list.append(DataSet_WithSource)

Full_dataset = ConcatDataset(Dataset_list)
#%%

#%%
Dataset_list = []
#%%

x,aoa = d.create_i_o(30,n_angle=2,Data_Size=Data_Size,max_gen_ang=70,
    return_angle=True,return_Y=False)
aoa = torch.tensor(aoa)
XrealPart = np.real(x)
XcomplexPart = np.imag(x)
Observation_Set = torch.tensor(np.hstack((XrealPart,XcomplexPart)))
DataSet_WithSource = TensorDataset(Observation_Set,aoa)

#%%
xVal,aoaVal = d.create_i_o(30,n_angle=2,Data_Size=1024,max_gen_ang=70,
    return_angle=True,return_Y=False)
aoaVal = torch.tensor(aoaVal)
XrealPartVal = np.real(xVal)
XcomplexPartVal = np.imag(xVal)
Observation_SetVal = torch.tensor(np.hstack((XrealPartVal,XcomplexPartVal)))
DataSet_WithSourceVal = TensorDataset(Observation_SetVal,aoaVal)



#%%
Full_dataset = torch.load('Aoa2_dataset.pt')
#%%
#%%
batch_size = 64
Train_loader = torch.utils.data.DataLoader(
       dataset=DataSet_WithSource,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True)

Val_loader = torch.utils.data.DataLoader(
       dataset=DataSet_WithSourceVal,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True)




#%%
import torch.nn.functional as F
class MLP(nn.Module): 
    def __init__(self, n_channels=72,channel1=1024,channel2=2048,channel3=4096,channel4=512, channel5=1024, channel6=2048, n_classes=14):
        super().__init__()
        #####################Encoder layers##############################
        self.encoder_1 = nn.Sequential(
            nn.Linear(n_channels, channel1),
            nn.BatchNorm1d(channel1),
            nn.ReLU(inplace = True),
        )

        self.encoder_2 = nn.Sequential(
            nn.Linear(channel1, channel2),
            nn.BatchNorm1d(channel1),
            nn.ReLU(inplace = True),
        )

        self.encoder_3 = nn.Sequential(
            nn.Linear(channel2, channel3),
            nn.BatchNorm1d(channel2),
            nn.ReLU(inplace = True),
        )
        self.encoder_4 = nn.Sequential(
            nn.Linear(channel3, channel3),
            nn.BatchNorm1d(channel3),
            nn.ReLU(inplace = True),
        )
        self.encoder_5 = nn.Sequential(
            nn.Linear(channel3, channel3),
            nn.BatchNorm1d(channel4),
            nn.ReLU(inplace = True),
        )
        self.encoder_6 = nn.Sequential(
            nn.Linear(channel3, channel2),
            nn.BatchNorm1d(channel5),
            nn.ReLU(inplace = True),
        )
        self.encoder_7 = nn.Sequential(
            nn.Linear(channel2, channel1),
            nn.BatchNorm1d(channel6),
            nn.ReLU(inplace = True),
        )
        self.regression4 = nn.Sequential(
            nn.Linear(channel1, 141),
        )
    def forward(self,x):
        
        x = self.encoder_1(x)

        x = self.encoder_2(x)

        x = self.encoder_3(x)

        x = self.encoder_4(x)

        x = self.encoder_5(x)

        x = self.encoder_6(x)

        x = self.encoder_7(x)

        reg4 = self.regression4(x)
        
        return reg4
#%%
model = MLP()
weight_decay = 1e-4
optimizer = optim.Adam(model.parameters(), lr=0.001)
#model = model.to(device=device)  # move the model parameters to CPU/GPU
epochs = 2000


#%%
for e in range(epochs):
    for t, (x, y) in enumerate((Train_loader)):
        model.train()  # put model to training mode
        optimizer.zero_grad()

        aoa4 = model(x)
        loss_regression = torch.sqrt(F.mse_loss(aoa4, y))
        
        loss_regression.backward()

        optimizer.step()

    if e%2 == 0:
        print('Epoch %d, Classification loss = %.4f' % (e, loss_regression.item()))

    with torch.no_grad():
        model.eval()
        for idx,(x,y) in enumerate((Val_loader)):
                reg4 = model(x)
                loss_regressionVal = torch.sqrt(F.mse_loss(reg4, y))
                loss = loss_regressionVal
        if e%2 == 0:
            print('Epoch %d, validation loss = %.4f' % (e, loss))
            print(f'Epoch {e}, Truth ={y[0,:]}')
            print(f'Epoch {e}, sources ={reg4[0,:]}')

# %%
for t, (x, y) in enumerate(tqdm(Train_loader)):
    print(y[0,:])
    if t == 8:
        break
# %%

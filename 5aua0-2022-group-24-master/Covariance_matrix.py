#%%
import os
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import TensorDataset, ConcatDataset,random_split
import torch.nn.functional as F
import torchvision
import PIL
import matplotlib.pyplot as plt
import pandas as pd
import zipfile
from tqdm import tqdm
import math
import torch.nn as nn
#%%
import data_generation as d
import Resnet_regression as r
#%%
SNR = 30
N_angle = 3
Max_sources = 4
Data_size = 4096
#%%
# transforms.ToTensor()
transform1 = torchvision.transforms.Compose([
    torchvision.transforms.Normalize(mean = (0.5, 0.5), std = (0.5, 0.5))
    ]
)
#%%
Dataset_list = []
#%%
SNR = [0,5,10,15,20,25,30]
for ratio in SNR:
    x,aoa = d.create_i_o(ratio,n_angle=N_angle,Data_Size=Data_size,max_gen_ang=70,
                    return_angle=True,return_Y=False)
    aoa = torch.tensor(aoa)
    x = torch.tensor(x)
    x = torch.unsqueeze(x,-2)
    hermitian_x = torch.conj(torch.transpose(x,-2,-1))
    cov_x = torch.matmul(hermitian_x,x)
    cov_x = torch.unsqueeze(cov_x,-3)
    real_cov = torch.real(cov_x)
    imag_cov = torch.imag(cov_x)
    concat = torch.cat((real_cov,imag_cov),1)
    DataSet_WithSource = TensorDataset(concat,aoa)
    Dataset_list.append(DataSet_WithSource)

Full_dataset = ConcatDataset(Dataset_list)
#%%
Dataset_list = []
#%%
SNR = [0,5,10,15,20,25,30]
for ratio in SNR:
    x,aoa = d.create_i_o(ratio,n_angle=N_angle,Data_Size=1024,max_gen_ang=70,
                    return_angle=True,return_Y=False)
    aoa = torch.tensor(aoa)
    x = torch.tensor(x)
    x = torch.unsqueeze(x,-2)
    hermitian_x = torch.conj(torch.transpose(x,-2,-1))
    cov_x = torch.matmul(hermitian_x,x)
    cov_x = torch.unsqueeze(cov_x,-3)
    real_cov = torch.real(cov_x)
    imag_cov = torch.imag(cov_x)
    concat = torch.cat((real_cov,imag_cov),1)
    DataSet_WithSource = TensorDataset(concat,aoa)
    Dataset_list.append(DataSet_WithSource)

Full_dataset_Val = ConcatDataset(Dataset_list)


# concat = transform1(concat)

#%%
# DataSet_WithSource = TensorDataset(concat,aoa)
# pad_angle = (0,Max_sources-N_angle)
# pad_source = (0,1)
# # aoa_p = F.pad(aoa,pad_angle,'constant',0)
# Label_Set = F.pad(aoa_p,pad_source,'constant',N_angle-1)
# %%
batch_size = 64
Train_loader = torch.utils.data.DataLoader(
       dataset=Full_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True)
batch_size = 64
Val_loader = torch.utils.data.DataLoader(
       dataset=Full_dataset_Val,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True)
# %%
model = r.ResNet18(3)
#%%
device = torch.device('cuda')
dtype = torch.float32
# %%
weight_decay = 1e-4
optimizer = optim.SGD(model.parameters(), lr=0.001, weight_decay=weight_decay,momentum=0.9)
model = model.to(device=device)  # move the model parameters to CPU/GPU
epochs = 2000
#%%
model.load_state_dict(torch.load("ResNet3_400.pth"))
#%%





#%%
for e in range(epochs):
    for t, (x, y) in enumerate((Train_loader)):
        x = x.to(device=device, dtype=dtype)  # move to device, e.g. GPU
        y = y.to(device=device, dtype=dtype)
        model.train()  # put model to training mode
        optimizer.zero_grad()

        aoa4 = model(x)
        loss_regression = F.mse_loss(aoa4, y)
        loss_regression = torch.sqrt(loss_regression)

        # Zero out all of the gradients for the variables which the optimizer
        # will update.

        # This is the backwards pass: compute the gradient of the loss with
        # respect to each  parameter of the model.
        # loss_regression.backward()
        # else:

        # optimizer.zero_grad()
        loss_regression.backward()
        # Actually update the parameters of the model using the gradients
        # computed by the backwards pass.
        optimizer.step()
        # print(f'prediction {doa1} \n truth {y[0]}')
        # print(f'classification {classification[0]+1}')
    if e%2 == 0:
        print('Epoch %d, Classification loss = %.4f' % (e, loss_regression.item()))
        print(f'Epoch {e}, Truth ={y[0,:]}')
        print(f'Epoch {e}, sources ={aoa4[0,:]}')
            # print()
            # print(scores[0,:])
            # print(y[0,:])
    with torch.no_grad():
        model.eval()
        for idx, (x, y) in enumerate((Val_loader)):
                x = x.to(device=device, dtype=dtype)  # move to device, e.g. GPU
                y = y.to(device=device, dtype=dtype)
                reg4 = model(x)
                reg4,_ = torch.sort(reg4)
                y,_ = torch.sort(y)
                loss_regression_val = F.mse_loss(reg4, y)
                loss_regression_val = torch.sqrt(loss_regression_val)
        if e%2 == 0:
            print('Epoch %d, Classification loss = %.4f' % (e, loss_regression_val))
            print(f'Epoch {e}, Truth ={y[0,:]}')
            print(f'Epoch {e}, sources ={reg4[0,:]}')
    if e == 200:
        torch.save(model.state_dict(),'ResNet3_200.pth')
    if e == 400:
        torch.save(model.state_dict(),'ResNet3_400.pth')
    if e == 600:
        torch.save(model.state_dict(),'ResNet3_600.pth')
    if e == 800:
        torch.save(model.state_dict(),'ResNet3_800.pth')
    if e == 1000:
        torch.save(model.state_dict(),'ResNet3_1000.pth')
    if e == 1200:
        torch.save(model.state_dict(),'ResNet3_1200.pth')
    if e == 1400:
        torch.save(model.state_dict(),'ResNet3_1400.pth')
    if e == 1600:
        torch.save(model.state_dict(),'ResNet3_1600.pth')
    if e == 1800:
        torch.save(model.state_dict(),'ResNet3_1800.pth')
    if e == 2000:
        torch.save(model.state_dict(),'ResNet3_2000.pth')

    
# %%
torch.save(model.state_dict(),'ResNet3_410.pth')
# %%

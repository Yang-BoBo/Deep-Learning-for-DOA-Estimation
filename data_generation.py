import os
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import TensorDataset
import torch.nn.functional as F
import torchvision
import PIL
import matplotlib.pyplot as plt
import pandas as pd
import zipfile
from tqdm import tqdm
import os
import math
import torch.nn as nn
#%%

#@title

#Parameters for the antenas
visible_region = 180
resolution = 1

num_antenna = 8
c0=3e8
fc= 76e9
lambda_var = c0/fc
d = lambda_var/(2*np.sin(np.pi/180*0.5*visible_region))

coordinate=np.array([0, 1, 4, 6, 13, 14, 17, 19])
NN_size_out = (visible_region/resolution)+1
NN_size_out =  np.array(NN_size_out,dtype='i')


#Create Input and Output for the NN
#aoa = matrix or vector
#A = matrix or vector
#max_gen_ang = Max_min allowed angle when generating own angles
#use_R=True Outputs the Covariance Matrix. If FALSE, outputs the snapshot
def create_i_o(SNR,aoa='random',A='default',n_angle=-1,Data_Size=-1,max_gen_ang=visible_region/2,
              return_angle=False,return_snapshot=False,use_R=False,return_Y=True,
              c0=c0,fc=fc,lambda_var=lambda_var,visible_region=visible_region,d=d,num_antenna=num_antenna,coordinate=coordinate, 
              NN_size_out=NN_size_out):
    
    ############# INTERNAL FUNCTIONS ########################
    #
    ########################
    def get_snapshot():
        r = np.zeros((Data_Size,n_angle,num_antenna))+0*1j
        for i in range(num_antenna):
            r[:,:,i] = A*np.exp(1j*(2*np.pi*fc*coordinate[i]*d*np.sin(np.pi*aoa/180)+2*np.pi*np.random.random())/c0)
        #    
        snapshot_org = np.sum(r[:,:,:],axis=1)
        #
        snapshot = awgn(snapshot_org)
        snapshot = np.array(snapshot)
        #
        if use_R==True:
            R = np.zeros((Data_Size,num_antenna,num_antenna))+0*1j
            for i in range(Data_Size):
                R[i,:,:] = np.outer(snapshot[i,:],snapshot[i,:])
            #
            if return_snapshot==False: #R_true, S_false
                return R
            else: #R_true, S_true
                return R,snapshot
        else: #R_false, S_true
            return snapshot
    #   
    ########################  
    def awgn(signal_snap): #add white gaussian noise
        power_snap = 10*np.log10(np.abs(signal_snap) ** 2)
        noise_dB = power_snap - SNR     
        noise_energy = 10 ** (noise_dB / 10)
        noise_to_signal = np.random.normal(0, np.sqrt(noise_energy))
        Re_noise_to_signal= np.random.random((Data_Size,num_antenna))*np.abs(noise_to_signal)
        Im_noise_to_signal = np.sqrt((noise_to_signal ** 2) - (Re_noise_to_signal ** 2))
        signal_noise = Re_noise_to_signal*((-1)**np.random.randint(0,2,(Data_Size,num_antenna))) + \
                    1j*Im_noise_to_signal*((-1)**np.random.randint(0,2,(Data_Size,num_antenna)))

        noisy_signal = signal_snap + signal_noise
        return noisy_signal
    #
    #######################################################################
    #
    ########   
    #Error#
    if n_angle > num_antenna-2:
        print('The number of sources is too big regarding the number of antennas!')   
        return
    if n_angle < 0 and aoa == 'random':
        print('Inform either "aoa" or the "n_angle"')
        return
    if Data_Size < 0 and aoa == 'random':
        print('Inform either "Data_Size" or the "n_angle"')
        return
    ########  
    #
    #Other Vars#
    if n_angle < 0:
        n_angle = aoa.shape[1]
    elif n_angle==0:
        n_angle=1
        A = np.zeros((Data_Size,n_angle))
        A[:,:] = 0.00000001       
    #
    if max_gen_ang < 90 and max_gen_ang > 0 and max_gen_ang != visible_region/2:
        visible_region = 2*max_gen_ang
    #
    if aoa == 'random':
        aoa = np.random.randint(-visible_region/2,1+visible_region/2,Data_Size*n_angle)
        aoa = np.reshape(aoa,(Data_Size,n_angle))
    else:
        if Data_Size < 0:
            Data_Size = aoa.shape[0]
    #   
    if A == 'default':
        A = np.ones((Data_Size,n_angle))
    #      
    if Data_Size < 1:   
        if aoa.ndim == 1:
            Data_Size = 1
        else:
            Data_Size = aoa.shape[0]
    Data_Size = np.array(Data_Size,dtype='i')
    #
    #Check SNR type:
    if (type(SNR) is list and len(SNR) == 2) or (type(SNR) is np.ndarray and SNR.size == 2):#If list or array and 2 elements
        SNR = np.array(SNR)
        SNR_init = SNR[0]
        SNR_end = SNR[1]
        SNR = np.random.randint(SNR_init,1+SNR_end,Data_Size) #creates the SNR with Data_Size elements
        SNR = np.transpose(np.tile(SNR, (num_antenna,1))) #Repeats the values in the row. SNR.shape is (data,num_antenna)
    #
    elif (type(SNR) is list and len(SNR) > 2) or (type(SNR) is np.ndarray and SNR.size > 2):#If more than 2 elements give error
        print('Too many arguments in for the SNR. 1 or 2 elements')
        return
    #        
    #Error#
    if A.size != aoa.size:
        print('"A" size is different than "aoa" size')
        return
    ######## 
    #
    #OUTPUT X
    if use_R==True:
        if return_snapshot==False: #R_true, S_false
            R = get_snapshot()
        else: #R_true, S_true
            R,snapshot = get_snapshot()
            
        X = R 
    #        
    else: #R_false, S_true
        snapshot = get_snapshot()
        X = snapshot
    #
    #OUTPUT Y
    if return_Y==True:
        NN_idx = ang2NN(aoa)
    #
        Y = (1 + np.random.random((Data_Size,NN_size_out))) / 10000
        for i in range(Data_Size):
            sumY = np.sum(Y[i,:])
            Y[i,NN_idx[i,:]] = (1 - sumY) / n_angle
    #    
    #RETURN 
    if  return_angle==False and return_snapshot==False and return_Y==True:
        return X,Y
    elif return_angle==True and return_snapshot==False and return_Y==True:
        return X,Y,aoa
    elif return_angle==True and return_snapshot==True  and return_Y==True:
        return X,Y,aoa,snapshot
    elif return_angle==False and return_snapshot==True and return_Y==True:
        return X,Y,snapshot
    elif return_angle==False and return_snapshot==True and return_Y==False:
        return X,snapshot
    elif return_angle==True and return_snapshot==False and return_Y==False:
        return X,aoa
    elif return_angle==False and return_snapshot==False and return_Y==False:
        return X
    elif return_angle==True and return_snapshot==True and return_Y==False:
        return X,aoa,snapshot
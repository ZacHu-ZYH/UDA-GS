import numpy as np
import torch
import random
import torch.nn as nn


def flip(flip, data = None, target = None):
    #Flip
    if flip == 1:
        if not (data is None): data = torch.flip(data,(3,))#np.array([np.fliplr(data[i]).copy() for i in range(np.shape(data)[0])])
        if not (target is None):
            target = torch.flip(target,(2,))#np.array([np.fliplr(target[i]).copy() for i in range(np.shape(target)[0])])
    return data, target

def cowMix(mask, data = None, target = None):
    #Mix
    if not (data is None):
        stackedMask, data = torch.broadcast_tensors(mask, data)
        stackedMask = stackedMask.clone()
        stackedMask[1::2]=1-stackedMask[1::2]
        data = (stackedMask*torch.cat((data[::2],data[::2]))+(1-stackedMask)*torch.cat((data[1::2],data[1::2]))).float()
    if not (target is None):
        stackedMask, target = torch.broadcast_tensors(mask, target)
        stackedMask = stackedMask.clone()
        stackedMask[1::2]=1-stackedMask[1::2]
        target = (stackedMask*torch.cat((target[::2],target[::2]))+(1-stackedMask)*torch.cat((target[1::2],target[1::2]))).float()
    return data, target


def oneMix(mask, data = None, target = None):
    #Mix
    if not (data is None):
        stackedMask0, _ = torch.broadcast_tensors(mask[0], data[0])
        data = (stackedMask0*data[0]+(1-stackedMask0)*data[1]).unsqueeze(0)
    if not (target is None):
        stackedMask0, _ = torch.broadcast_tensors(mask[0], target[0])
        target = (stackedMask0*target[0]+(1-stackedMask0)*target[1]).unsqueeze(0)
    return data, target


def normalize(MEAN, STD, data = None, target = None):
    #Normalize
    if not (data is None):
        if data.shape[1]==3:
            STD = torch.Tensor(STD).unsqueeze(0).unsqueeze(2).unsqueeze(3).cuda()
            MEAN = torch.Tensor(MEAN).unsqueeze(0).unsqueeze(2).unsqueeze(3).cuda()
            STD, data = torch.broadcast_tensors(STD, data)
            MEAN, data = torch.broadcast_tensors(MEAN, data)
            data = ((data-MEAN)/STD).float()
    return data, target

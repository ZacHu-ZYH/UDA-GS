import torch
import torch.nn as nn
import math
import numpy as np
import torch.nn.functional as F
from torch.autograd import Function




class Linear_classifier(nn.Module):
    def __init__(self, input_dim=256, hidden_dim=256):
        super(Linear_classifier, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.dis1 = nn.Linear(input_dim, hidden_dim)
        self.dis2 = nn.Linear(hidden_dim, 3)

    def forward(self, x):
        x = F.relu(self.dis1(x))
        x = self.dis2(x)
        x = torch.sigmoid(x)
        return x


def mttl(source, target,x_set,y_set,scale, input_dim=256, hidden_dim=512):
    domain_loss = nn.MSELoss()
    # !!! Pay attention to .cuda !!!
    adv_net = Linear_classifier(input_dim, hidden_dim).cuda()

    # print('dd',reverse_src,reverse_tar)
    inputs = torch.cat([source,target],1)
    print(inputs.shape)
    pred = adv_net(inputs)
    # pred_tar = adv_net(target)
    # print(pred_src, domain_src,pred_tar, domain_tar)
    label = torch.Tensor([[x_set/100,y_set/100,scale/4],[x_set/100,y_set/100,scale/4]])
    label = label.cuda()
    loss_s = domain_loss(pred, label)
    return loss_s
import numpy as np
import torch
from torch import nn
from torch.nn import init



class EMSA(nn.Module):

    def __init__(self, d_model, d_k, d_v, h,dropout=.1,H=7,W=7,ratio=3,apply_transform=True):

        super(EMSA, self).__init__()
        self.H=H
        self.W=W
        self.fc_q = nn.Linear(d_model, h * d_k)
        self.fc_k = nn.Linear(d_model, h * d_k)
        self.fc_v = nn.Linear(d_model, h * d_v)
        self.fc_o = nn.Linear(h * d_v, d_model)
        self.dropout=nn.Dropout(dropout)

        self.ratio=ratio
        if(self.ratio>1):
            self.sr=nn.Sequential()
            self.sr_conv=nn.Conv2d(d_model,d_model,kernel_size=ratio+1,stride=ratio,padding=ratio//2,groups=d_model)
            self.sr_ln=nn.LayerNorm(d_model)

        self.apply_transform=apply_transform and h>1
        if(self.apply_transform):
            self.transform=nn.Sequential()
            self.transform.add_module('conv',nn.Conv2d(h,h,kernel_size=1,stride=1))
            self.transform.add_module('softmax',nn.Softmax(-1))
            self.transform.add_module('in',nn.InstanceNorm2d(h))

 
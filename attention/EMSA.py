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
            self.sr=nn.Sequential
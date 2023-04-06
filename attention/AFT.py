import numpy as np
import torch
from torch import nn
from torch.nn import init



class AFT_FULL(nn.Module):

    def __init__(self, d_model,n=49,simple=False):

        super(AFT_FULL, self).__init__()
        self.fc_q = nn.Linear(d_model, d_model)
        self
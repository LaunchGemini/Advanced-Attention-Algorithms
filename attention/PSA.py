import numpy as np
import torch
from torch import nn
from torch.nn import init



class PSA(nn.Module):

    def __init__(self, channel=512,reduction=4,S=4):
        super().__init__()
        self.S=S

        self.convs=[]
        for i in range(S):
            self.convs.append(nn.Conv2d(channel//S,channel//S,kernel_size=2*(i+1)+1,padding=i+1))

        self.se_blocks=[]
        for i in range(S):
            self.se_blocks.append(nn.Sequential(
           
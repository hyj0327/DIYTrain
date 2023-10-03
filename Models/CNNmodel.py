#!/usr/bin/env python
# coding: utf-8

# In[10]:


import torch.nn as nn
import torch
import numpy as np


# In[8]:

batch_size=100



class CNN(nn.Module):
    def __init__(self,num_classes):
        super(CNN,self).__init__()


        self.layer=nn.Sequential(
            nn.Conv2d(in_channels=1,out_channels=16,kernel_size=5),
            nn.ReLU(),
            nn.Conv2d(in_channels=16,out_channels=32,kernel_size=5),
            nn.ReLU(),

            nn.MaxPool2d(kernel_size=2,stride=2),

            nn.Conv2d(in_channels=32,out_channels=64,kernel_size=5),
            nn.ReLU(),

            nn.MaxPool2d(kernel_size=2,stride=2))

        self.fc_layer=nn.Sequential(
            nn.Linear(64*3*3,100),
            nn.ReLU(),
            nn.Linear(100,num_classes))


    def forward(self,x):
        out=self.layer(x)
        out=out.view(batch_size,-1)
        out=self.fc_layer(out)
        return out


# In[ ]:





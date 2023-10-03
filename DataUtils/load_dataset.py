#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import numpy as np
import torch
from torch.utils.data import Dataset


# In[2]:

def load_dataset2(dtype,root='Dataset'):
    num_classes=0
    with open("./DataUtils/class_names.txt",'r') as f:
        for line in f:
            num_classes=num_classes+1
    
    data=np.load(os.path.join(root,dtype+'.npz'))
    X_data=data['data']
    Y_data=data['target']
    
    X_data=X_data.reshape(len(X_data),28,28)
    X_data=np.expand_dims(X_data,1)
    
    return X_data.astype('float32'),Y_data.astype('int64'),num_classes


def load_dataset(dtype,root='Dataset'):
    num_classes=0
    with open("./DataUtils/class_names.txt",'r') as f:
        for line in f:
            num_classes=num_classes+1
            
    data=np.load(os.path.join(root,dtype+'.npz'))
    return data['data'].astype('float32'),data['target'].astype('int64'),num_classes


# In[3]:


class QuickDrawDataset(Dataset):
    def __init__(self,dtype):
        self.x,self.y,self.num_classes=load_dataset(dtype=dtype)
        self.x=torch.from_numpy(self.x)
        self.y=torch.from_numpy(self.y)

        
    def __getitem__(self,index):
        return self.x[index],self.y[index]
    
    def __len__(self):
        return len(self.x)
    
    def get_num_classes(self):
        return self.num_classes


# In[ ]:





#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import torch
import os
import glob
import matplotlib.pyplot as plt
from random import randint


# In[2]:


npyfile_root='Data'
npzfile_root='Dataset'
max_samples_per_class=5000
vfold_ratio=0.2


# In[3]:


#파일명 불러오기
all_files=glob.glob(os.path.join('../'+npyfile_root,'*.npy'))
print(all_files)


# In[4]:


data=np.load(all_files[1])
print(data.shape)


# In[5]:


x=np.empty([0,784])
y=np.empty([0])


# In[6]:


class_names=[]
class_samples_num=[]


# In[ ]:





# In[7]:


for idx,file in enumerate(all_files):
    data=np.load(file)
    indices=np.arange(0,data.shape[0])
    
    indices=np.random.choice(indices,max_samples_per_class,replace=False)
    data = data[indices]
    
    #print(data.shape)
    labels=np.full(data.shape[0],idx)
    
    
    x=np.concatenate((x,data),axis=0)
    y=np.append(y,labels)
    
    class_name,ext=os.path.splitext(os.path.basename(file))
    class_names.append(class_name)
    class_samples_num.append(str(data.shape[0]))
    
    
    data=None
    labels=None
    
    permutation=np.random.permutation(y.shape[0])
    x=x[permutation,:]
    y=y[permutation]
    
    vfold_size=int(x.shape[0]/100*(vfold_ratio*100))
    
    x_test=x[0:vfold_size:]
    y_test=y[0:vfold_size]
    
    x_train=x[vfold_size:x.shape[0],:]
    y_train=y[vfold_size:y.shape[0]]
      



# In[11]:


idx=randint(0,len(x_train))
plt.imshow(x_train[idx].reshape(28,28))


# In[12]:


#데이터 npz파일로 압축
np.savez_compressed('../'+npzfile_root+"/train", data=x_train,target=y_train)

np.savez_compressed('../'+npzfile_root+"/test", data=x_test,target=y_test)


# In[5]:


with open("./class_names.txt", 'w') as f:
    for i in range(len(class_names)):
        f.write("class name: "+class_names[i]+"\t\tnumber of samples: "+class_samples_num[i]+"\n")


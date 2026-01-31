#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
sys.path.append("..")

#import jax
#jax.config.update("jax_enable_x64", True)

import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "0,2"

from utils import *
from model_tn import *
from keras_utils import *
from jax_utils import *

keras.mixed_precision.set_global_policy("mixed_float16")


# In[2]:


def analys(model, input, frames=1):
    model(input)
    t0 = time.time()
    for _ in range(5):
        model(input).block_until_ready()
    latency = int((time.time()-t0)*1000/5)
    r = jax.jit(model).lower(input).compile().cost_analysis()[0]
    params = model.count_params()
    flops = int(r['flops'])
    throughput = int(np.sum([j for i, j in r.items() if 'bytes accessed' in i]))
    return f'Params: {params/1e3:.0f} K, FLOPs: {flops/frames/1e6:.2f} M, MT: {throughput/frames/1e6:.2f} MB, Latency: {latency/frames*900:.2f} ms'


# In[3]:


tnts_san = SAN(TS=True, TN=False, depth=2)
input = np.random.random((1, 160, 36, 36, 3)).astype('float16')
analys(tnts_san, input, 160) # proposed w/o TN


# In[4]:


tnts_san = SAN(TS=True, TN=True, depth=2)
input = np.random.random((1, 160, 36, 36, 3)).astype('float16')
analys(tnts_san, input, 160) # proposed


# In[5]:


class TSCANToEnd(keras.Model):
    
    def __init__(self, model):
        super().__init__()
        self.inner = model
    
    def call(self, x, training=None):
        x_ = x[:, 1:] - x[:, :-1]
        x_ = (x_ - ops.mean(x_, axis=(2,3 ), keepdims=True))/(ops.std(x_, axis=(2, 3), keepdims=True)+1e-6)
        return self.inner((ops.concatenate([x_, x_[:, -1:]], axis=1), ops.mean(x, axis=(1, ), keepdims=True)), training=training)

model = TSCANToEnd(TSCAN())
input = np.random.random((1, 160, 36, 36, 3)).astype('float16')
analys(model, input, 160)


# In[6]:


class TSCANToEnd(keras.Model):
    
    def __init__(self, model):
        super().__init__()
        self.inner = model
    
    def call(self, x, training=None):
        return self.inner((x, ops.mean(x, axis=(1, ), keepdims=True)), training=training)

model = TSCANToEnd(TSCAN(TN=True))
input = np.random.random((1, 160, 36, 36, 3)).astype('float16')
analys(model, input, 160)


# In[7]:


model = PhysNet()
input = np.random.random((1, 128, 32, 32, 3)).astype('float16')
analys(model, input, 128)


# In[8]:


model = PhysNet(TN=True)
input = np.random.random((1, 128, 32, 32, 3)).astype('float16')
analys(model, input, 128)


# In[9]:


model = EfficientPhys()
input = np.random.random((1, 160, 72, 72, 3)).astype('float16')
analys(model, input, 160)


# In[10]:


model = EfficientPhys(TN=True)
input = np.random.random((1, 160, 72, 72, 3)).astype('float16')
analys(model, input, 160)


# In[11]:


model = PhysFormer()
input = np.random.random((1, 160, 128, 128, 3)).astype('float16')
analys(model, input, 160)


# In[12]:


model = PhysFormer(TN=True)
input = np.random.random((1, 160, 128, 128, 3)).astype('float16')
analys(model, input, 160)


# In[ ]:





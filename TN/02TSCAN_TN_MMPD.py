#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
sys.path.append("..")

#import jax
#jax.config.update("jax_enable_x64", True)

from utils import *
from model_tn import *
from keras_utils import *
from jax_utils import *

keras.mixed_precision.set_global_policy("mixed_float16")
#keras.mixed_precision.set_global_policy("float32")
#keras.mixed_precision.set_global_policy("float64")


# In[2]:


class TSCANToEnd(keras.Model):
    
    def __init__(self, model):
        super().__init__()
        self.inner = model
    
    def call(self, x, training=None):
        return self.inner((x, ops.mean(x, axis=(1, ), keepdims=True)), training=training)

model = TSCANToEnd(TSCAN(TN=True))
model.compile(optimizer='adamw', loss='mse')
model(np.random.random((4, 160, 36, 36, 3)));
model.inner.summary()


# In[3]:


batch_size = 128

tape = "/root/ssd_cache/rppg_training_data/mmpd_160x36x36_all"

train_tape = load_datatape(tape, fold='train', batch=batch_size)
valid_tape = load_datatape(tape, fold='val', extended_hr='False', batch=batch_size)

#train_rlap = DatatapeMonitor(train_rlap)
training_set, validation_set = KerasDataset(train_tape), KerasDataset(valid_tape)

training_set = training_set.apply_fn(compress_aug)


# In[4]:


#model.fit(training_set, validation_data=validation_set, epochs=10, callbacks=[keras.callbacks.ModelCheckpoint('../weights/tscantn_mmpd.weights.h5', monitor='val_loss', save_best_only=True, save_weights_only=True)])
train(model, training_set, validation_set, epochs=20, check_point_path='../weights/tscantn_mmpd.weights.h5')
model.load_weights('../weights/tscantn_mmpd.weights.h5')


# In[5]:


eval_on_dataset(dataset_H5_rlap, pmodel(model), 160, (36, 36), step=4, batch=16, save='../results/TSCANtn_MMPD_RLAP.h5', scenes=['R1', 'R2', 'R3', 'R4'])
get_metrics('../results/TSCANtn_MMPD_RLAP.h5')


# In[6]:


eval_on_dataset(dataset_H5_cohface, pmodel(model), 160, (36, 36), step=4, batch=16, save='../results/TSCANtn_MMPD_COHFACE.h5', fps=30)
get_metrics('../results/TSCANtn_MMPD_COHFACE.h5')


# In[7]:


eval_on_dataset(dataset_H5_pure, pmodel(model), 160, (36, 36), step=4, batch=16, save='../results/TSCANtn_MMPD_PURE.h5')
get_metrics('../results/TSCANtn_MMPD_PURE.h5') 


# In[8]:


eval_on_dataset(dataset_H5_ubfc_rppg2, pmodel(model), 160, (36, 36), step=4, batch=16, save='../results/TSCANtn_MMPD_UBFC.h5')
get_metrics('../results/TSCANtn_MMPD_UBFC.h5')


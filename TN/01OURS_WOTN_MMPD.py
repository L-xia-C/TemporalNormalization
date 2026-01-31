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
#keras.mixed_precision.set_global_policy("float32")
#keras.mixed_precision.set_global_policy("float64")


# In[2]:


tnts_san = SAN(TS=True, TN=False, depth=2)
tnts_san.compile(optimizer='adam', loss='mae')
tnts_san(np.random.random((1, 160, 36, 36, 3)));
tnts_san.summary()


# In[3]:


batch_size = 128

tape = "/root/ssd_cache/rppg_training_data/mmpd_160x36x36_all"

train_tape = load_datatape(tape, fold='train', batch=batch_size)
valid_tape = load_datatape(tape, fold='val', extended_hr='False', batch=batch_size)

training_set, validation_set = KerasDataset(train_tape), KerasDataset(valid_tape)

#training_set = training_set.apply_fn(load_to_gpu) # pre-load
training_set = training_set.apply_fn(compress_aug)
#training_set = training_set.apply_fn(gnoise_aug)


# In[4]:


#tnts_san.fit(training_set, validation_data=validation_set, epochs=20, callbacks=[keras.callbacks.ModelCheckpoint('../weights/ts_san_36_mmpd.weights.h5', monitor='val_loss', save_best_only=True, save_weights_only=True)])
train(tnts_san, training_set, validation_set, epochs=20, check_point_path='../weights/ts_san_36_mmpd.weights.h5')
tnts_san.load_weights('../weights/ts_san_36_mmpd.weights.h5')


# In[5]:


eval_on_dataset(dataset_H5_rlap, pmodel(tnts_san), 160, (36, 36), step=4, batch=16, save='../results/TS-SAN-F2W_MMPD_RLAP.h5', scenes=['R1', 'R2', 'R3', 'R4'])
get_metrics('../results/TS-SAN-F2W_MMPD_RLAP.h5')


# In[6]:


eval_on_dataset(dataset_H5_cohface, pmodel(tnts_san), 160, (36, 36), step=4, batch=16, save='../results/TS-SAN-F2W_MMPD_COHFACE.h5', fps=30)
get_metrics('../results/TS-SAN-F2W_MMPD_COHFACE.h5')


# In[7]:


eval_on_dataset(dataset_H5_pure, pmodel(tnts_san), 160, (36, 36), step=4, batch=16, save='../results/TS-SAN-F2W_MMPD_PURE.h5')
get_metrics('../results/TS-SAN-F2W_MMPD_PURE.h5')


# In[8]:


eval_on_dataset(dataset_H5_ubfc_rppg2, pmodel(tnts_san), 160, (36, 36), step=4, batch=16, save='../results/TS-SAN-F2W_MMPD_UBFC.h5')
get_metrics('../results/TS-SAN-F2W_MMPD_UBFC.h5')


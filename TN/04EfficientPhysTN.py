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


model = EfficientPhys(TN=True)
model.compile(optimizer='adamw', loss='mse')
model(np.random.random((4, 160, 72, 72, 3)));
model.summary()


# In[3]:


batch_size = 32

rlap_tape = "/root/ssd_cache/rppg_training_data/rlap_160x72x72_all"

train_rlap = load_datatape(rlap_tape, fold='train', batch=batch_size)
valid_rlap = load_datatape(rlap_tape, fold='val', extended_hr='False', batch=batch_size)

train_rlap = DatatapeMonitor(train_rlap)
training_set, validation_set = KerasDataset(train_rlap), KerasDataset(valid_rlap)

training_set = training_set.apply_fn(compress_aug)


# In[4]:


#model.fit(training_set, validation_data=validation_set, epochs=10, callbacks=[keras.callbacks.ModelCheckpoint('../weights/efficientphys.weights.h5', monitor='val_loss', save_best_only=True, save_weights_only=True)])
train(model, training_set, validation_set, epochs=20, check_point_path='../weights/efficientphystn.weights.h5')
model.load_weights('../weights/efficientphystn.weights.h5')


# In[5]:


eval_on_dataset(dataset_H5_mmpd, pmodel(model), 160, (72, 72), step=4, batch=16, save='../results/EfficientPhystn_RLAP_MMPD.h5')
get_metrics('../results/EfficientPhystn_RLAP_MMPD.h5', dropped='False')


# In[6]:


eval_on_dataset(dataset_H5_cohface, pmodel(model), 160, (72, 72), step=4, batch=16, save='../results/EfficientPhystn_RLAP_COHFACE.h5', fps=30)
get_metrics('../results/EfficientPhystn_RLAP_COHFACE.h5')


# In[7]:


eval_on_dataset(dataset_H5_pure, pmodel(model), 160, (72, 72), step=4, batch=16, save='../results/EfficientPhystn_RLAP_PURE.h5')
get_metrics('../results/EfficientPhystn_RLAP_PURE.h5')


# In[8]:


eval_on_dataset(dataset_H5_ubfc_rppg2, pmodel(model), 160, (72, 72), step=4, batch=16, save='../results/EfficientPhystn_RLAP_UBFC.h5')
get_metrics('../results/EfficientPhystn_RLAP_UBFC.h5')


# In[ ]:





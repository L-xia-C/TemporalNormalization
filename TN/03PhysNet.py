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


model = PhysNet()
model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-4), loss=np_loss)
model(np.random.random((4, 128, 32, 32, 3)));
model.summary()


# In[3]:


batch_size = 128

rlap_tape = "/root/ssd_cache/rppg_training_data/rlap_128x32x32_all"

train_rlap = load_datatape(rlap_tape, fold='train', batch=batch_size)
valid_rlap = load_datatape(rlap_tape, fold='val', extended_hr='False', batch=batch_size)

#train_rlap = DatatapeMonitor(train_rlap)
training_set, validation_set = KerasDataset(train_rlap), KerasDataset(valid_rlap)

training_set = training_set.apply_fn(compress_aug)


# In[4]:


#model.fit(training_set, validation_data=validation_set, epochs=10, callbacks=[keras.callbacks.ModelCheckpoint('../weights/physnet.weights.h5', monitor='val_loss', save_best_only=True, save_weights_only=True)])
train(model, training_set, validation_set, epochs=20, check_point_path='../weights/physnet.weights.h5')
model.load_weights('../weights/physnet.weights.h5')


# In[5]:


model = PhysNet()
model(np.random.random((4, 128, 32, 32, 3)));
model.load_weights('../weights/physnet.weights.h5')


# In[6]:


eval_on_dataset(dataset_H5_mmpd, pmodel(model), 128, (32, 32), step=3, batch=16, save='../results/PhysNet_RLAP_MMPD.h5')
get_metrics('../results/PhysNet_RLAP_MMPD.h5', dropped='False')


# In[7]:


eval_on_dataset(dataset_H5_cohface, pmodel(model), 128, (32, 32), step=3, batch=16, save='../results/PhysNet_RLAP_COHFACE.h5', fps=30)
get_metrics('../results/PhysNet_RLAP_COHFACE.h5')


# In[8]:


eval_on_dataset(dataset_H5_pure, pmodel(model), 128, (32, 32), step=3, batch=16, save='../results/PhysNet_RLAP_PURE.h5')
get_metrics('../results/PhysNet_RLAP_PURE.h5')


# In[9]:


eval_on_dataset(dataset_H5_ubfc_rppg2, pmodel(model), 128, (32, 32), step=3, batch=16, save='../results/PhysNet_RLAP_UBFC.h5')
get_metrics('../results/PhysNet_RLAP_UBFC.h5')


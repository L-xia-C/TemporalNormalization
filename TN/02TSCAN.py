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
        x_ = x[:, 1:] - x[:, :-1]
        x_ = (x_ - ops.mean(x_, axis=(2,3 ), keepdims=True))/(ops.std(x_, axis=(2, 3), keepdims=True)+1e-6)
        return self.inner((ops.concatenate([x_, x_[:, -1:]], axis=1), ops.mean(x, axis=(1, ), keepdims=True)), training=training)

model = TSCANToEnd(TSCAN())
model.compile(optimizer='adamw', loss='mse')
model(np.random.random((4, 160, 36, 36, 3)));
model.inner.summary()


# In[3]:


@jax.jit
@partial(shard_map, mesh=mesh, in_specs=P('x'), out_specs=P('x'))
def diff_label(x, y):
    y = ops.concatenate([y[:,1:]-y[:,:-1], ops.zeros_like(y[:,:1])], axis=1)
    return x, y

batch_size = 128

rlap_tape = "/root/ssd_cache/rppg_training_data/rlap_160x36x36_all"

train_rlap = load_datatape(rlap_tape, fold='train', batch=batch_size)
valid_rlap = load_datatape(rlap_tape, fold='val', extended_hr='False', batch=batch_size)

#train_rlap = DatatapeMonitor(train_rlap)
training_set, validation_set = KerasDataset(train_rlap), KerasDataset(valid_rlap)

training_set = training_set.apply_fn(compress_aug)
training_set = training_set.apply_fn(diff_label)
validation_set = validation_set.apply_fn(diff_label)


# In[4]:


#model.fit(training_set, validation_data=validation_set, epochs=10, callbacks=[keras.callbacks.ModelCheckpoint('../weights/tscan.weights.h5', monitor='val_loss', save_best_only=True, save_weights_only=True)])
train(model, training_set, validation_set, epochs=20, check_point_path='../weights/tscan.weights.h5')
model.load_weights('../weights/tscan.weights.h5')


# In[5]:


model = TSCANToEnd(TSCAN())
model(np.random.random((4, 160, 36, 36, 3)));
model.load_weights('../weights/tscan.weights.h5')


# In[6]:


eval_on_dataset(dataset_H5_mmpd, pmodel(model), 160, (36, 36), step=4, batch=16, save='../results/TSCAN_RLAP_MMPD.h5', cumsum=True)
get_metrics('../results/TSCAN_RLAP_MMPD.h5', dropped='False')


# In[7]:


eval_on_dataset(dataset_H5_cohface, pmodel(model), 160, (36, 36), step=4, batch=16, save='../results/TSCAN_RLAP_COHFACE.h5', cumsum=True, fps=30)
get_metrics('../results/TSCAN_RLAP_COHFACE.h5')


# In[8]:


eval_on_dataset(dataset_H5_pure, pmodel(model), 160, (36, 36), step=4, batch=16, save='../results/TSCAN_RLAP_PURE.h5', cumsum=True)
get_metrics('../results/TSCAN_RLAP_PURE.h5') 


# In[9]:


eval_on_dataset(dataset_H5_ubfc_rppg2, pmodel(model), 160, (36, 36), step=4, batch=16, save='../results/TSCAN_RLAP_UBFC.h5', cumsum=True)
get_metrics('../results/TSCAN_RLAP_UBFC.h5')


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


@jax.jit
@partial(shard_map, mesh=mesh, in_specs=P('x'), out_specs=P('x'))
def u8_to_fp16(x, y):
    x = jnp.astype(x/255., 'float16')
    return x, y

def random_horizontal_flip(x, y, p=.5, root_key=[jax.random.PRNGKey(0)]):
    root_key[0], key = jax.random.split(root_key[0], 2)
    if jax.random.uniform(key)<p:
        return x[..., ::-1 ,:], y
    return x, y

batch_size = 16

tape = "/root/ssd_cache/rppg_training_data/mmpd_160x128x128_all"

train_tape = load_datatape(tape, fold='train', batch=batch_size, dtype='uint8')
valid_tape = load_datatape(tape, fold='val', extended_hr='False', batch=batch_size, dtype='uint8')

#train_rlap = DatatapeMonitor(train_tape)
training_set, validation_set = KerasDataset(train_tape), KerasDataset(valid_tape)

training_set = training_set.apply_fn(random_horizontal_flip)
training_set = training_set.apply_fn(u8_to_fp16)
validation_set = validation_set.apply_fn(u8_to_fp16)
training_set = training_set.apply_fn(compress_aug)


# In[3]:


model = PhysFormer()
#lr_schedule = keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=1e-4, decay_steps=50, decay_rate=0.5)
#opti = keras.optimizers.Adam(learning_rate=lr_schedule, weight_decay=1e-5) # convergence is too slow

a_start, b_start, exp_b = 1., .0, 1.
a, b = a_start, b_start
def combined_loss(y, pred):
    return a*np_loss(y, pred) + b*kl_ce_loss(y, pred)

model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-4), loss=combined_loss, metrics=[np_loss, kl_ce_loss])
y = model(np.random.random((4, 160, 128, 128, 3)));
model.summary()


# In[4]:


stat = None
for _ in range(10):
    stat = train(model, training_set, validation_set, epochs=1, check_point_path='../weights/physformer_mmpd.weights.h5', training_stat=stat)
    b = b_start + exp_b*(2**(stat['epoch']/10)-1) # update combined loss weight
stat['best_loss'] = 1e20
train(model, training_set, validation_set, epochs=10, check_point_path='../weights/physformer_mmpd.weights.h5', training_stat=stat)
model.load_weights('../weights/physformer_mmpd.weights.h5')


# In[5]:


eval_on_dataset(dataset_H5_mmpd, pmodel(lambda x:model(x/255.)), 160, (128, 128), step=4, batch=8, save='../results/PhysFormer_MMPD_RLAP.h5', ipt_dtype='uint8', scenes=['R1', 'R2', 'R3', 'R4'])
get_metrics('../results/PhysFormer_MMPD_RLAP.h5')


# In[6]:


eval_on_dataset(dataset_H5_cohface, pmodel(lambda x:model(x/255.)), 160, (128, 128), step=4, batch=8, save='../results/PhysFormer_MMPD_COHFACE.h5', ipt_dtype='uint8', fps=30)
get_metrics('../results/PhysFormer_MMPD_COHFACE.h5')


# In[7]:


eval_on_dataset(dataset_H5_pure, pmodel(lambda x:model(x/255.)), 160, (128, 128), step=4, batch=8, save='../results/PhysFormer_MMPD_PURE.h5', ipt_dtype='uint8')
get_metrics('../results/PhysFormer_MMPD_PURE.h5')


# In[8]:


eval_on_dataset(dataset_H5_ubfc_rppg2, pmodel(lambda x:model(x/255.)), 160, (128, 128), step=4, batch=8, save='../results/PhysFormer_MMPD_UBFC.h5', ipt_dtype='uint8')
get_metrics('../results/PhysFormer_MMPD_UBFC.h5')


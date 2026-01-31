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


# In[2]:


class PhysNet16(keras.Model):

    def __init__(self, TN=False):
        super().__init__()
        self.TN = TN
        self.ConvBlock3 = keras.Sequential([
            layers.Conv3D(64, kernel_size=(3, 3, 3), strides=1, padding='same'),
            layers.BatchNormalization(),
            layers.Activation('relu')
        ])
        self.ConvBlock5 = keras.Sequential([
            layers.Conv3D(64, kernel_size=(3, 3, 3), strides=1, padding='same'),
            layers.BatchNormalization(),
            layers.Activation('relu')
        ])
        self.ConvBlock7 = keras.Sequential([
            layers.Conv3D(64, kernel_size=(3, 3, 3), strides=1, padding='same'),
            layers.BatchNormalization(),
            layers.Activation('relu')
        ])
        self.ConvBlock9 = keras.Sequential([
            layers.Conv3D(16, kernel_size=(3, 3, 3), strides=1, padding='same'),
            layers.BatchNormalization(),
            layers.Activation('relu')
        ])
        self.upsample = keras.Sequential([
            layers.Conv3DTranspose(16, kernel_size=(4, 1, 1), strides=(2, 1, 1), padding='same'),
            layers.BatchNormalization(),
            layers.Activation('elu')
        ])
        self.upsample2 = keras.Sequential([
            layers.Conv3DTranspose(8, kernel_size=(4, 1, 1), strides=(2, 1, 1), padding='same'),
            layers.BatchNormalization(),
            layers.Activation('elu')
        ])
        self.convBlock10 = layers.Conv3D(1, kernel_size=(1, 1, 1), strides=1)
        self.MaxpoolSpa = layers.MaxPool3D((1, 2, 2), strides=(1, 2, 2))
        self.MaxpoolSpaTem = layers.MaxPool3D((2, 2, 2), strides=2)
        self.poolspa = layers.AvgPool3D((1, 2, 2))
        self.flatten = layers.Reshape((-1,))

    def call(self, x, training=None):
        #x = TNM(self.TN, axis=1, frames=x.shape[1])(x)
        #x = self.ConvBlock1(x, training=training)
        #x = self.MaxpoolSpa(x)
        #x = TNM(self.TN, axis=1, frames=x.shape[1])(x)
        #x = self.ConvBlock2(x, training=training)
        x = TNM(self.TN, axis=1, frames=x.shape[1])(x)
        x = self.ConvBlock3(x, training=training)
        x = self.MaxpoolSpaTem(x)
        #x = TNM(self.TN, axis=1, frames=x.shape[1])(x)
        #x = self.ConvBlock4(x, training=training)
        x = TNM(self.TN, axis=1, frames=x.shape[1])(x)
        x = self.ConvBlock5(x, training=training)
        x = self.MaxpoolSpaTem(x)
        #x = TNM(self.TN, axis=1, frames=x.shape[1])(x)
        #x = self.ConvBlock6(x, training=training)
        x = TNM(self.TN, axis=1, frames=x.shape[1])(x)
        x = self.ConvBlock7(x, training=training)
        x = self.MaxpoolSpa(x)
        #x = TNM(self.TN, axis=1, frames=x.shape[1])(x)
        #x = self.ConvBlock8(x, training=training)
        x = TNM(self.TN, axis=1, frames=x.shape[1])(x)
        x = self.ConvBlock9(x, training=training)
        x = self.upsample(x, training=training)
        x = self.upsample2(x, training=training)
        x = self.poolspa(x)
        x = self.convBlock10(x, training=training)
        x = self.flatten(x)
        x = (x-ops.mean(x, axis=1, keepdims=True))/ops.std(x, axis=1, keepdims=True)
        return x

class PhysNet8(keras.Model):

    def __init__(self, TN=False):
        super().__init__()
        self.TN = TN
        self.ConvBlock3 = keras.Sequential([
            layers.Conv3D(32, kernel_size=(3, 3, 3), strides=1, padding='same'),
            layers.BatchNormalization(),
            layers.Activation('relu')
        ])
        self.ConvBlock5 = keras.Sequential([
            layers.Conv3D(32, kernel_size=(3, 3, 3), strides=1, padding='same'),
            layers.BatchNormalization(),
            layers.Activation('relu')
        ])
        self.ConvBlock9 = keras.Sequential([
            layers.Conv3D(32, kernel_size=(3, 3, 3), strides=1, padding='same'),
            layers.BatchNormalization(),
            layers.Activation('relu')
        ])
        self.upsample = keras.Sequential([
            layers.Conv3DTranspose(8, kernel_size=(4, 1, 1), strides=(2, 1, 1), padding='same'),
            layers.BatchNormalization(),
            layers.Activation('elu')
        ])
        self.convBlock10 = layers.Conv3D(1, kernel_size=(1, 1, 1), strides=1)
        self.MaxpoolSpa = layers.MaxPool3D((1, 2, 2), strides=(1, 2, 2))
        self.MaxpoolSpaTem = layers.MaxPool3D((2, 2, 2), strides=2)
        self.poolspa = layers.AvgPool3D((1, 2, 2))
        self.flatten = layers.Reshape((-1,))

    def call(self, x, training=None):
        #x = TNM(self.TN, axis=1, frames=x.shape[1])(x)
        #x = self.ConvBlock1(x, training=training)
        #x = self.MaxpoolSpa(x)
        #x = TNM(self.TN, axis=1, frames=x.shape[1])(x)
        #x = self.ConvBlock2(x, training=training)
        x = TNM(self.TN, axis=1, frames=x.shape[1])(x)
        x = self.ConvBlock3(x, training=training)
        #x = self.MaxpoolSpaTem(x)
        #x = TNM(self.TN, axis=1, frames=x.shape[1])(x)
        #x = self.ConvBlock4(x, training=training)
        x = TNM(self.TN, axis=1, frames=x.shape[1])(x)
        x = self.ConvBlock5(x, training=training)
        x = self.MaxpoolSpaTem(x)
        #x = TNM(self.TN, axis=1, frames=x.shape[1])(x)
        #x = self.ConvBlock6(x, training=training)
        #x = TNM(self.TN, axis=1, frames=x.shape[1])(x)
        #x = self.ConvBlock7(x, training=training)
        x = self.MaxpoolSpa(x)
        #x = TNM(self.TN, axis=1, frames=x.shape[1])(x)
        #x = self.ConvBlock8(x, training=training)
        x = TNM(self.TN, axis=1, frames=x.shape[1])(x)
        x = self.ConvBlock9(x, training=training)
        x = self.upsample(x, training=training)
        #x = self.upsample2(x, training=training)
        x = self.poolspa(x)
        x = self.convBlock10(x, training=training)
        x = self.flatten(x)
        x = (x-ops.mean(x, axis=1, keepdims=True))/ops.std(x, axis=1, keepdims=True)
        return x

class PhysNet4(keras.Model):

    def __init__(self, TN=False):
        super().__init__()
        self.TN = TN
        self.ConvBlock3 = keras.Sequential([
            layers.Conv3D(16, kernel_size=(3, 3, 3), strides=1, padding='same'),
            layers.BatchNormalization(),
            layers.Activation('relu')
        ])
        self.ConvBlock5 = keras.Sequential([
            layers.Conv3D(16, kernel_size=(3, 3, 3), strides=1, padding='same'),
            layers.BatchNormalization(),
            layers.Activation('relu')
        ])
        self.ConvBlock9 = keras.Sequential([
            layers.Conv3D(4, kernel_size=(3, 3, 3), strides=1, padding='same'),
            layers.BatchNormalization(),
            layers.Activation('relu')
        ])
        self.convBlock10 = layers.Conv3D(1, kernel_size=(1, 1, 1), strides=1)
        self.MaxpoolSpa = layers.MaxPool3D((1, 2, 2), strides=(1, 2, 2))
        self.poolspa = layers.AvgPool3D((1, 2, 2))
        self.flatten = layers.Reshape((-1,))

    def call(self, x, training=None):
        x = TNM(self.TN, axis=1, frames=x.shape[1])(x)
        x = self.ConvBlock3(x, training=training)
        x = self.MaxpoolSpa(x)
        #x = TNM(self.TN, axis=1, frames=x.shape[1])(x)
        #x = self.ConvBlock2(x, training=training)
        #x = TNM(self.TN, axis=1, frames=x.shape[1])(x)
        #x = self.ConvBlock4(x, training=training)
        x = TNM(self.TN, axis=1, frames=x.shape[1])(x)
        x = self.ConvBlock5(x, training=training)
        #x = self.MaxpoolSpaTem(x)
        #x = TNM(self.TN, axis=1, frames=x.shape[1])(x)
        #x = self.ConvBlock6(x, training=training)
        #x = TNM(self.TN, axis=1, frames=x.shape[1])(x)
        #x = self.ConvBlock7(x, training=training)
        #x = self.MaxpoolSpa(x)
        #x = TNM(self.TN, axis=1, frames=x.shape[1])(x)
        #x = self.ConvBlock8(x, training=training)
        x = TNM(self.TN, axis=1, frames=x.shape[1])(x)
        x = self.ConvBlock9(x, training=training)
        #x = self.upsample(x, training=training)
        #x = self.upsample2(x, training=training)
        x = self.poolspa(x)
        x = self.convBlock10(x, training=training)
        x = self.flatten(x)
        x = (x-ops.mean(x, axis=1, keepdims=True))/ops.std(x, axis=1, keepdims=True)
        return x


# In[3]:


model = PhysNet16()
model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-4), loss=np_loss)
a = model(np.random.random((1, 64, 16, 16, 3)));
model.summary()


# In[4]:


batch_size = 128

rlap_tape = "/root/ssd_cache/rppg_training_data/rlap_64x16x16_all"

train_rlap = load_datatape(rlap_tape, fold='train', batch=batch_size)
valid_rlap = load_datatape(rlap_tape, fold='val', extended_hr='False', batch=batch_size)

#train_rlap = DatatapeMonitor(train_rlap)
training_set, validation_set = KerasDataset(train_rlap), KerasDataset(valid_rlap)

training_set = training_set.apply_fn(compress_aug)


# In[5]:


train(model, training_set, validation_set, epochs=20, check_point_path='../weights/physnet16.weights.h5')
model.load_weights('../weights/physnet16.weights.h5')


# In[6]:


eval_on_dataset(dataset_H5_mmpd, pmodel(model), 64, (16, 16), step=2, batch=16, save='../results/PhysNet16_RLAP_MMPD.h5')
get_metrics('../results/PhysNet16_RLAP_MMPD.h5', dropped='False')


# In[7]:


eval_on_dataset(dataset_H5_cohface, pmodel(model), 64, (16, 16), step=2, batch=16, save='../results/PhysNet16_RLAP_COHFACE.h5', fps=30)
get_metrics('../results/PhysNet16_RLAP_COHFACE.h5')


# In[8]:


eval_on_dataset(dataset_H5_pure, pmodel(model), 64, (16, 16), step=2, batch=16, save='../results/PhysNet16_RLAP_PURE.h5')
get_metrics('../results/PhysNet16_RLAP_PURE.h5')


# In[9]:


eval_on_dataset(dataset_H5_ubfc_rppg2, pmodel(model), 64, (16, 16), step=2, batch=16, save='../results/PhysNet16_RLAP_UBFC.h5')
get_metrics('../results/PhysNet16_RLAP_UBFC.h5')


# In[10]:


model = PhysNet16(TN=True)
model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-4), loss=np_loss)
a = model(np.random.random((1, 64, 16, 16, 3)));
model.summary()


# In[11]:


train(model, training_set, validation_set, epochs=20, check_point_path='../weights/physnet16tn.weights.h5')
model.load_weights('../weights/physnet16tn.weights.h5')


# In[12]:


eval_on_dataset(dataset_H5_mmpd, pmodel(model), 64, (16, 16), step=2, batch=16, save='../results/PhysNet16TN_RLAP_MMPD.h5')
get_metrics('../results/PhysNet16TN_RLAP_MMPD.h5', dropped='False')


# In[13]:


eval_on_dataset(dataset_H5_cohface, pmodel(model), 64, (16, 16), step=2, batch=16, save='../results/PhysNet16TN_RLAP_COHFACE.h5', fps=30)
get_metrics('../results/PhysNet16TN_RLAP_COHFACE.h5')


# In[14]:


eval_on_dataset(dataset_H5_pure, pmodel(model), 64, (16, 16), step=2, batch=16, save='../results/PhysNet16TN_RLAP_PURE.h5')
get_metrics('../results/PhysNet16TN_RLAP_PURE.h5')


# In[15]:


eval_on_dataset(dataset_H5_ubfc_rppg2, pmodel(model), 64, (16, 16), step=2, batch=16, save='../results/PhysNet16TN_RLAP_UBFC.h5')
get_metrics('../results/PhysNet16TN_RLAP_UBFC.h5')


# In[16]:


model = PhysNet8()
model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-4), loss=np_loss)
a = model(np.random.random((4, 32, 8, 8, 3)));
model.summary()


# In[17]:


batch_size = 128

rlap_tape = "/root/ssd_cache/rppg_training_data/rlap_32x8x8_all"

train_rlap = load_datatape(rlap_tape, fold='train', batch=batch_size)
valid_rlap = load_datatape(rlap_tape, fold='val', extended_hr='False', batch=batch_size)

#train_rlap = DatatapeMonitor(train_rlap)
training_set, validation_set = KerasDataset(train_rlap), KerasDataset(valid_rlap)

training_set = training_set.apply_fn(compress_aug)


# In[18]:


train(model, training_set, validation_set, epochs=20, check_point_path='../weights/physnet8.weights.h5')
model.load_weights('../weights/physnet8.weights.h5')


# In[19]:


eval_on_dataset(dataset_H5_mmpd, pmodel(model), 32, (8, 8), step=1, batch=16, save='../results/PhysNet8_RLAP_MMPD.h5')
get_metrics('../results/PhysNet8_RLAP_MMPD.h5', dropped='False')


# In[20]:


eval_on_dataset(dataset_H5_cohface, pmodel(model), 32, (8, 8), step=1, batch=16, save='../results/PhysNet8_RLAP_COHFACE.h5')
get_metrics('../results/PhysNet8_RLAP_COHFACE.h5')


# In[21]:


eval_on_dataset(dataset_H5_pure, pmodel(model), 32, (8, 8), step=1, batch=16, save='../results/PhysNet8_RLAP_PURE.h5')
get_metrics('../results/PhysNet8_RLAP_PURE.h5')


# In[22]:


eval_on_dataset(dataset_H5_ubfc_rppg2, pmodel(model), 32, (8, 8), step=1, batch=16, save='../results/PhysNet8_RLAP_UBFC.h5')
get_metrics('../results/PhysNet8_RLAP_UBFC.h5')


# In[23]:


model = PhysNet8(TN=True)
model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-4), loss=np_loss)
a = model(np.random.random((4, 32, 8, 8, 3)));
model.summary()


# In[24]:


train(model, training_set, validation_set, epochs=20, check_point_path='../weights/physnet8tn.weights.h5')
model.load_weights('../weights/physnet8tn.weights.h5')


# In[25]:


eval_on_dataset(dataset_H5_mmpd, pmodel(model), 32, (8, 8), step=1, batch=16, save='../results/PhysNet8TN_RLAP_MMPD.h5')
get_metrics('../results/PhysNet8TN_RLAP_MMPD.h5', dropped='False')


# In[26]:


eval_on_dataset(dataset_H5_cohface, pmodel(model), 32, (8, 8), step=1, batch=16, save='../results/PhysNet8TN_RLAP_COHFACE.h5')
get_metrics('../results/PhysNet8TN_RLAP_COHFACE.h5')


# In[27]:


eval_on_dataset(dataset_H5_pure, pmodel(model), 32, (8, 8), step=1, batch=16, save='../results/PhysNet8TN_RLAP_PURE.h5')
get_metrics('../results/PhysNet8TN_RLAP_PURE.h5')


# In[28]:


eval_on_dataset(dataset_H5_ubfc_rppg2, pmodel(model), 32, (8, 8), step=1, batch=16, save='../results/PhysNet8TN_RLAP_UBFC.h5')
get_metrics('../results/PhysNet8TN_RLAP_UBFC.h5')


# In[29]:


model = PhysNet4()
model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-4), loss=np_loss)
a = model(np.random.random((4, 32, 4, 4, 3)));
model.summary()


# In[30]:


batch_size = 128

rlap_tape = "/root/ssd_cache/rppg_training_data/rlap_32x4x4_all"

train_rlap = load_datatape(rlap_tape, fold='train', batch=batch_size)
valid_rlap = load_datatape(rlap_tape, fold='val', extended_hr='False', batch=batch_size)

#train_rlap = DatatapeMonitor(train_rlap)
training_set, validation_set = KerasDataset(train_rlap), KerasDataset(valid_rlap)

training_set = training_set.apply_fn(compress_aug)


# In[31]:


train(model, training_set, validation_set, epochs=20, check_point_path='../weights/physnet4.weights.h5', cut_nan=True)
model.load_weights('../weights/physnet4.weights.h5')


# In[32]:


eval_on_dataset(dataset_H5_mmpd, pmodel(model), 32, (4, 4), step=1, batch=16, save='../results/PhysNet4_RLAP_MMPD.h5')
get_metrics('../results/PhysNet4_RLAP_MMPD.h5', dropped='False')


# In[33]:


eval_on_dataset(dataset_H5_cohface, pmodel(model), 32, (4, 4), step=1, batch=16, save='../results/PhysNet4_RLAP_COHFACE.h5')
get_metrics('../results/PhysNet4_RLAP_COHFACE.h5')


# In[34]:


eval_on_dataset(dataset_H5_pure, pmodel(model), 32, (4, 4), step=1, batch=16, save='../results/PhysNet4_RLAP_PURE.h5')
get_metrics('../results/PhysNet4_RLAP_PURE.h5')


# In[35]:


eval_on_dataset(dataset_H5_ubfc_rppg2, pmodel(model), 32, (4, 4), step=1, batch=16, save='../results/PhysNet4_RLAP_UBFC.h5')
get_metrics('../results/PhysNet4_RLAP_UBFC.h5')


# In[36]:


model = PhysNet4(TN=True)
model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-4), loss=np_loss)
a = model(np.random.random((4, 32, 4, 4, 3)));
model.summary()


# In[37]:


train(model, training_set, validation_set, epochs=20, check_point_path='../weights/physnet4tn.weights.h5')
model.load_weights('../weights/physnet4tn.weights.h5')


# In[38]:


eval_on_dataset(dataset_H5_mmpd, pmodel(model), 32, (4, 4), step=1, batch=16, save='../results/PhysNet4TN_RLAP_MMPD.h5')
get_metrics('../results/PhysNet4TN_RLAP_MMPD.h5', dropped='False')


# In[39]:


eval_on_dataset(dataset_H5_cohface, pmodel(model), 32, (4, 4), step=1, batch=16, save='../results/PhysNet4TN_RLAP_COHFACE.h5')
get_metrics('../results/PhysNet4TN_RLAP_COHFACE.h5')


# In[40]:


eval_on_dataset(dataset_H5_pure, pmodel(model), 32, (4, 4), step=1, batch=16, save='../results/PhysNet4TN_RLAP_PURE.h5')
get_metrics('../results/PhysNet4TN_RLAP_PURE.h5')


# In[41]:


eval_on_dataset(dataset_H5_ubfc_rppg2, pmodel(model), 32, (4, 4), step=1, batch=16, save='../results/PhysNet4TN_RLAP_UBFC.h5')
get_metrics('../results/PhysNet4TN_RLAP_UBFC.h5')


# In[42]:


import matplotlib.pyplot as plt
from matplotlib.patches import Circle

# 创建一个图形和坐标轴
fig, ax = plt.subplots()

# 创建一个圆，圆心在 (0.5, 0.5)，半径为 0.2
circle = Circle((0.5, 0.5), 0.2, facecolor='blue')

# 将圆添加到坐标轴
ax.add_patch(circle)

# 设置坐标轴的范围和纵横比
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_aspect('equal')

# 显示图形
plt.show()


# In[178]:


import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator
from matplotlib.ticker import ScalarFormatter
import matplotlib.patheffects as path_effects

plt.rcParams['font.family'] = 'DeJavu Serif'
plt.rcParams['font.serif'] = ['Times New Roman']

# 数据
models = ['PhysNet', 'PhysNet TN']
params = np.array([10121, 59441, 256721, 769825])/1000
mae_model_a = [25.5, 16.6, 9.94, 6.09]
mae_model_b = [3.59, 3.22, 2.80, 3.13]


# 绘制折线图
plt.figure(figsize=(10, 6))
plt.plot(params, mae_model_a, marker='o', label='PhysNet', color='#FA7F6F')
plt.plot(params, mae_model_b, marker='o', label='PhysNet TN', color='#82B0D2')

# 标注数据点
for i, (param, name) in enumerate(zip(params, ['Nano','Small','Medium','Large'])):
    plt.text(param, mae_model_a[i], f'{name}', fontsize=16, color='#FA7F6F').set_path_effects([path_effects.Stroke(linewidth=1, foreground='gray'), path_effects.Normal()])
    plt.text(param, mae_model_b[i], f'{name}', fontsize=16, color='#82B0D2').set_path_effects([path_effects.Stroke(linewidth=1, foreground='gray'), path_effects.Normal()])

plt.xscale('log')
plt.yscale('log')
plt.gca().xaxis.set_major_formatter(ScalarFormatter())
#plt.gca().xaxis.set_major_locator(LogLocator(base=10, subs='all'))
plt.gca().yaxis.set_major_formatter(ScalarFormatter())
plt.gca().yaxis.set_major_locator(LogLocator(base=10.0, subs='auto', numticks=10))
plt.grid(True, which='both', linestyle='--', linewidth=0.4, alpha=0.7)

plt.xlim((7, 1.3e3))

# 添加标题和标签
#plt.title('Parameter Count vs Mean Absolute Error (MAE)')
plt.xlabel('Parameters (K)', fontsize=16)
plt.ylabel('Average MAE of HR (BPM)', fontsize=16)
plt.legend(fontsize=16)

# 显示图表
plt.savefig('plot.png', dpi=300)
plt.show()


# In[177]:


import torch
import torch.nn as nn
import torch.nn.functional as F

class TNM(nn.Module):
    def __init__(self, enabled=True, frames=0, axis=2, eps=1e-6):
        super(TNM, self).__init__()
        self.enabled = enabled
        self.frames = frames
        self.axis = axis
        self.eps = eps

    def forward(self, x):
        if self.enabled:
            dtype = x.dtype
            x_ = x.to(torch.float32)
            x_ = x_.reshape((*x.shape[:self.axis], -1, self.frames, *x.shape[self.axis+1:]))
            
            mean = x_.mean(dim=self.axis + 1, keepdim=True)
            tshape = [1] * len(x_.shape)
            tshape[self.axis + 1] = self.frames
            t = torch.linspace(0, 1, self.frames).reshape(tshape).to(x.device)
            
            n = ((t - 0.5) * (x_ - mean)).sum(dim=self.axis + 1, keepdim=True)
            d = ((t - 0.5) ** 2).sum(dim=self.axis + 1, keepdim=True)
            i = mean - n / d * 0.5
            trend = n / d * t + i
            x_ = x_ - trend
            
            std = ((x_ ** 2).mean(dim=self.axis + 1, keepdim=True) + self.eps).sqrt()
            x_ = x_ / std
            
            x_ = x_.reshape(x.shape)
            return x_.to(dtype)
        else:
            return x

# 示例使用
input_tensor = torch.randn(4, 3, 10, 32, 32).cuda()  # 假设输入是一个10帧的视频，每帧大小为32x32
tnm_layer = TNM(enabled=True, frames=10, axis=2)
output_tensor = tnm_layer(input_tensor)
print(output_tensor.shape)


# In[ ]:





import os
os.environ["KERAS_BACKEND"] = "jax"
#os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.9'
#os.environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = 'platform'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import keras
from keras import ops
from keras import layers
from jax import numpy as jnp
import jax

#data_parallel = keras.distribution.DataParallel()
#keras.distribution.set_distribution(data_parallel)

pdf = lambda x,u=0,sig=1:ops.exp(-(x-u)**2/2/sig**2)/(sig*((2*3.1415926535)**0.5))

class Attention_mask(layers.Layer):
    
    def call(self, x):
        return 0.5*x/ops.mean(x, axis=(1, 2), keepdims=True)


class TSM(layers.Layer):
    def __init__(self, n, frames=160):
        super().__init__()
        self.n = n
        self.frames = frames

    def call(self, x, fold_div=3):
        if self.n==0:
            return x
        shape = x.shape
        x = ops.reshape(x, (-1, self.frames, *x.shape[1:]))
        b, nt, *c = x.shape
        x = ops.reshape(x, (b, -1, self.n, *c))
        fold = c[-1] // fold_div
        out = ops.concatenate([ops.concatenate([x[:, :, 1:, ..., :fold], x[:, :, -1:, ..., :fold]], axis=2),
                        ops.concatenate([x[:, :, :1, ..., fold:fold*2], x[:, :, :-1, ..., fold:fold*2]], axis=2),
                        x[:, :, :, ..., fold*2:]], axis=-1)
        return ops.reshape(out, shape)


'''
class TSM(layers.Layer):
    def __init__(self, n, frames=160):
        super().__init__()
        self.n = n
        self.frames = frames

    def call(self, x, fold_div=4):
        if self.n==0:
            return x
        shape = x.shape
        x = ops.reshape(x, (-1, self.frames, *x.shape[1:]))
        b, nt, *c = x.shape
        x = ops.reshape(x, (b, -1, self.n, *c))
        fold = c[-1] // fold_div
        out = ops.concatenate([ops.concatenate([x[:, :, [0]*i, ..., fold*i:fold*(i+1)], x[:, :, :nt-(i%fold_div), ..., fold*i:fold*(i+1)]], axis=2) for i in range(fold_div)], axis=-1)
        return ops.reshape(out, shape)
'''
'''
class TSM(layers.Layer):
    def __init__(self, n, frames=160):
        super().__init__()
        self.n = n
        self.frames = frames

    def call(self, x, fold_div=3):
        if self.n==0:
            return x
        shape = x.shape
        x = ops.reshape(x, (-1, self.frames, *x.shape[1:]))
        b, nt, *c = x.shape
        x = ops.reshape(x, (b, -1, self.n, *c))
        fold = c[-1] // fold_div
        out = ops.concatenate([ops.concatenate([x[:, :, [1, 0], ..., :fold], x[:, :, :-2, ..., :fold]], axis=2),
                        ops.concatenate([x[:, :, :1, ..., fold:fold*2], x[:, :, :-1, ..., fold:fold*2]], axis=2),
                        x[:, :, :, ..., fold*2:]], axis=-1)
        return ops.reshape(out, shape)
'''
class TNM(layers.Layer):

    def __init__(self, enabled=True, frames=0, axis=0, eps=1e-6):
        super().__init__()
        def norm(x):
            if not frames:
                _frames = x.shape[axis]
            else:
                _frames = frames
            dtype = x.dtype
            x_ = ops.cast(x, 'float32')
            x_ = ops.reshape(x, (*x.shape[:axis], -1, _frames, *x.shape[axis+1:]))
            mean = ops.mean(x_, axis=axis+1, keepdims=True)
            tshape = [1]*len(x_.shape)
            tshape[axis+1] = _frames
            t = ops.reshape(ops.linspace(0, 1, _frames), tshape)
            n = ops.sum((t-0.5)*(x_-mean), axis=axis+1, keepdims=True)
            d = ops.sum((t-0.5)**2, axis=axis+1, keepdims=True)
            i = mean - n/d * 0.5
            trend = n/d * t + i
            x_ = x_ - trend
            # mean = ops.mean(x_, axis=axis+1, keepdims=True)
            mean = 0
            std = (ops.mean((x_-mean)**2, axis=axis+1, keepdims=True)+eps)**0.5
            x_ = (x_-mean)/std
                
            r = ops.reshape(x_, (*x.shape[:axis], -1, *x.shape[axis+1:])) 
            return ops.cast(r, dtype)
        if enabled:
            self.n = norm
        else:
            self.n = lambda x:x
    
    def call(self, x):
        return self.n(x)


class CHP(layers.Layer):
    def __init__(self, size=2):
        super().__init__()
        self.size = size
        self.pdf = lambda x,u=0,sig=1:ops.exp(-(x-u)**2/2/sig**2)/(sig*((2*3.1415926535)**0.5))
        self.H = lambda x,axis=None,no_bias=True:-ops.mean((lambda i:i*ops.log2(i))(self.pdf(x-int(no_bias)*x.mean(axis=axis,keepdims=1))),axis=axis,keepdims=1)

    def call(self, x):
        dtype = x.dtype
        x = ops.cast(x, 'float32')
        x = ops.reshape(x, (*x.shape[:-1], -1, self.size))
        dims = len(x.shape)
        axes = [i for i in range(dims) if i not in (-1%dims, 0)]
        CH = self.H(x, axes)
        idx = ops.argsort(CH, axis=-1)
        idx = idx[...,[0]]
        x = ops.take_along_axis(x, idx, axis=-1)
        x = x[...,0]
        return ops.cast(x, dtype)


class SAN(keras.Model):

    def __init__(self, input_frames=160, TN=False, TS=False, depth=2):  
        super().__init__()
        self.depth = depth 
        n = input_frames//int(TS) if TS else 0 

        self.sample = [keras.Sequential([
            layers.Conv2D(2**(4+depth-_), (2, 2), (2, 2)),
        ], name=f'Sample {_}') for _ in range(depth)]
        
        self.d1 = [keras.Sequential([ 
            TNM(TN, input_frames),
            layers.Conv2D(2**(3+depth-_), (3, 3), padding='same', activation='tanh'),
            TSM(n, input_frames),
            TNM(TN, input_frames),
            layers.Conv2D(2**(3+depth-_), (3, 3), padding='same', activation='tanh'),
            TSM(n, input_frames),
        ], name=f'Conv1 {_}') for _ in range(depth)]
        
        self.d2 = [keras.Sequential([
            TNM(TN, input_frames),
            layers.Conv2D(2**(4+depth-_), (3, 3), padding='same', activation='tanh'),
            TSM(n, input_frames),
            TNM(TN, input_frames),
            layers.Conv2D(2**(4+depth-_), (3, 3), padding='same', activation='tanh'),
            TSM(n, input_frames),
        ], name=f'Conv2 {_}') for _ in range(depth)]
        
        self.g1 = [keras.Sequential([
            layers.Conv2D(2**(2+depth-_), (3, 3), padding='same', activation='tanh'),
            layers.Conv2D(2**(2+depth-_), (3, 3), padding='same', activation='tanh'),
        ], name=f'AttnConv1 {_}') for _ in range(depth)]
        
        self.g2 = [keras.Sequential([
            layers.Conv2D(2**(3+depth-_), (3, 3), padding='same', activation='tanh'),
            layers.Conv2D(2**(3+depth-_), (3, 3), padding='same', activation='tanh'),
        ], name=f'AttnConv2 {_}') for _ in range(depth)]
        
        self.attn1 = [keras.Sequential([
            layers.Conv2D(1, (1, 1), activation='relu', dtype='float32'),
        ], name=f'Mask1 {_}') for _ in range(depth)]
        
        self.attn2 = [keras.Sequential([
            layers.Conv2D(1, (1, 1), activation='relu', dtype='float32'),
        ], name=f'Mask2 {_}') for _ in range(depth)]
        
        self.pd = keras.Sequential([
            layers.AvgPool2D((2, 2)),
        ], name='Pooling & Downsample')

        self.da = keras.Sequential([
            layers.Flatten(),
            TNM(TN, input_frames),
            layers.Dense(8, activation='tanh', activity_regularizer=keras.regularizers.l2(0.1))])
        
        self.db = keras.Sequential([
            TSM(n, input_frames),
            layers.Dense(1, dtype='float32'),
        ], name='Flatten & Dense')

    def call(self, x, training=None):
        d = ops.reshape(x, (-1, *x.shape[2:]))
        for i in range(self.depth):
            d0 = self.sample[i](d)
            d = self.d1[i](d, training=training)
            g = self.g1[i](d)
            d = d*self.attn1[i](g)
            d = self.pd(d, training=training)
            d = self.d2[i](d, training=training)
            g = self.g2[i](d)
            d = d*self.attn2[i](g)
            d = d + d0
        d = self.pd(d, training=training) 
        d = self.da(d, training=training) 
        d = self.db(d, training=training) 
        return ops.reshape(d, (-1, x.shape[1])) 
'''
class SAN(keras.Model):

    def __init__(self, input_frames=160, TN=False, TS=False, depth=2):  
        super().__init__()
        self.depth = depth 
        n = input_frames//int(TS) if TS else 0 

        self.sample = [keras.Sequential([
            layers.Conv3D(2**(4+depth-_), (1, 2, 2), (1, 2, 2)),
        ], name=f'Sample {_}') for _ in range(depth)]

        self.d0 = keras.Sequential([ 
            layers.Conv3D(2**(3+depth), (1, 1, 1)),
        ], name=f'Conv0')
        
        self.d1 = [keras.Sequential([ 
            TNM(TN, axis=1),
            #layers.Conv3D(2**(3+depth-_), (3, 1, 1), padding='same'),
            layers.Conv3D(2**(3+depth-_), (3, 3, 3), padding='same', activation='relu'),
            TNM(TN, axis=1),
            #layers.Conv3D(2**(3+depth-_), (3, 1, 1), padding='same'),
            layers.Conv3D(2**(3+depth-_), (3, 3, 3), padding='same', activation='relu'),
        ], name=f'Conv1 {_}') for _ in range(depth)]
        
        self.d2 = [keras.Sequential([
            TNM(TN, axis=1),
            #layers.Conv3D(2**(4+depth-_), (3, 1, 1), padding='same'),
            layers.Conv3D(2**(4+depth-_), (3, 3, 3), padding='same', activation='relu'),
            TNM(TN, axis=1),
            #layers.Conv3D(2**(4+depth-_), (3, 1, 1), padding='same'),
            layers.Conv3D(2**(4+depth-_), (3, 3, 3), padding='same', activation='relu'),
        ], name=f'Conv2 {_}') for _ in range(depth)]
        
        self.g1 = [keras.Sequential([
            layers.Conv3D(2**(2+depth-_), (1, 3, 3), padding='same', activation='relu'),
            layers.Conv3D(2**(2+depth-_), (1, 3, 3), padding='same', activation='relu'),
        ], name=f'AttnConv1 {_}') for _ in range(depth)]
        
        self.g2 = [keras.Sequential([
            layers.Conv3D(2**(3+depth-_), (1, 3, 3), padding='same', activation='relu'),
            layers.Conv3D(2**(3+depth-_), (1, 3, 3), padding='same', activation='relu'),
        ], name=f'AttnConv2 {_}') for _ in range(depth)]
        
        self.attn1 = [keras.Sequential([
            layers.Conv3D(1, (1, 1, 1), activation='relu', dtype='float32'),
        ], name=f'Mask1 {_}') for _ in range(depth)]
        
        self.attn2 = [keras.Sequential([
            layers.Conv3D(1, (1, 1, 1), activation='relu', dtype='float32'),
        ], name=f'Mask2 {_}') for _ in range(depth)]
        
        self.pd = keras.Sequential([
            layers.AvgPool3D((1, 2, 2)),
            layers.SpatialDropout3D(0.25)
        ], name=f'Pooling & Downsample')
        
        self.d = keras.Sequential([
            TNM(TN, axis=1),
            #TSM(n, input_frames),
            layers.Dense(128, activation='tanh'),
            layers.Dropout(0.5),
            TNM(TN, axis=1),
            #TSM(n, input_frames),
            layers.Dense(1, dtype='float32')
        ], name='Flatten & Dense')

    def call(self, x, training=None):
        #d = ops.reshape(x, (-1, *x.shape[2:]))
        d = self.d0(x)
        for i in range(self.depth):
            d0 = self.sample[i](d)
            d = self.d1[i](d)
            g = self.g1[i](d)
            d = d*self.attn1[i](g)
            d = self.pd(d, training=training)
            d = self.d2[i](d)
            g = self.g2[i](d)
            d = d*self.attn2[i](g)
            d = d + d0
        d = self.pd(d, training=training) 
        d = ops.reshape(d, (d.shape[0], d.shape[1], -1))
        d = self.d(d, training=training)[..., 0]
        #return ops.reshape(d, (-1, x.shape[1]))
        return d
'''

class TSCAN(keras.Model):

    def __init__(self, input_frames=160, TN=False):
        super().__init__()
        n = input_frames//8
        self.d1 = keras.Sequential([
            TSM(n),
            TNM(TN, axis=0, frames=input_frames),
            layers.Conv2D(32, (3, 3), padding='same', activation='tanh'),
            TSM(n),
            TNM(TN, axis=0, frames=input_frames),
            layers.Conv2D(32, (3, 3), activation='tanh'),
        ])
        self.d2 = keras.Sequential([
            TSM(n),
            TNM(TN, axis=0, frames=input_frames),
            layers.Conv2D(64, (3, 3), padding='same', activation='tanh'),
            TSM(n),
            TNM(TN, axis=0, frames=input_frames),
            layers.Conv2D(64, (3, 3), activation='tanh'),
        ])
        self.g1 = keras.Sequential([
            layers.Conv2D(32, (3, 3), padding='same', activation='tanh'),
            layers.Conv2D(32, (3, 3), activation='tanh'),
        ])
        self.g2 = keras.Sequential([
            layers.Conv2D(64, (3, 3), padding='same', activation='tanh'),
            layers.Conv2D(64, (3, 3), activation='tanh'),
        ])
        self.attn1 = keras.Sequential([
            layers.Conv2D(1, (1, 1), activation='sigmoid'),
            Attention_mask()
        ])
        self.attn2 = keras.Sequential([
            layers.Conv2D(1, (1, 1), activation='sigmoid'),
            Attention_mask()
        ])
        self.pd = keras.Sequential([
            layers.AveragePooling2D((2, 2)),
            layers.Dropout(0.25)
        ])
        self.d = keras.Sequential([
            layers.Flatten(),
            layers.Dense(128, activation='tanh'),
            layers.Dropout(0.5),
            layers.Dense(1)
        ])

    def call(self, x, training=None, return_attn=False):
        d, g = x 
        b = d.shape[0]
        d = ops.reshape(d, (-1, *d.shape[2:]))
        g = ops.reshape(g, (-1, *g.shape[2:]))
        d = self.d1(d)
        g = self.g1(g)
        attn1 = self.attn1(g)
        d = ops.stack(ops.split(d, b))*ops.stack(ops.split(self.attn1(g), b))
        d = ops.reshape(d, (-1, *d.shape[2:]))
        d = self.pd(d, training=training)
        d = self.d2(d)
        g = self.pd(g, training=training)
        g = self.g2(g)
        attn2 = self.attn2(g)
        d = ops.stack(ops.split(d, b))*ops.stack(ops.split(self.attn2(g), b))
        d = ops.reshape(d, (-1, *d.shape[2:]))
        d = self.pd(d, training=training)
        d = self.d(d, training=training)
        if return_attn:
            return ops.reshape(d, (-1, x[0].shape[1])), attn1, attn2
        return ops.reshape(d, (-1, x[0].shape[1]))
    
class EfficientPhys(keras.Model):
    def __init__(self, input_frames=160, TN=False):
        super().__init__()
        self.TSM = TSM(8, input_frames)
        self.mc1 = layers.Conv2D(32, kernel_size=3, padding='same', activation='tanh')
        self.mc2 = layers.Conv2D(32, kernel_size=3, activation='tanh')
        self.mc3 = layers.Conv2D(64, kernel_size=3, padding='same', activation='tanh')
        self.mc4 = layers.Conv2D(64, kernel_size=3, activation='tanh')
        self.attc1 = layers.Conv2D(32, kernel_size=1, activation='sigmoid')
        self.msk = Attention_mask()
        self.attc2 = layers.Conv2D(64, kernel_size=1, activation='sigmoid')
        self.avgp = layers.AvgPool2D(2)
        self.dp1 = layers.Dropout(0.25)
        self.dp2 = layers.Dropout(0.5)
        self.dense1 = layers.Dense(128, activation='tanh')
        self.dense2 = layers.Dense(1)
        self.bn = layers.BatchNormalization(axis=1)
        self.ft = layers.Flatten()
        self.TN = TN
        self.IF = input_frames

    def call(self, y, training=None, return_attn=False):
        x = ops.reshape(y, (-1, *y.shape[2:]))
        if not self.TN:
            x = x[1:]-x[:-1]
            x = ops.concatenate([x, ops.zeros((1, *x.shape[-3:]))], axis=0)
        x = ops.reshape(x, (x.shape[0], -1, x.shape[3]))
        x = self.bn(x, training=training)
        x = ops.reshape(x, (-1, *y.shape[2:]))
        x = self.TSM(x)
        x = TNM(self.TN, axis=0, frames=self.IF)(x)
        x = self.mc1(x)
        x = self.TSM(x)
        x = TNM(self.TN, axis=0, frames=self.IF)(x)
        x = self.mc2(x)
        msk1 = self.msk(self.attc1(x))
        x = layers.multiply([x, msk1])
        x = self.avgp(x)
        x = self.dp1(x, training=training)
        x = self.TSM(x)
        x = TNM(self.TN, axis=0, frames=self.IF)(x)
        x = self.mc3(x)
        x = self.TSM(x)
        x = TNM(self.TN, axis=0, frames=self.IF)(x)
        x = self.mc4(x)
        msk2 = self.msk(self.attc2(x))
        x = layers.multiply([x, msk2])
        x = self.avgp(x)
        x = self.dp1(x, training=training)
        #x = ops.reshape(x, (-1, ops.reduce_prod(x.get_shape()[1:])))
        x = self.ft(x)
        x = self.dense1(x)
        x = self.dp2(x, training=training)
        x = self.dense2(x)
        if return_attn:
            return ops.reshape(x, (-1, y.shape[1])), msk1, msk2
        return ops.reshape(x, (-1, y.shape[1]))
    
class PhysNet(keras.Model):

    def __init__(self, TN=False):
        super().__init__()
        self.TN = TN
        self.ConvBlock1 = keras.Sequential([
            layers.Conv3D(16, kernel_size=(1, 5, 5), strides=1, padding='same'),
            layers.BatchNormalization(),
            layers.Activation('relu')
        ])
        self.ConvBlock2 = keras.Sequential([
            layers.Conv3D(32, kernel_size=(3, 3, 3), strides=1, padding='same'),
            layers.BatchNormalization(),
            layers.Activation('relu')
        ])
        self.ConvBlock3 = keras.Sequential([
            layers.Conv3D(64, kernel_size=(3, 3, 3), strides=1, padding='same'),
            layers.BatchNormalization(),
            layers.Activation('relu')
        ])
        self.ConvBlock4 = keras.Sequential([
            layers.Conv3D(64, kernel_size=(3, 3, 3), strides=1, padding='same'),
            layers.BatchNormalization(),
            layers.Activation('relu')
        ])
        self.ConvBlock5 = keras.Sequential([
            layers.Conv3D(64, kernel_size=(3, 3, 3), strides=1, padding='same'),
            layers.BatchNormalization(),
            layers.Activation('relu')
        ])
        self.ConvBlock6 = keras.Sequential([
            layers.Conv3D(64, kernel_size=(3, 3, 3), strides=1, padding='same'),
            layers.BatchNormalization(),
            layers.Activation('relu')
        ])
        self.ConvBlock7 = keras.Sequential([
            layers.Conv3D(64, kernel_size=(3, 3, 3), strides=1, padding='same'),
            layers.BatchNormalization(),
            layers.Activation('relu')
        ])
        self.ConvBlock8 = keras.Sequential([
            layers.Conv3D(64, kernel_size=(3, 3, 3), strides=1, padding='same'),
            layers.BatchNormalization(),
            layers.Activation('relu')
        ])
        self.ConvBlock9 = keras.Sequential([
            layers.Conv3D(64, kernel_size=(3, 3, 3), strides=1, padding='same'),
            layers.BatchNormalization(),
            layers.Activation('relu')
        ])
        self.upsample = keras.Sequential([
            layers.Conv3DTranspose(64, kernel_size=(4, 1, 1), strides=(2, 1, 1), padding='same'),
            layers.BatchNormalization(),
            layers.Activation('elu')
        ])
        self.upsample2 = keras.Sequential([
            layers.Conv3DTranspose(64, kernel_size=(4, 1, 1), strides=(2, 1, 1), padding='same'),
            layers.BatchNormalization(),
            layers.Activation('elu')
        ])
        self.convBlock10 = layers.Conv3D(1, kernel_size=(1, 1, 1), strides=1)
        self.MaxpoolSpa = layers.MaxPool3D((1, 2, 2), strides=(1, 2, 2))
        self.MaxpoolSpaTem = layers.MaxPool3D((2, 2, 2), strides=2)
        self.poolspa = layers.AvgPool3D((1, 2, 2))
        self.flatten = layers.Reshape((-1,))

    def call(self, x, training=None):
        x = TNM(self.TN, axis=1, frames=x.shape[1])(x)
        x = self.ConvBlock1(x, training=training)
        x = self.MaxpoolSpa(x)
        x = TNM(self.TN, axis=1, frames=x.shape[1])(x)
        x = self.ConvBlock2(x, training=training)
        x = TNM(self.TN, axis=1, frames=x.shape[1])(x)
        x = self.ConvBlock3(x, training=training)
        x = self.MaxpoolSpaTem(x)
        x = TNM(self.TN, axis=1, frames=x.shape[1])(x)
        x = self.ConvBlock4(x, training=training)
        x = TNM(self.TN, axis=1, frames=x.shape[1])(x)
        x = self.ConvBlock5(x, training=training)
        x = self.MaxpoolSpaTem(x)
        x = TNM(self.TN, axis=1, frames=x.shape[1])(x)
        x = self.ConvBlock6(x, training=training)
        x = TNM(self.TN, axis=1, frames=x.shape[1])(x)
        x = self.ConvBlock7(x, training=training)
        x = self.MaxpoolSpa(x)
        x = TNM(self.TN, axis=1, frames=x.shape[1])(x)
        x = self.ConvBlock8(x, training=training)
        x = TNM(self.TN, axis=1, frames=x.shape[1])(x)
        x = self.ConvBlock9(x, training=training)
        x = self.upsample(x, training=training)
        x = self.upsample2(x, training=training)
        x = self.poolspa(x)
        x = self.convBlock10(x, training=training)
        x = self.flatten(x)
        x = (x-ops.mean(x, axis=1, keepdims=True))/ops.std(x, axis=1, keepdims=True)
        return x

class CDC(layers.Layer):
    def __init__(self, filters, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same', theta=0.6, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.theta = theta

    def build(self, input_shape):
        self.kernel = self.add_weight(shape=self.kernel_size + (input_shape[-1], self.filters)) # (D, H, W, I, O)
        super().build(input_shape)

    def call(self, inputs):
        if self.kernel_size[0] == 3:
            conv_out = ops.nn.conv(inputs, self.kernel, strides=self.strides, padding=self.padding.upper())
            tdc_kernel = ops.sum(self.kernel[ops.array([0, -1])], axis=(0, 1, 2), keepdims=True)
            diff_out = ops.nn.conv(inputs, tdc_kernel, strides=self.strides, padding=self.padding.upper())
            return conv_out - self.theta*diff_out
        else:
            return ops.nn.conv(inputs, self.kernel, strides=self.strides, padding=self.padding.upper())
        
class MultiHeadedSelfAttention_TDC_gra_sharp(keras.Layer):
    def __init__(self, ch, num_heads, dropout=0.1, theta=0.7):
        super().__init__()
        
        self.proj_q = keras.Sequential([
            CDC(ch, (3, 3, 3), theta=theta, padding='same'),  
            layers.BatchNormalization(),
        ])
        self.proj_k = keras.Sequential([
            CDC(ch, (3, 3, 3), theta=theta, padding='same'),  
            layers.BatchNormalization(),
        ])
        self.proj_v = keras.Sequential([
            layers.Conv3D(ch, (1, 1, 1), use_bias=False),
        ])
        
        self.drop = layers.Dropout(dropout)
        self.n_heads = num_heads

    def call(self, x, gra_sharp=2.0, training=None):
        B, P, C = x.shape # Batch, 640, 96
        x = ops.reshape(x, (B, P//16, 4, 4, C))
        q, k, v = self.proj_q(x), self.proj_k(x), self.proj_v(x)
        q, k, v = [ops.transpose(layers.Reshape((P, self.n_heads, -1))(i), (0, 2, 1, 3)) for i in (q, k, v)] # Batch, Heads, 640, Channel//Heads
        scores = q @ ops.transpose(k, (0, 1, 3, 2)) / gra_sharp
        scores = ops.cast(scores, 'float32')
        scores = self.drop(ops.softmax(scores, axis=-1), training=training)
        h = layers.Reshape((P, -1))(ops.transpose(scores @ v, (0, 2, 1, 3)))
        
        return h, scores
    
class PositionWiseFeedForward_ST(keras.Layer):
    def __init__(self, ich, och):
        super().__init__()
        
        self.fc1 = keras.Sequential([
            layers.Conv3D(och, 1, use_bias=False),
            layers.BatchNormalization(),
            layers.Activation('elu')
        ])
        
        self.STConv = keras.Sequential([
            layers.Conv3D(och, 3, padding='same', groups=och, use_bias=False),
            layers.BatchNormalization(),
            layers.Activation('elu')
        ])
        
        self.fc2 = keras.Sequential([
            layers.Conv3D(ich, 1, use_bias=False),
            layers.BatchNormalization(),
        ])
        
    def call(self, x):
        B, P, C = x.shape
        x = ops.reshape(x, (B, P//16, 4, 4, C))
        x = self.fc1(x)
        x = self.STConv(x)
        x = self.fc2(x)
        x = ops.reshape(x, (B, P, C))
        
        return x

class Block_ST_TDC_gra_sharp(keras.Layer):
    def __init__(self, num_heads, ich, och, dropout=0.1, theta=0.7):
        super().__init__()
        self.attn = MultiHeadedSelfAttention_TDC_gra_sharp(ich, num_heads, dropout, theta)
        self.proj = layers.Dense(ich)
        self.norm1 = layers.LayerNormalization(epsilon=1e-6)
        self.pwff = PositionWiseFeedForward_ST(ich, och)
        self.norm2 = layers.LayerNormalization(epsilon=1e-6)
        self.drop = layers.Dropout(dropout)
        
    def call(self, x, gra_sharp=2., training=None):
        attn, score = self.attn(self.norm1(x), gra_sharp=gra_sharp, training=training)
        h = self.drop(self.proj(attn), training=training)
        x = x + h
        h = self.drop(self.pwff(self.norm2(x)), training=training)
        x = x + h
        
        return x, score
    
class Transformer_ST_TDC_gra_sharp(keras.Layer):
    def __init__(self, num_layers, num_heads, ich, och, dropout=0.1, theta=0.7):
        super().__init__()
        self.blocks = [Block_ST_TDC_gra_sharp(num_heads, ich, och, dropout, theta) for _ in range(num_layers)]
    
    def call(self, x, training=None, gra_sharp=2.):
        for i in self.blocks:
            x, score = i(x, gra_sharp=gra_sharp, training=training)
            
        return x, score

class PhysFormer(keras.Model):
    def __init__(self, patches=(4, 4, 4), conv_ch=96, ff_ch=144, num_heads=12, num_layers=12, dropout_rate=0.1, theta=0.7, TN=False):
        super().__init__()
        ft, fh, fw = patches
        
        self.patch_embedding = layers.Conv3D(conv_ch, kernel_size=(ft, fh, fw), strides=(ft, fh, fw))
        self.transformers = [Transformer_ST_TDC_gra_sharp(num_layers=num_layers//3, num_heads=num_heads, ich=conv_ch, och=ff_ch, dropout=dropout_rate, theta=theta) for _ in range(3)]
        self.stem0 = keras.Sequential([
            layers.Conv3D(conv_ch//4, (1, 5, 5), padding='same'),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.MaxPool3D((1, 2, 2), (1, 2, 2))
        ])
        
        self.stem1 = keras.Sequential([
            layers.Conv3D(conv_ch//2, (3, 3, 3), padding='same'),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.MaxPool3D((1, 2, 2), (1, 2, 2))
        ])
        
        self.stem2 = keras.Sequential([
            layers.Conv3D(conv_ch, (3, 3, 3), padding='same'),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.MaxPool3D((1, 2, 2), (1, 2, 2))
        ])
        self.unsample1 = keras.Sequential([
            layers.UpSampling3D((2, 1, 1)),
            layers.Conv3D(conv_ch, (3, 1, 1), padding='same'),
            layers.BatchNormalization(),
            layers.Activation('elu')
        ])
        self.unsample2 = keras.Sequential([
            layers.UpSampling3D((2, 1, 1)),
            layers.Conv3D(conv_ch//2, (3, 1, 1), padding='same'),
            layers.BatchNormalization(),
            layers.Activation('elu')
        ])
        self.ConvBlockLast = layers.Conv1D(1, 1, dtype='float32')
        
        self.tn = TN
        
    def call(self, x, gra_sharp=2., training=None, return_scores=False):
        b, t, fh, fw, c = x.shape
        
        x = TNM(self.tn, axis=1, frames=x.shape[1])(x)
        x = self.stem0(x, training=training)
        x = TNM(self.tn, axis=1, frames=x.shape[1])(x)
        x = self.stem1(x, training=training)
        x = TNM(self.tn, axis=1, frames=x.shape[1])(x)
        x = self.stem2(x, training=training) # (Batch, 160, 16, 16, 96)
        
        x = self.patch_embedding(x) # (Batch, 40, 4, 4, 96)
        
        x = layers.Reshape((-1, x.shape[-1]))(x) # (B, 640, 96)
        
        x = layers.Reshape((t//4, 4, 4, -1))(x)         # reshape to B, T, H, W, C video features
        x = TNM(self.tn, axis=1, frames=x.shape[1])(x)  # apply TN on T axis
        x = layers.Reshape((-1, x.shape[-1]))(x)        # reshape to transformers features
        x, score1 = self.transformers[0](x, gra_sharp=gra_sharp, training=training)
        
        x = layers.Reshape((t//4, 4, 4, -1))(x)
        x = TNM(self.tn, axis=1, frames=x.shape[1])(x)
        x = layers.Reshape((-1, x.shape[-1]))(x)
        x, score2 = self.transformers[1](x, gra_sharp=gra_sharp, training=training)
        
        x = layers.Reshape((t//4, 4, 4, -1))(x)
        x = TNM(self.tn, axis=1, frames=x.shape[1])(x)
        x = layers.Reshape((-1, x.shape[-1]))(x)
        x, score3 = self.transformers[2](x, gra_sharp=gra_sharp, training=training) # (B, 640, 96)
        
        x = layers.Reshape((t//4, 4, 4, -1))(x) # (B, 40, 4, 4, 96)
        
        x = self.unsample1(x, training=training)
        x = self.unsample2(x, training=training) # (B, 160, 4, 4, 48)
        
        x = ops.mean(x, axis=(2, 3))
        
        x = self.ConvBlockLast(x)[...,0]
        
        if return_scores:
            return x, score1, score2, score3
        
        return x

def np_loss(x, y, eps=1e-3):
    x_, y_ = ops.mean(x, axis=-1, keepdims=True), ops.mean(y, axis=-1, keepdims=True)
    return 1-ops.sum((x-x_)*(y-y_), axis=-1)/((ops.sum((x-x_)**2, axis=-1)*ops.sum((y-y_)**2, axis=-1))**0.5+eps)

def psd(x, fs=30.): # 20 ~ 240 BPM
    x = jax.scipy.signal.welch(x, fs=fs, nfft=int(fs*60))[1][...,20:240]
    return x/ops.sum(x, axis=-1, keepdims=True)
    
def to_target_psd(hr, std=1.): # 20 ~ 240 BPM
    return jax.vmap(lambda hr:jax.scipy.stats.norm.pdf(jnp.arange(220), loc=hr-20, scale=std))(hr)

kl_loss = keras.losses.KLDivergence()

ce_loss = keras.losses.CategoricalCrossentropy(from_logits=True)

def psd_to_hr(x):
    return jnp.argmax(x, axis=-1)+20

def kl_ce_loss(y, pred, fs=30., std=8.):
    if len(pred.shape) == 1:
        pred, y = pred[None,:], y[None,:]
    ppsd, ypsd = psd(pred, fs=fs), psd(y, fs=fs)
    target_psd = to_target_psd(psd_to_hr(ypsd), std=std)
    kl = kl_loss(target_psd, ppsd)
    y_hr = psd_to_hr(ypsd)
    ce = ce_loss(ops.one_hot(y_hr, 240)[...,20:], ppsd)
    return kl + ce
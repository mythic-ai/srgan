# -*- coding: utf-8 -*-
"""Residual Dense Net, translated into Keras.
   Based on the original work: https://github.com/yulunzhang/RDN
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras import backend as K
from keras.layers import Input, Conv2D, Activation, Add, Concatenate, Conv2DTranspose, Lambda
from keras.models import Model

def res_dense_block(input_tensor, C, G, d):
    """Residual dense block"""
    if K.image_data_format() == 'channels_last':
        ch_axis = 3
    else:
        ch_axis = 1
    G_0 = K.int_shape(input_tensor)[ch_axis]
    conv_base_name = 'conv' + str(d) + '_'

    # Dense connected layers, max cost = (G_0 + (C-1)*G) * G * 9
    fused_layer = input_tensor
    for c in range(C):
        F_d_c = Conv2D(G, (3, 3), padding='same', activation='relu', name=conv_base_name+str(c))(fused_layer)
        fused_layer = Concatenate(axis=ch_axis)([fused_layer, F_d_c])

    # Local feature fusion, cost = (G_0 + C*G) * G_0
    fused_layer = Conv2D(G_0, (1, 1), name=conv_base_name+'fusion')(fused_layer)

    # Local residual learning
    output_tensor = Add()([input_tensor, fused_layer])

    return output_tensor

def upsample(input_tensor, channels, scale):
    """Up-samples with the efficient sub-pixel convolution (ESPCN), using the tensorflow backend."""
    import tensorflow as tf

    packed = Conv2D(channels * scale * scale, (3, 3), padding='same')(input_tensor)
    output_tensor = Lambda(tf.depth_to_space, arguments={'block_size': scale})(packed)

    return output_tensor

def RDN(D, C, F, G, G_0, height=None, width=None):
    """Residual Dense Network, with depth D * (C+1) + 7"""
    if K.image_data_format() == 'channels_last':
        input_shape = (height, width, 3)
        ch_axis = 3
    else:
        input_shape = (3, height, width)
        ch_axis = 1

    # Shallow feature extraction
    img_lr = Input(shape=input_shape)
    F_minus1 = Conv2D(F, (3, 3), padding='same')(img_lr)

    # Residual dense blocks
    F_d = Conv2D(G_0, (3, 3), padding='same')(F_minus1)
    dense_outputs = []
    for d in range(D):
        F_d = res_dense_block(F_d, C, G, d)
        dense_outputs.append(F_d)

    # Global feature fusion, cost = D * G_0 * F
    F_GF = Concatenate(axis=ch_axis)(dense_outputs)
    F_GF = Conv2D(F, (1, 1))(F_GF)
    F_GF = Conv2D(F, (3, 3), padding='same')(F_GF)

    # Global residual learning
    F_DF = Add()([F_minus1, F_GF])

    # Up-sampling x4 (x2 twice)
    img_sr = upsample(F_DF, F, 2)
    img_sr = upsample(img_sr, F, 2)

    # Output RGB colors
    img_sr = Conv2D(3, (3, 3), padding='same')(img_sr)

    return Model(img_lr, img_sr)

def RDN147():
    """RDN-147 with x4 scaling factor, the original model described in the paper"""
    return RDN(20, 6, 64, 32, 64)

def RDN27():
    """RDN-27 with x4 scaling factor, a smaller version. Warning: receptive field is small."""
    return RDN(5, 3, 32, 16, 32)

"""Ignore everything after this line :)"""

def aram_blocks(input_tensor, D, C):
    """Aram's version of res-dense-blocks"""
    if K.image_data_format() == 'channels_last':
        ch_axis = 3
    else:
        ch_axis = 1
    block_depth = K.int_shape(input_tensor)[ch_axis] // C

    # First Aram block, without residual connections
    mem = []
    mem_plus = []
    for c in range(C):
        mem.append(Conv2D(block_depth, (3, 3), padding='same', name='conv0_' + str(c))(input_tensor))
        mem_plus.append(Activation('relu'))(mem[c])
    F_d = Concatenate(axis=ch_axis)(mem_plus)

    # D more Aram blocks, with residual connections
    dense_outputs = []
    for d in range(D):
        dense_outputs.append(mem_plus[0])
        for c in range(C):
            res = Conv2D(block_depth, (3, 3), padding='same', name='conv' + str(d + 1) + '_' + str(c))(F_d)
            mem[c] = Add()([mem[c], res])
            mem_plus[c] = Activation('relu')(mem[c])
            F_d = Concatenate(axis=ch_axis)(mem_plus)

    dense_outputs.append(F_d)
    return dense_outputs

def AramNet(D=8, C=4, M=32, height=None, width=None):
    """AramNet, based on RDN"""
    """D+1 should be about equal to the kernel area: 8+1 = 3x3.
       C does not affect overall network size, and should be as high as possible without sacrificing parallelism.
       M is the memory or network width."""
    if K.image_data_format() == 'channels_last':
        input_shape = (height, width, 3)
        ch_axis = 3
    else:
        input_shape = (3, height, width)
        ch_axis = 1

    # Shallow feature extraction
    img_lr = Input(shape=input_shape)
    F_minus1 = Conv2D(M, (3, 3), padding='same')(img_lr)

    dense_outputs = aram_blocks(F_minus1, D, C)

    # Global feature fusion
    F_GF = Concatenate(axis=ch_axis)(dense_outputs)
    F_GF = Conv2D(M, (1, 1))(F_GF)
    F_GF = Conv2D(M, (3, 3), padding='same')(F_GF)

    # Global residual learning
    F_DF = Add()([F_minus1, F_GF])

    # Up-sampling net
    img_sr = upsample(F_DF, M, 2)
    img_sr = upsample(img_sr, M, 2)

    # Output RGB colors
    img_sr = Conv2D(3, (3, 3), padding='same')(img_sr)

    return Model(img_lr, img_sr)
# -*- coding: utf-8 -*-
"""Residual Dense Net, translated into Keras
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras import backend as K
from keras.layers import Input, Conv2D, Add, Concatenate, Conv2DTranspose
from keras.models import Model
from IPython import embed

def RDB(input_tensor, C, G, d):
    """Residual dense block"""
    if K.image_data_format() == 'channels_last':
        ch_axis = 3
    else:
        ch_axis = 1
    G_0 = K.int_shape(input_tensor)[ch_axis]
    conv_base_name = 'conv' + str(d) + '_'

    # Dense connected layers
    fused_layer = input_tensor
    for c in range(C):
        F_d_c = Conv2D(G, (3, 3), padding='same', activation='relu', name=conv_base_name+str(c))(fused_layer)
        fused_layer = Concatenate(axis=ch_axis)([fused_layer, F_d_c])

    # Local feature fusion
    fused_layer = Conv2D(G_0, (1, 1), name=conv_base_name+'fusion')(fused_layer)

    # Local residual learning
    output_tensor = Add()([input_tensor, fused_layer])

    return output_tensor

def UPNet(input_tensor):
    """Up-sampling net"""
    if K.image_data_format() == 'channels_last':
        ch_axis = 3
    else:
        ch_axis = 1
    G_0 = K.int_shape(input_tensor)[ch_axis]

    output_tensor = Conv2DTranspose(G_0, (1, 1), strides=(2, 2))(input_tensor)
    return output_tensor

def get_model(D=20, C=6, G=32, G_0=64, height=None, width=None):
    """Residual Dense Network"""

    if K.image_data_format() == 'channels_last':
        input_shape = (height, width, 3)
        ch_axis = 3
    else:
        input_shape = (3, height, width)
        ch_axis = 1

    # Shallow feature extraction
    img_lr = Input(shape=input_shape)
    F_minus1 = Conv2D(G_0, (3, 3), padding='same')(img_lr)
    F_d = Conv2D(G_0, (3, 3), padding='same')(F_minus1)

    # Residual dense blocks
    dense_outputs = []
    for d in range(D):
        F_d = RDB(F_d, C, G, d)
        dense_outputs.append(F_d)

    # Global feature fusion
    F_GF = Concatenate(axis=ch_axis)(dense_outputs)
    F_GF = Conv2D(G_0, (1, 1))(F_GF)
    F_GF = Conv2D(G_0, (3, 3), padding='same')(F_GF)

    # Global residual learning
    F_DF = Add()([F_minus1, F_GF])

    # Up-sampling net
    img_sr = UPNet(F_DF)
    img_sr = Conv2D(3, (3, 3), padding='same')(img_sr)

    return Model(img_lr, img_sr)
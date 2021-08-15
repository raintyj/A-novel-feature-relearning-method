import tensorflow as tf
from tensorflow.keras.layers import Concatenate,Lambda,Activation,Add,GlobalMaxPooling1D,Conv1D,BatchNormalization,MaxPool1D,GlobalAveragePooling1D,Dense,Input,LSTM,multiply,Reshape
from tensorflow.keras import optimizers, losses, activations, models
import numpy as np
import tensorflow.keras.backend as K
from tensorflow.keras.regularizers import l1


def channel_attention(input_feature, ratio=4, reg_coeff=1e-8):
    channel = input_feature.shape[-1]

    shared_layer_one = Dense(channel // ratio,
                             activation='relu',
                             kernel_initializer='he_normal',
                             kernel_regularizer=l1(reg_coeff),
                             use_bias=False,
                             bias_initializer='zeros',
                             bias_regularizer=l1(reg_coeff)
                             )
    shared_layer_two = Dense(channel,
                             kernel_initializer='he_normal',
                             kernel_regularizer=l1(reg_coeff),
                             use_bias=True,
                             bias_initializer='zeros',
                             bias_regularizer=l1(reg_coeff)

                             )

    avg_pool = GlobalAveragePooling1D()(input_feature)
    avg_pool = Reshape((1, channel))(avg_pool)
    avg_pool = shared_layer_one(avg_pool)
    avg_pool = shared_layer_two(avg_pool)

    max_pool = GlobalMaxPooling1D()(input_feature)
    max_pool = Reshape((1, channel))(max_pool)
    max_pool = shared_layer_one(max_pool)
    max_pool = shared_layer_two(max_pool)

    cbam_feature = Add()([avg_pool, max_pool])
    cbam_feature = Activation('sigmoid')(cbam_feature)

    return multiply([input_feature, cbam_feature])

def channel_block(cbam_feature, ratio=4):
    cbam_feature = channel_attention(cbam_feature, ratio)
    return cbam_feature

def spatial_attention(input_feature, kernel_size=7, reg_coeff=1e-8):
    channel_size = input_feature.shape[-1]
    avg_pool = Lambda(lambda x: K.mean(x, axis=2, keepdims=True))(input_feature)
    max_pool = Lambda(lambda x: K.max(x, axis=2, keepdims=True))(input_feature)
    concat = Concatenate(axis=2)([avg_pool, max_pool])
    cbam_feature = tf.keras.layers.Conv1D(filters=channel_size,
                                          kernel_size=kernel_size,
                                          strides=1,
                                          padding='same',
                                          activation='relu',
                                          kernel_initializer='he_normal',
                                          kernel_regularizer=l1(reg_coeff),
                                          use_bias=False)(concat)

    output_feature = multiply([input_feature, cbam_feature])
    return output_feature

def spatialattention_block(cbam_feature):
    cbam_feature = spatial_attention(cbam_feature)
    return cbam_feature

def cbam_block(cbam_feature, ratio=4 ):
    cbam_feature = channel_attention(cbam_feature, ratio)
    cbam_feature = spatial_attention(cbam_feature)
    return cbam_feature

def blockconv(myfilter,kernels,mystride,x,sty):
    if sty == 1:
        x_all = BatchNormalization()(Conv1D(filters=myfilter, kernel_size=kernels,
                                strides=mystride,padding="same", activation='relu',use_bias=False)(x))
    else:
        x_all = BatchNormalization()(MaxPool1D(pool_size=2,strides=2,padding='same')(Conv1D(filters=myfilter, kernel_size=kernels,
                                            strides=mystride, padding="same", activation='relu', use_bias=False)(x)))
    x_all = cbam_block(x_all)
    return x_all


def _upsample(inputs, new_width):
    inps = tf.identity(inputs)
    w1 = (new_width -inputs.shape[1]) // 2
    w2 = (new_width -inputs.shape[1]) -w1
    pad2 = np.array([[0, 0], [w1, w2], [0, 0]])
    tsr_pad2 = tf.pad(inps, pad2, mode="SYMMETRIC")
    return tsr_pad2


def to_upsample(x, y):
    HW = y.shape[1]
    up_result = cbam_block(tf.concat((_upsample(x, HW),y),axis=-1))
    return  up_result

def model_part1():
    inp = Input(shape=(3000, 1))
    c1 = blockconv(12,3,2,inp,0)
    c2 = blockconv(12,3,2,c1,1)
    c3 = blockconv(12,3,2,c2,1)
    c4 = blockconv(12,3,2,c3,1)
    c5 = blockconv(12,3,2,c4,1)

    p5 = LSTM(units=12,return_sequences=True)(c5)
    p4 = to_upsample(p5,LSTM(units=12,return_sequences=True)(c4))
    p3 = to_upsample(p4,LSTM(units=12,return_sequences=True)(c3))
    p = to_upsample(p3,LSTM(units=12,return_sequences=True)(c2))
    p = blockconv(24, 3, 2, p, 1)
    p = GlobalAveragePooling1D()(p)
    p = Dense(12, activation=activations.relu)(p)
    p = Dense(4, activation=activations.softmax)(p)
    model = models.Model(inputs=inp, outputs=p)
    opt = optimizers.Adam(0.001)
    model.compile(optimizer=opt, loss=losses.sparse_categorical_crossentropy, metrics=['acc'])
    return model

def model_part2():
    inp = Input(shape=(3000, 1))
    p = Conv1D(filters=24, kernel_size=3, strides=2,padding="same", use_bias=False)(inp)
    p = MaxPool1D(pool_size=3, strides=2, padding="same")(p)
    p = BatchNormalization()(p)
    p = cbam_block(p)
    p= LSTM(units=36, return_sequences=True)(p)

    p = GlobalAveragePooling1D()(p)
    p = Dense(10, activation=activations.relu)(p)
    p = Dense(2, activation=activations.softmax)(p)
    model = models.Model(inputs=inp, outputs=p)
    opt = optimizers.Adam(0.001)
    model.compile(optimizer=opt, loss=losses.sparse_categorical_crossentropy, metrics=['acc'])
    return model



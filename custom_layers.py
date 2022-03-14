import tensorflow as tf
import numpy as np
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Layer
from tensorflow.keras import initializers
from tensorflow.keras import regularizers
from tensorflow.keras import constraints


class Average(Layer):
    def __init__(self, **kwargs):
        self.supports_masking = True
        super(Average, self).__init__(**kwargs)

    def call(self, x, mask):
        if mask is not None:
            mask = tf.cast(mask, tf.float32)
            mask = tf.expand_dims(mask, axis=-1)
            x *= mask
        avg = tf.reduce_sum(x, axis=-2) / tf.reduce_sum(mask, axis=-2)
        return avg


class Attention(Layer):
    def __init__(self, M_regularizer=None, M_constraint=None, **kwargs):
        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')
        self.M_regularizer = regularizers.get(M_regularizer)
        self.M_constraint = constraints.get(M_constraint)
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.steps = input_shape[0][1]
        self.M = self.add_weight(shape=(input_shape[0][-1], input_shape[1][-1]), initializer=self.init,
                                 name='{}_M'.format(self.name), regularizer=self.M_regularizer, constraint=self.M_constraint)
        self.built = True

    def call(self, input_tensor, mask=None):
        x = input_tensor[0]
        y = input_tensor[1]
        mask = mask[0]

        y = tf.transpose(tf.linalg.matmul(self.M, tf.transpose(y)))
        y = tf.expand_dims(y, axis=-2)
        y = tf.repeat(y, self.steps, axis=1)
        eij = tf.reduce_sum(x*y, axis=-1)

        eij = tf.math.tanh(eij)
        a = tf.math.exp(eij)

        if mask is not None:
            a *= tf.cast(mask, K.floatx())

        a /= tf.cast(tf.reduce_sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())
        return a


class WeightedSum(Layer):
    def __init__(self, **kwargs):
        self.supports_masking = True
        super(WeightedSum, self).__init__(**kwargs)

    def call(self, input_tensor, mask=None):
        x = input_tensor[0]
        a = tf.expand_dims(input_tensor[1], axis=-1)
        weighted_input = x * a
        return tf.reduce_sum(weighted_input, axis=1)


class MaxMargin(Layer):
    def __init__(self, **kwargs):
        super(MaxMargin, self).__init__(**kwargs)

    def call(self, outputs):
        zp = outputs[0]
        zn = outputs[1]
        rs = outputs[2]
        zp = tf.math.l2_normalize(zp, axis=-1)
        zn = tf.math.l2_normalize(zn, axis=-1)
        rs = tf.math.l2_normalize(rs, axis=-1)

        steps = zn.shape[1]

        pos = tf.reduce_sum(rs * zp, axis=-1, keepdims=True)
        pos = tf.repeat(pos, steps, axis=1)
        rs = tf.expand_dims(rs, axis=-2)
        rs = tf.repeat(rs, steps, axis=1)
        neg = tf.reduce_sum(rs * zn, axis=-1)

        loss = tf.cast(tf.reduce_sum(tf.maximum(0., (1. - pos + neg)), axis=-1, keepdims=True), K.floatx())
        return loss


class WeightedAspectEmb(Layer):
    def __init__(self, input_dim, output_dim, init='uniform', W_regularizer=None, W_constraint=None, dropout=0., **kwargs):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.init = initializers.get(init)
        self.W_regularizer = regularizers.get(W_regularizer)
        self.W_constraint = constraints.get(W_constraint)
        self.dropout = dropout
        if 0. < self.dropout < 1.:
            self.uses_learning_phase = True
        super(WeightedAspectEmb, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(shape=[self.input_dim, self.output_dim], regularizer=self.W_regularizer,
                                 initializer=self.init, constraint=self.W_constraint, name='{}_W'.format(self.name))
        self.built = True

    def call(self, x, mask=None):
        return tf.matmul(x, self.W)
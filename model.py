import tensorflow as tf
import numpy as np
import tensorflow.keras.backend as K
from tensorflow.keras import Model
from custom_layers import Average, Attention, MaxMargin, WeightedAspectEmb, WeightedSum
from tensorflow.keras.layers import Dense, Activation, Embedding
from tensorflow.keras import Input
from Word2VecReader import W2VEmbReader


def build_model(args, maxlen, vocab):
    def ortho_reg(weight_matrix):
        w_n = weight_matrix / tf.cast(K.epsilon() + tf.math.sqrt(tf.reduce_sum(tf.math.square(weight_matrix), axis=-1, keepdims=True)), tf.float32)
        reg = tf.reduce_sum(tf.math.square(tf.matmul(w_n, tf.transpose(w_n)) - tf.eye(w_n.shape[0])))
        return args['ortho_reg'] * reg

    vocab_size = len(vocab)

    #### Входные данные
    pos_input = Input(shape=(maxlen), dtype='int32', name='sentence_input')
    neg_input = Input(shape=(args['neg_size'], maxlen), dtype='int32', name='neg_input')

    #### Создается репрезентация предложения
    embedding = Embedding(input_dim=vocab_size, output_dim=args['emb_dim'], mask_zero=True, name='word_embedding', trainable=False)
    pos_emb = embedding(pos_input)
    y_s = Average(name='y_s')(pos_emb)
    attn = Attention(name='attention')([pos_emb, y_s])
    z_p = WeightedSum(name='z_p')([pos_emb, attn])

    #### Создается реперезнтация предложение через негативные примеры
    neg_emb = embedding(neg_input)
    z_n = Average()(neg_emb)

    #### Создается реконструкция предложения
    p_t = Dense(args['aspect_size'])(z_p)
    p_t = Activation('softmax', name='p_t')(p_t)

    r_s = WeightedAspectEmb(args['aspect_size'], args['emb_dim'], name='aspect_emb', W_regularizer=ortho_reg)(p_t)

    #### Подсчитывается потери
    loss = MaxMargin(name='max_margin')([z_p, z_n, r_s])

    #### Создается модель
    new_model = Model(inputs=[pos_input, neg_input], outputs=loss)

    #### Назначения embedding
    emb_reader = W2VEmbReader(args['emb_path'])
    print('Initializing word embedding matrix')
    K.set_value(new_model.get_layer('word_embedding').embeddings, emb_reader.get_emb_matrix_given_vocab(vocab,
                                                                                                        K.get_value(
                                                                                                            new_model.get_layer(
                                                                                                                'word_embedding').embeddings)))
    #####################################################################
    print('Initializing aspect embedding matrix as centroid of kmean clusters')
    K.set_value(new_model.get_layer('aspect_emb').weights[0], emb_reader.get_aspect_matrix(args['aspect_size']))

    return new_model

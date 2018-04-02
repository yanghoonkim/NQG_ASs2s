# Following are functions that I always use
# Made by Yanghoon Kim, SNU
# Tensorflow 1.4
# 2018.03.01


import numpy as np
import tensorflow as tf 


def embed_op(inputs, params, name = 'embedding'):
    if params['embedding'] == None:
        with tf.variable_scope('EmbeddingScope'):
            embedding = tf.get_variable(
                    name, 
                    [params['voca_size'], params['hidden_size']], 
                    dtype = params['dtype']
                    )
    else:
        glove = np.load(params['embedding'])
        with tf.variable_scope('EmbeddingScope'):
            init = tf.constant_initializer(glove)
            embedding = tf.get_variable(
                    name,
                    [params['voca_size'], params['hidden_size']],
                    dtype = tf.float32,
                    trainable = params['embedding_trainable'] 
                    )

    tf.summary.histogram(embedding.name + '/value', embedding)
    return tf.nn.embedding_lookup(embedding, inputs)
    
    

def conv_op(embd_inp, params):
    fltr = tf.get_variable(
        'conv_fltr', 
        params['kernel'], 
        params['dtype'], 
        regularizer = tf.contrib.layers.l2_regularizer(1.0)
            )

    convout = tf.nn.conv1d(embd_inp, fltr, params['stride'], params['conv_pad'])
    return convout




def ffn_op(x, params):
    out = x
    if params['ffn_size'] == None:
        ffn_size = []
    else:
        ffn_size = params['ffn_size']
    for unit_size in ffn_size[:-1]:
        out = tf.layers.dense(
            out, 
            unit_size, 
            activation = tf.tanh, 
            use_bias = True, 
            kernel_regularizer = tf.contrib.layers.l2_regularizer(1.0)
        )
    return tf.layers.dense(
        out, 
        params['label_size'], 
        activation = None, 
        use_bias = True, 
        kernel_regularizer = tf.contrib.layers.l2_regularizer(1.0)
    )
    

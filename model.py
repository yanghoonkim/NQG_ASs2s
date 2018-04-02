import numpy as np
import tensorflow as tf

import sys
sys.path.append('submodule/')
from mytools import *
import attention_wrapper_mod


def q_generation(features, labels, mode, params):
    
    hidden_size = params['hidden_size']
    voca_size = params['voca_size']   
    
    sentence = features['s'] # [batch, length]
    len_s = features['len_s']
    
    if mode != tf.estimator.ModeKeys.PREDICT:
        question = features['q'] # label
        len_q = features['len_q']
    else:
        question = features['q']
        len_q = features['len_q']
    
    # Embedding for sentence, question and rnn encoding of sentence
    with tf.variable_scope('SharedScope'):
        # Embedded inputs
        embd_s = embed_op(sentence, params, name = 'embedding_s')
        embd_q = embed_op(question, params, name = 'embedding_q')

        # Build encoder cell
        def gru_cell():
            cell = tf.nn.rnn_cell.GRUCell(hidden_size)
            #cell = tf.contrib.rnn.BasicLSTMCell(hidden_size)
            return tf.contrib.rnn.DropoutWrapper(cell, 
                    output_keep_prob = 1 - params['rnn_dropout'] if mode == tf.estimator.ModeKeys.TRAIN else 1)

        encoder_cell_fw = gru_cell() if params['encoder_layer'] == 1 else tf.nn.rnn_cell.MultiRNNCell([gru_cell() for _ in range(params['encoder_layer'])])
        encoder_cell_bw = gru_cell() if params['encoder_layer'] == 1 else tf.nn.rnn_cell.MultiRNNCell([gru_cell() for _ in range(params['encoder_layer'])])


        # Run Dynamic RNN
        #   encoder_outputs: [max_time, batch_size, num_units]
        #   encoder_state: last hidden state of encoder, [batch_size, num_units]
        #encoder_outputs, encoder_state = tf.nn.dynamic_rnn(
        #    encoder_cell, embd_s,
        #    sequence_length=len_s,
        #    dtype = tf.float32    
        #    )

        encoder_outputs, encoder_state = tf.nn.bidirectional_dynamic_rnn(
                encoder_cell_fw,
                encoder_cell_bw,
                inputs = embd_s,
                sequence_length = len_s,
                dtype = tf.float32)

        encoder_outputs = tf.concat(encoder_outputs, 2)
        #encoder_outputs = tf.layers.dense(encoder_outputs, params['hidden_size'])
        
    # This part should be moved into QuestionGeneration scope    
    with tf.variable_scope('SharedScope/EmbeddingScope', reuse = True):
        embedding_q = tf.get_variable('embedding_q')
 
    # Rnn decoding of sentence with attention 
    with tf.variable_scope('QuestionGeneration'):
        # Memory for attention
        attention_states = encoder_outputs

        # Create an attention mechanism
        attention_mechanism = tf.contrib.seq2seq.LuongAttention(
                hidden_size, attention_states,
                memory_sequence_length=len_s)

        # batch_size should not be specified
        # if fixed, then the redundant eval_data will make error
        # it may related to bug of tensorflow api
        batch_size = attention_mechanism._batch_size
        # Build decoder cell
        decoder_cell = gru_cell() if params['decoder_layer'] == 1 else tf.nn.rnn_cell.MultiRNNCell([gru_cell() for _ in range(params['decoder_layer'])])

        decoder_cell = tf.contrib.seq2seq.AttentionWrapper(
                decoder_cell, attention_mechanism,
                attention_layer_size=hidden_size)
                #initial_cell_state = encoder_state)

        # Helper for decoder cell

        if mode == tf.estimator.ModeKeys.TRAIN:
            #len_q_fake = params['maxlen_q_train'] * tf.ones([batch_size], dtype = tf.int32)
            len_q = tf.cast(len_q, tf.int32)
            helper = tf.contrib.seq2seq.TrainingHelper(
                    embd_q, len_q
                    )
        else: # EVAL & TEST
            #start_token = params['start_token'] * tf.ones([batch_size], dtype = tf.int32)
            #helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
            #        embedding_q, start_token, params['fake_end_token']
            #        )
            len_q = tf.cast(len_q, tf.int32)
            helper = tf.contrib.seq2seq.TrainingHelper(
                    embd_q, len_q)
        # Decoder
        initial_state = decoder_cell.zero_state(dtype = tf.float32, batch_size = batch_size)
        projection_q = tf.layers.Dense(voca_size, use_bias = False)

        decoder = tf.contrib.seq2seq.BasicDecoder(
            decoder_cell, helper, initial_state,
            output_layer=None)

        # Dynamic decoding
        max_iter = params['maxlen_q_dev'] 

        if mode == tf.estimator.ModeKeys.TRAIN:
            outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder, impute_finished=True, maximum_iterations = None)
        else: # Test
            #outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder, impute_finished=True, maximum_iterations = max_iter)
            outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder, impute_finished=True, maximum_iterations = None)

        logits_q = projection_q(outputs.rnn_output)

    
    # Predictions
    softmax_q = tf.nn.softmax(logits_q)
    predictions_q = tf.argmax(softmax_q, axis = -1)

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(
                mode = mode,
                predictions = {
                    'question' : predictions_q
                    })
    # Loss
    label_q = tf.cast(question[:,1:], tf.int32, name = 'label_q')
    label_q = label_q[:, :tf.reduce_max(len_q)]
    
    if mode == tf.estimator.ModeKeys.TRAIN:
        weight_q = tf.sequence_mask(len_q, tf.reduce_max(len_q), tf.float32)
    elif mode == tf.estimator.ModeKeys.EVAL:
        weight_q = tf.sequence_mask(len_q, tf.reduce_max(len_q), tf.float32)

    loss_q = tf.contrib.seq2seq.sequence_loss(
            logits_q, 
            label_q,
            weight_q, # [batch, length]
            average_across_timesteps = True,
            average_across_batch = True,
            softmax_loss_function = None # default : sparse_softmax_cross_entropy
            )
    
    loss_reg = tf.reduce_sum([tf.nn.l2_loss(v) for v in tf.trainable_variables()])
    loss = loss_q + params['regularization'] * loss_reg

    # eval_metric for estimator
    eval_metric_ops = None

    # Summary
    tf.summary.scalar('loss_reg', loss_reg)
    tf.summary.scalar('loss_question', loss_q)
    tf.summary.scalar('total_loss', loss)


    # Optimizer
    learning_rate = params['learning_rate']
    if params['decay_step'] is not None:
        learning_rate = tf.train.exponential_decay(learning_rate, tf.train.get_global_step(), params['decay_step'], params['decay_rate'], staircase = True)
    optimizer = tf.train.AdamOptimizer(learning_rate)

    grad_and_var = optimizer.compute_gradients(loss, tf.trainable_variables())
    grad, var = zip(*grad_and_var)
    clipped_grad, norm = tf.clip_by_global_norm(grad, 5)
    train_op = optimizer.apply_gradients(zip(clipped_grad, var), global_step = tf.train.get_global_step())
        
    return tf.estimator.EstimatorSpec(
        mode=mode,
        loss=loss,
        train_op=train_op,
        eval_metric_ops=eval_metric_ops)
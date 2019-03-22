import numpy as np
import tensorflow as tf

import sys
sys.path.append('submodule/')
from mytools import *
import rnn_wrapper as wrapper

def _attention(params, memory, memory_length):
    if params['attn'] == 'bahdanau':
        return tf.contrib.seq2seq.BahdanauAttention(
                params['hidden_size'] * 2,
                memory,
                memory_length)
    elif params['attn'] == 'normed_bahdanau':
        return tf.contrib.seq2seq.BahdanauAttention(
                params['hidden_size'] * 2,
                memory,
                memory_length,
                normalize = True)
    
    elif params['attn'] == 'luong':
        return tf.contrib.seq2seq.LuongAttention(
                params['hidden_size'] * 2,
                memory,
                memory_length)

    elif params['attn'] == 'scaled_luong':
        return tf.contrib.seq2seq.LuongAttention(
                params['hidden_size'] * 2,
                memory,
                memory_length,
                scale = True)
    else:
        raise ValueError('Unknown attention mechanism : %s' %params['attn'])

def q_generation(features, labels, mode, params):

    
    dtype = params['dtype']
    hidden_size = params['hidden_size']
    voca_size = params['voca_size']   
    
    sentence = features['s'] # [batch, length]
    len_s = features['len_s']

    answer = features['a']
    len_a = features['len_a']

    beam_width = params['beam_width']
    length_penalty_weight = params['length_penalty_weight']
    
    # batch_size should not be specified
    # if fixed, then the redundant eval_data will make error
    # it may related to bug of tensorflow api
    batch_size = tf.shape(sentence)[0]
    
    if mode != tf.estimator.ModeKeys.PREDICT:
        question = tf.cast(features['q'], tf.int32) # label
        len_q = tf.cast(features['len_q'], tf.int32) 
    else:
        question = None
        len_q = None
    
    # Embedding for sentence, question and rnn encoding of sentence
    with tf.variable_scope('SharedScope'):
        # Embedded inputs
        # Same name == embedding sharing
        embd_s = embed_op(sentence, params, name = 'embedding')
        embd_a = embed_op(answer, params, name = 'embedding')
        if question is not None:
            embd_q = embed_op(question, params, name = 'embedding')

        # Build encoder cell
        def lstm_cell_enc():
            cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_size)
            return tf.contrib.rnn.DropoutWrapper(cell, 
                    input_keep_prob = 1 - params['rnn_dropout'] if mode == tf.estimator.ModeKeys.TRAIN else 1)
        def lstm_cell_dec():
            cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_size * 2)
            return tf.contrib.rnn.DropoutWrapper(cell,
                    input_keep_prob = 1 - params['rnn_dropout'] if mode == tf.estimator.ModeKeys.TRAIN else 1)

        encoder_cell_fw = lstm_cell_enc() if params['encoder_layer'] == 1 else tf.nn.rnn_cell.MultiRNNCell([lstm_cell_enc() for _ in range(params['encoder_layer'])])
        encoder_cell_bw = lstm_cell_enc() if params['encoder_layer'] == 1 else tf.nn.rnn_cell.MultiRNNCell([lstm_cell_enc() for _ in range(params['encoder_layer'])])


        encoder_outputs, encoder_state = tf.nn.bidirectional_dynamic_rnn(
                encoder_cell_fw,
                encoder_cell_bw,
                inputs = embd_s,
                sequence_length = len_s,
                dtype = dtype)

        encoder_outputs = tf.concat(encoder_outputs, -1)

        if params['encoder_layer'] == 1:
            encoder_state_c = tf.concat([encoder_state[0].c, encoder_state[1].c], axis = 1)
            encoder_state_h = tf.concat([encoder_state[0].h, encoder_state[1].h], axis = 1)
            encoder_state = tf.contrib.rnn.LSTMStateTuple(c = encoder_state_c, h = encoder_state_h)

        else:
            _encoder_state = list()
            for state_fw, state_bw in zip(encoder_state[0], encoder_state[1]):
                partial_state_c = tf.concat([state_fw.c, state_bw.c], axis = 1)
                partial_state_h = tf.concat([state_fw.h, state_bw.h], axis = 1)
                partial_state = tf.contrib.rnn.LSTMStateTuple(c = partial_state_c, h = partial_state_h)
                _encoder_state.append(partial_state)
            encoder_state = tuple(_encoder_state)


        answer_cell_fw = lstm_cell_enc() if params['answer_layer'] == 1 else tf.nn.rnn_cell.MultiRNNCell([lstm_cell_enc() for _ in range(params['answer_layer'])])
        answer_cell_bw = lstm_cell_enc() if params['answer_layer'] == 1 else tf.nn.rnn_cell.MultiRNNCell([lstm_cell_enc() for _ in range(params['answer_layer'])])

        answer_outputs, answer_state = tf.nn.bidirectional_dynamic_rnn(
                answer_cell_fw,
                answer_cell_bw,
                inputs = embd_a,
                sequence_length = len_a,
                dtype = dtype,
                scope = 'answer_scope')
        
        answer_outputs = tf.concat(answer_outputs, -1)
        if params['answer_layer'] == 1:
            answer_state_c = tf.concat([answer_state[0].c, answer_state[1].c], axis = 1)
            answer_state_h = tf.concat([answer_state[0].h, answer_state[1].h], axis = 1)
            answer_state = tf.contrib.rnn.LSTMStateTuple(c = answer_state_c, h = answer_state_h)
        else:
            _answer_state = list()
            for state_fw, state_bw in zip(answer_state[0], answer_state[1]):
                partial_state_c = tf.concat([state_fw.c, state_bw.c], axis = 1)
                partial_state_h = tf.concat([state_fw.h, state_bw.h], axis = 1)
                partial_state = tf.contrib.rnn.LSTMStateTuple(c = partial_state_c, h = partial_state_h)
                _answer_state.append(partial_state)
            answer_state = tuple(_answer_state)


        if params['dec_init_ans'] and params['decoder_layer'] == params['answer_layer']:
            copy_state = answer_state
        elif params['encoder_layer'] == params['decoder_layer']:
            copy_state = encoder_state
        else:
            copy_state = None

        if mode == tf.estimator.ModeKeys.PREDICT and beam_width > 0:
            encoder_outputs = tf.contrib.seq2seq.tile_batch(encoder_outputs, beam_width)
            len_s = tf.contrib.seq2seq.tile_batch(len_s, beam_width)

        if mode == tf.estimator.ModeKeys.PREDICT and beam_width > 0 and copy_state is not None:
            copy_state = tf.contrib.seq2seq.tile_batch(copy_state, beam_width)

    # This part should be moved into QuestionGeneration scope    
    with tf.variable_scope('SharedScope/EmbeddingScope', reuse = True):
        embedding_q = tf.get_variable('embedding')
 
    # Rnn decoding of sentence with attention 
    with tf.variable_scope('QuestionGeneration'):
        # Memory for attention
        attention_states = encoder_outputs

        # Create an attention mechanism
        attention_mechanism = _attention(params, attention_states, len_s)

        # Build decoder cell
        decoder_cell = lstm_cell_dec() if params['decoder_layer'] == 1 else tf.nn.rnn_cell.MultiRNNCell([lstm_cell_dec() for _ in range(params['decoder_layer'])])
        
        if params['use_keyword'] > 0:
            #cell_input_fn = lambda inputs, attention : tf.concat([inputs, attention, o_s], -1)
            def cell_input_fn(inputs, attention):
                if mode == tf.estimator.ModeKeys.PREDICT and beam_width > 0:
                    last_attention = attention[0::beam_width]
                else:
                    last_attention = attention
                last_attention = tf.expand_dims(last_attention, 2)
                m = answer_outputs
                p_s = tf.nn.softmax(tf.matmul(m, last_attention), name = 'p_s')
                o_s = tf.reduce_sum(p_s * m, axis = 1)

                if mode == tf.estimator.ModeKeys.PREDICT and beam_width > 0:
                    o_s = tf.contrib.seq2seq.tile_batch(o_s, beam_width)

                return tf.concat([inputs, o_s], -1)
                
        else:
            cell_input_fn = None
            
            #def cell_input_fn(inputs, attention):
            #    global o_s
            #    if mode == tf.estimator.ModeKeys.PREDICT and beam_width > 0:
            #        inputs = inputs[0::beam_width]
            #    o_s = tf.layers.dense(tf.concat([inputs, o_s], -1), params['hidden_size'], activation = tf.tanh)
            #    if mode == tf.estimator.ModeKeys.PREDICT and beam_width > 0:
            #        o_s = tf.contrib.seq2seq.tile_batch(o_s, beam_width)
            #    return tf.concat([o_s, attention], -1)

        decoder_cell = tf.contrib.seq2seq.AttentionWrapper(
                decoder_cell, attention_mechanism,
                attention_layer_size = 2 * hidden_size,
                cell_input_fn = cell_input_fn,
                initial_cell_state = None)
                #initial_cell_state = encoder_state if params['encoder_layer'] == params['decoder_layer'] else None)
        
        if params['if_wean']:
            decoder_cell = wrapper.WeanWrapper(decoder_cell, embedding_q)
        else:
            decoder_cell = tf.contrib.rnn.OutputProjectionWrapper(decoder_cell, voca_size)

        # Helper for decoder cell
        if mode == tf.estimator.ModeKeys.TRAIN:
            helper = tf.contrib.seq2seq.ScheduledEmbeddingTrainingHelper(
                    inputs = embd_q,
                    sequence_length = len_q,
                    embedding = embedding_q,
                    sampling_probability = 0.25)
        else: # EVAL & TEST
            start_token = params['start_token'] * tf.ones([batch_size], dtype = tf.int32)
            helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
                    embedding_q, start_token, params['end_token']
                    )

        # Decoder
        if mode != tf.estimator.ModeKeys.PREDICT or beam_width == 0:
            initial_state = decoder_cell.zero_state(dtype = dtype, batch_size = batch_size)
            if copy_state is not None:
                initial_state = initial_state.clone(cell_state = copy_state)
        
            decoder = tf.contrib.seq2seq.BasicDecoder(
                decoder_cell, helper, initial_state,
                output_layer=None)
        else:
            initial_state = decoder_cell.zero_state(dtype = dtype, batch_size = batch_size * beam_width)
            if copy_state is not None:
                initial_state = initial_state.clone(cell_state = copy_state)
            decoder = tf.contrib.seq2seq.BeamSearchDecoder(
                    cell = decoder_cell,
                    embedding = embedding_q,
                    start_tokens = start_token,
                    end_token = params['end_token'],
                    initial_state = initial_state,
                    beam_width = beam_width,
                    length_penalty_weight = length_penalty_weight)

        # Dynamic decoding
        if mode == tf.estimator.ModeKeys.TRAIN:
            outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder, impute_finished=True, maximum_iterations = None)
            logits_q = outputs.rnn_output
            softmax_q = tf.nn.softmax(logits_q)
            predictions_q = tf.argmax(softmax_q, axis = -1)
        elif mode == tf.estimator.ModeKeys.PREDICT and beam_width > 0:
            outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder, impute_finished = False, maximum_iterations = params['maxlen_q_test'])
            predictions_q = outputs.predicted_ids # [batch, length, beam_width]
            predictions_q = tf.transpose(predictions_q, [0, 2, 1]) # [batch, beam_width, length]
            predictions_q = predictions_q[:, 0, :] # [batch, length]
        else:
            max_iter = params['maxlen_q_test'] if mode == tf.estimator.ModeKeys.PREDICT else params['maxlen_q_dev']
            outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder, impute_finished=True, maximum_iterations = max_iter)
            logits_q = outputs.rnn_output
            softmax_q = tf.nn.softmax(logits_q)
            predictions_q = tf.argmax(softmax_q, axis = -1)       
    
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(
                mode = mode,
                predictions = {
                    'question' : predictions_q
                    })
    # Loss
    label_q = tf.concat([question[:,1:], tf.zeros([batch_size, 1], dtype = tf.int32)], axis = 1, name = 'label_q')
    maxlen_q = params['maxlen_q_train'] if mode == tf.estimator.ModeKeys.TRAIN else params['maxlen_q_dev']
    current_length = tf.shape(logits_q)[1]
    def concat_padding():
        num_pad = maxlen_q - current_length
        padding = tf.zeros([batch_size, num_pad, voca_size], dtype = dtype)

        return tf.concat([logits_q, padding], axis = 1)

    def slice_to_maxlen():
        return tf.slice(logits_q, [0,0,0], [batch_size, maxlen_q, voca_size])

    logits_q = tf.cond(current_length < maxlen_q,
            concat_padding,
            slice_to_maxlen)
    
    weight_q = tf.sequence_mask(len_q, maxlen_q, dtype)

    loss_q = tf.contrib.seq2seq.sequence_loss(
            logits_q, 
            label_q,
            weight_q, # [batch, length]
            average_across_timesteps = True,
            average_across_batch = True,
            softmax_loss_function = None # default : sparse_softmax_cross_entropy
            )
    
    loss = loss_q 

    # eval_metric for estimator
    eval_metric_ops = {
            'bleu' : bleu_score(label_q, predictions_q)
            }

    # Summary
    tf.summary.scalar('loss_question', loss_q)
    tf.summary.scalar('total_loss', loss)


    # Optimizer
    learning_rate = params['learning_rate']
    if params['decay_step'] is not None:
        learning_rate = tf.train.exponential_decay(learning_rate, tf.train.get_global_step(), params['decay_step'], params['decay_rate'], staircase = True)
    optimizer = tf.train.AdamOptimizer(learning_rate)

    grad_and_var = optimizer.compute_gradients(loss, tf.trainable_variables())
    grad, var = zip(*grad_and_var)
    #clipped_grad, norm = tf.clip_by_global_norm(grad, 5)
    train_op = optimizer.apply_gradients(zip(grad, var), global_step = tf.train.get_global_step())

    return tf.estimator.EstimatorSpec(
        mode=mode,
        loss=loss,
        train_op=train_op,
        eval_metric_ops=eval_metric_ops)

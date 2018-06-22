import tensorflow as tf

from tensorflow.python.framework import ops
from tensorflow.python.ops import rnn_cell_impl


RNNCell = rnn_cell_impl.RNNCell
_Linear = rnn_cell_impl._Linear
_like_rnncell = rnn_cell_impl._like_rnncell


class WeanWrapper(RNNCell):
    ''' Implementation of Word Embedding Attention Network(WEAN)
    '''

    def __init__(self, cell, embedding, use_context = True):
        super(WeanWrapper, self).__init__()
        if not _like_rnncell(cell):
            raise TypeError('The parameter cell is not RNNCell.')

        self._cell = cell
        self._embedding = embedding
        self._use_context = use_context
        self._linear = None

    @property
    def state_size(self):
        return self._cell.state_size

    @property
    def output_size(self):
        return self._embedding.get_shape()[0]

    def zero_state(self, batch_size, dtype):
        with ops.name_scope(type(self).__name__ + 'ZeroState', values=[batch_size]):
            return self._cell.zero_state(batch_size, dtype)

    def call(self, inputs, state):
        '''Run the cell and build WEAN over the output'''
        output, res_state = self._cell(inputs, state)

        context = res_state.attention

        hidden_size = output.get_shape()[-1]
        embedding_size = self._embedding.get_shape()[-1]

        if self._use_context == True:
            query = tf.layers.dense(tf.concat([output, context], -1), hidden_size, tf.tanh, name = 'q_t')
        else:
            query = output
        
        qw = tf.layers.dense(query, embedding_size, name = 'qW')
        score = tf.matmul(qw, self._embedding, transpose_b = True, name = 'score')
        return score, res_state


class CopyWrapper(RNNCell):
    ''' Implementation of Copy Mechanism
    '''
    
    def __init__(self, cell, output_size, sentence_index, activation = None):
        super(CopyWrapper, self).__init__()
        if not _like_rnncell(cell):
            raise TypeError('The parameter cell is not RNNCell.')

        self._cell = cell
        self._output_size = output_size
        self._sentence_index = sentence_index
        self._activation = activation
        self._linear = None
        
    @property
    def state_size(self):
        return self._cell.state_size

    @property
    def output_size(self):
        return self._output_size
    
    def zero_state(self, batch_size, dtype):
        with ops.name_scope(type(self).__name__ + "ZeroState", values =[batch_size]):
            return self._cell.zero_state(batch_size, dtype)
    
    def attention_vocab(attention_weight, sentence_index):
        ''' return indices and updates for tf.scatter_nd_update

        Args:
            attention_weight : [batch, length]
            sentence_index : [batch, length]
        '''
        batch_size = attention_weight.get_shape()[0]
        sentencen_length = attention_weight.get_shape()[-1]

        batch_index = tf.range(batch_size)
        batch_index = tf.expand_dims(batch_index, [1])
        batch_index = tf.tile(batch_index, [1, sentence_length])
        batch_index = tf.reshape(batch_index, [-1, 1]) # looks like [0,0,0,0,0,1,1,1,1,1,2,2,2,2,2,....]

        zeros = tf.zeros([batch_size, self._output_size])

        flat_index = tf.reshape(sentence_index, [-1, 1])
        indices = tf.concat([batch_index, flat_index], 1)

        updates = tf.reshape(attention_weight, [-1])

        p_attn = tf.scatter_nd_update(zeros, indices, updates)

        return p_attn


    def call(self, inputs, state):
        current_alignment = state.alignments # attention weight(normalized)
        previous_state = state.cell_state # s(t-1)
        current_attention = state.attention

        # Copy mechanism
        p_attn = attention_vocab(current_alignment, self._sentence_index)

        output, res_state = self._cell(inputs, state)
        if self._linear is None:
            self._linear = _Linear(output, self._output_size, True)
        p_vocab = self._linear(output)
        if self._activation:
            p_vocab = self._activation(projected)

        weighted_c = tf.layers.dense(current_attention, 1)
        weighted_s = tf.layers.dense(previous_state, 1)
        g = tf.sigmoid(weighted_c + weighted_s)
        
        p_final = g * p_vocab + (1-g)*p_attn
        return p_final, res_state


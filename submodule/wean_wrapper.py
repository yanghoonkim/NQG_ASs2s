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

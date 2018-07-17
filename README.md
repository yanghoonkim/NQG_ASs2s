settting 7

if tensorflow 1.8:
  submodule/rnn_wrapper.py should be modified:
    from tensorflow.contrib.rnn.python.ops import core_rnn_cell
    
    _Linear = core_rnn_cell._Linear

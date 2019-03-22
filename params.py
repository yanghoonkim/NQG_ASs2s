import tensorflow as tf

def basic_params():
    '''A set of basic hyperparameters'''
    return tf.contrib.training.HParams(
        dtype = tf.float32,
        voca_size = 34004,
        embedding_trainable = False,
        hidden_size = 800,
        encoder_layer = 1,
        decoder_layer = 1,
        answer_layer = 1,
        dec_init_ans = True,
        
        maxlen_q_train = 32,
        maxlen_q_dev = 27,
        maxlen_q_test = 27,
            
        rnn_dropout = 0.4,
        
        start_token = 1, # <GO> index
        end_token = 2, # <EOS> index
        
        # Keyword-net related parameters
        use_keyword = 2,

        # Attention related parameters
        attn = 'normed_bahdanau',
        
        # Output layer related parameters
        if_wean = True,

        # Training related parameters
        batch_size = 64,
        learning_rate = 0.001,
        decay_step = None,
        decay_rate = 0.5,
        
        # Beam Search
        beam_width = 10,
        length_penalty_weight = 2.1
        )



def h200_batch64():
    params = basic_params()
    params.hidden_size = 200
    params.batch_size = 64
    return params

def h512_batch128():
    params = basic_params()
    params.hidden_size = 512
    params.batch_size = 128
    return params

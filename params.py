import tensorflow as tf

def basic_params():
    '''A set of basic hyperparameters'''
    return tf.contrib.training.HParams(
        dtype = tf.float32,
        voca_size = 34004,
        embedding = '../qa_generation/data/processed/mpqg_substitute_a_vocab_include_a/glove840b_mpqg_vocab300.npy',
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
        
        num_heads = 1,
        context_depth = 512,
        
        # Attention related parameters
        attn = 'normed_bahdanau',
        attn_dropout = 0.4,
        
        # Output layer related parameters
        if_wean = True,
        copy_mechanism = True,

        # Training related parameters
        batch_size = 64,
        learning_rate = 0.001,
        decay_step = None,
        decay_rate = 0.5,
        
        # Beam Search
        beam_width = 10,
        length_penalty_weight = 2.1
        )





def h200_s2s_plus_a_batch64():
    params = basic_params()
    params.hidden_size = 200
    params.batch_size = 64
    params.latent_type_with_a = 0
    params.latent_type_with_s = 0
    params.if_wean = False
    params.copy_mechanism = False
    return params

def h250_s2s_plus_a_batch64():
    params = basic_params()
    params.hidden_size = 250
    params.batch_size = 64
    params.latent_type_with_a = 0
    params.latent_type_with_s = 0
    params.if_wean = False
    params.copy_mechanism = False
    return params

def h300_s2s_plus_a_batch64():
    params = basic_params()
    params.hidden_size = 300
    params.batch_size = 64
    params.latent_type_with_a = 0
    params.latent_type_with_s = 0
    params.if_wean = False
    params.copy_mechanism = False
    return params

def h350_s2s_plus_a_batch64():
    params = basic_params()
    params.hidden_size = 350
    params.batch_size = 64
    params.latent_type_with_a = 0
    params.latent_type_with_s = 0
    params.if_wean = False
    params.copy_mechanism = False
    return params

def h512_s2s_plus_a_batch64():
    params = basic_params()
    params.hidden_size = 512
    params.batch_size = 64
    params.latent_type_with_a = 0
    params.latent_type_with_s = 0
    params.if_wean = False
    params.copy_mechanism = False
    return params

def h800_s2s_plus_a_batch64():
    params = basic_params()
    params.hidden_size = 800
    params.batch_size = 64
    params.latent_type_with_a = 0
    params.latent_type_with_s = 0
    params.if_wean = False
    params.copy_mechanism = False
    return params

def h200_s2s_plus_a_batch128():
    params = basic_params()
    params.hidden_size = 200
    params.batch_size = 128
    params.latent_type_with_a = 0
    params.latent_type_with_s = 0
    params.if_wean = False
    params.copy_mechanism = False
    return params

def h250_s2s_plus_a_batch128():
    params = basic_params()
    params.hidden_size = 250
    params.batch_size = 128
    params.latent_type_with_a = 0
    params.latent_type_with_s = 0
    params.if_wean = False
    params.copy_mechanism = False
    return params

def h300_s2s_plus_a_batch128():
    params = basic_params()
    params.hidden_size = 300
    params.batch_size = 128
    params.latent_type_with_a = 0
    params.latent_type_with_s = 0
    params.if_wean = False
    params.copy_mechanism = False
    return params

def h350_s2s_plus_a_batch128():
    params = basic_params()
    params.hidden_size = 350
    params.batch_size = 128
    params.latent_type_with_a = 0
    params.latent_type_with_s = 0
    params.if_wean = False
    params.copy_mechanism = False
    return params

def h512_s2s_plus_a_batch128():
    params = basic_params()
    params.hidden_size = 512
    params.batch_size = 128
    params.latent_type_with_a = 0
    params.latent_type_with_s = 0
    params.if_wean = False
    params.copy_mechanism = False
    return params

def h800_s2s_plus_a_batch128():
    params = basic_params()
    params.hidden_size = 800
    params.batch_size = 128
    params.latent_type_with_a = 0
    params.latent_type_with_s = 0
    params.if_wean = False
    params.copy_mechanism = False
    return params

def h200_s2s_plus_a_batch256():
    params = basic_params()
    params.hidden_size = 200
    params.batch_size = 256
    params.latent_type_with_a = 0
    params.latent_type_with_s = 0
    params.if_wean = False
    params.copy_mechanism = False
    return params

def h250_s2s_plus_a_batch256():
    params = basic_params()
    params.hidden_size = 250
    params.batch_size = 256
    params.latent_type_with_a = 0
    params.latent_type_with_s = 0
    params.if_wean = False
    params.copy_mechanism = False
    return params

def h300_s2s_plus_a_batch256():
    params = basic_params()
    params.hidden_size = 300
    params.batch_size = 256
    params.latent_type_with_a = 0
    params.latent_type_with_s = 0
    params.if_wean = False
    params.copy_mechanism = False
    return params

def h350_s2s_plus_a_batch256():
    params = basic_params()
    params.hidden_size = 350
    params.batch_size = 256
    params.latent_type_with_a = 0
    params.latent_type_with_s = 0
    params.if_wean = False
    params.copy_mechanism = False
    return params

def h512_s2s_plus_a_batch256():
    params = basic_params()
    params.hidden_size = 512
    params.batch_size = 256
    params.latent_type_with_a = 0
    params.latent_type_with_s = 0
    params.if_wean = False
    params.copy_mechanism = False
    return params

def h800_s2s_plus_a_batch256():
    params = basic_params()
    params.hidden_size = 800
    params.batch_size = 256
    params.latent_type_with_a = 0
    params.latent_type_with_s = 0
    params.if_wean = False
    params.copy_mechanism = False
    return params

def h200_s2s_plus_a_batch400():
    params = basic_params()
    params.hidden_size = 200
    params.batch_size = 400
    params.latent_type_with_a = 0
    params.latent_type_with_s = 0
    params.if_wean = False
    params.copy_mechanism = False
    return params

def h250_s2s_plus_a_batch400():
    params = basic_params()
    params.hidden_size = 250
    params.batch_size = 400
    params.latent_type_with_a = 0
    params.latent_type_with_s = 0
    params.if_wean = False
    params.copy_mechanism = False
    return params

def h300_s2s_plus_a_batch400():
    params = basic_params()
    params.hidden_size = 300
    params.batch_size = 400
    params.latent_type_with_a = 0
    params.latent_type_with_s = 0
    params.if_wean = False
    params.copy_mechanism = False
    return params

def h350_s2s_plus_a_batch400():
    params = basic_params()
    params.hidden_size = 350
    params.batch_size = 400
    params.latent_type_with_a = 0
    params.latent_type_with_s = 0
    params.if_wean = False
    params.copy_mechanism = False
    return params

def h512_s2s_plus_a_batch400():
    params = basic_params()
    params.hidden_size = 512
    params.batch_size = 400
    params.latent_type_with_a = 0
    params.latent_type_with_s = 0
    params.if_wean = False
    params.copy_mechanism = False
    return params

def h800_s2s_plus_a_batch400():
    params = basic_params()
    params.hidden_size = 800
    params.batch_size = 400
    params.latent_type_with_a = 0
    params.latent_type_with_s = 0
    params.if_wean = False
    params.copy_mechanism = False
    return params


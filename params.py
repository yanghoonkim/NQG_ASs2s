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
        
        # Memory network related parameters
        latent_type_with_s = 2,
        latent_type_with_a = 0,

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
   


def h800_divert_s1_batch128():
    params = basic_params()
    params.latent_type_with_s = 1
    params.latent_type_with_a = 0
    params.batch_size = 128
    return params

def h800_divert_s2_batch128():
    params = basic_params()
    params.latent_type_with_s = 2
    params.latent_type_with_a = 0
    params.batch_size = 128
    return params

def h800_divert_s8_batch128():
    params = basic_params()
    params.latent_type_with_s = 8
    params.latent_type_with_a = 0
    params.batch_size = 128
    return params

def h800_divert_s20_batch128():
    params = basic_params()
    params.latent_type_with_s = 20
    params.latent_type_with_a = 0
    params.batch_size = 128
    return params

def h800_divert_s50_batch128():
    params = basic_params()
    params.latent_type_with_s = 50
    params.latent_type_with_a = 0
    params.batch_size = 128
    return params

def h800_divert_s100_batch128():
    params = basic_params()
    params.latent_type_with_s = 100
    params.latent_type_with_a = 0
    params.batch_size = 128
    return params    

def h800_divert_s256_batch128():
    params = basic_params()
    params.latent_type_with_s = 256
    params.latent_type_with_a = 0
    params.batch_size = 128
    return params

def h800_memory_a_8():
    params = h800_divert_s1_batch128()
    params.latent_type_with_a = 8
    return params

def h800_memory_a_30():
    params = h800_memory_a_1()
    params.latent_type_with_a = 30
    return params

def h800_memory_a_60():
    params = h800_memory_a_1()
    params.latent_type_with_a = 60
    return params

def h800_memory_s_1():
    params = basic_params()
    params.latent_type_with_s = 1
    return params

def h800_memory_s_5():
    params = basic_params()
    params.latent_type_with_s = 5
    return params

def h800_memory_s_20():
    params = basic_params()
    params.latent_type_with_s = 20
    return params


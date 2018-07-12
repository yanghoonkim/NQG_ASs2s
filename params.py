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
   


def h800_memory_a_1():
    params = basic_params()
    params.latent_type_with_s = 0
    params.latent_type_with_a = 1
    return params

def h800_memory_a_2():
    params = h800_memory_a_1()
    params.latent_type_with_a = 2
    return params

def h800_memory_a_3():
    params = h800_memory_a_1()
    params.latent_type_with_a = 3
    return params

def h800_memory_a_5():
    params = h800_memory_a_1()
    params.latent_type_with_a = 5
    return params

def h800_memory_a_8():
    params = h800_memory_a_1()
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

##

def h300_memory_setting5_batch64():
    params = basic_params()
    params.hidden_size = 300
    return params

def h300_memory_setting5_batch32():
    params = basic_params()
    params.hidden_size = 300
    params.batch_size = 32
    return params

def h512_memory_setting5_batch64():
    params = basic_params()
    params.hidden_size = 512
    return params

def h512_memory_setting5_batch128():
    params = basic_params()
    params.hidden_size = 512
    params.batch_size = 128
    return params

def h1024_memory_setting5_batch64():
    params = basic_params()
    params.hidden_size = 1024
    params.batch_size = 64
    return params

def h350_memory_setting5_batch128():
    params = basic_params()
    params.hidden_size = 350
    params.batch_size = 128
    return params

def h800_memory_setting5_batch128():
    params = basic_params()
    params.hidden_size = 800
    params.batch_size = 128
    return params
###

def h200_memory_setting5_batch64_2layer():
    params = basic_params()
    params.hidden_size = 200
    params.encoder_layer = 2
    params.decoder_layer = 2
    params.answer_layer = 2
    return params

def h350_memory_setting5_batch64_2layer():
    params = h200_memory_setting4_batch64_2layer()
    params.hidden_size = 350
    return params

def h512_memory_setting5_batch64_2layer():
    params = h200_memory_setting4_batch64_2layer()
    params.hidden_size = 512
    return params

def h512_memory_setting5_batch64_212layer():
    params = basic_params()
    params.hidden_size = 512
    params.encoder_layer = 2
    params.decoder_layer = 1
    params.answer_layer = 2
    return params

def h512_memory_setting5_batch64_121layer():
    params = basic_params()
    params.hidden_size = 512
    params.decoder_layer = 2
    return params

def h512_memory_setting5_batch64_221layer():
    params = basic_params()
    params.encoder_layer = 2
    params.decoder_layer = 2
    params.answer_layer = 1
    params.hidden_size = 512
    return params

def h512_memory_setting5_batch64_112layer():
    params = basic_params()
    params.answer_layer = 2
    params.hidden_size = 512
    return params

def h350_memory_setting5_batch256():
    params = basic_params()
    params.hidden_size = 350
    params.batch_size = 256
    return params

def h512_memory_setting5_batch256():
    params = basic_params()
    params.hidden_size = 512
    params.batch_size = 256
    return params

def h250_memory_setting5_batch128():
    params = basic_params()
    params.hidden_size = 250
    params.batch_size = 128
    return params

def h250_memory_setting5_batch256():
    params = basic_params()
    params.hidden_size = 250
    params.batch_size = 256
    return params



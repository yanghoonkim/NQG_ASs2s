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
        latent_type_with_s = 0,
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
   


def h200_setting7_batch32():
    params = basic_params()
    params.hidden_size = 200
    params.batch_size = 32
    params.latent_type_with_a = 2
    return params

def h200_setting7_batch64():
    params = basic_params()
    params.hidden_size = 200
    params.batch_size = 64
    params.latent_type_with_a = 8
    return params

def h200_setting7_batch128():
    params = basic_params()
    params.hidden_size = 200
    params.batch_size = 128
    params.latent_type_with_a = 20
    return params

def h200_setting7_batch256():
    params = basic_params()
    params.hidden_size = 200
    params.batch_size = 256
    params.latent_type_with_a = 50
    return params

def h350_setting7_batch32():
    params = basic_params()
    params.hidden_size = 350
    params.batch_size = 32
    params.latent_type_with_a = 2
    return params

def h350_setting7_batch64():
    params = basic_params()
    params.hidden_size = 350
    params.batch_size = 64
    params.latent_type_with_a = 8
    return params

def h350_setting7_batch128():
    params = basic_params()
    params.hidden_size = 350
    params.batch_size = 128
    params.latent_type_with_a = 20
    return params

def h350_setting7_batch256():
    params = basic_params()
    params.hidden_size = 350
    params.batch_size = 256
    params.latent_type_with_a = 50
    return params

def h512_setting7_batch32():
    params = basic_params()
    params.hidden_size = 512
    params.batch_size = 32
    params.latent_type_with_a = 2
    return params

def h512_setting7_batch64():
    params = basic_params()
    params.hidden_size = 512
    params.batch_size = 64
    params.latent_type_with_a = 8
    return params

def h512_setting7_batch128():
    params = basic_params()
    params.hidden_size = 512
    params.batch_size = 128
    params.latent_type_with_a = 20
    return params

def h512_setting7_batch256():
    params = basic_params()
    params.hidden_size = 512
    params.batch_size = 256
    params.latent_type_with_a = 50
    return params

def h800_setting7_batch32():
    params = basic_params()
    params.hidden_size = 800
    params.batch_size = 32
    params.latent_type_with_a = 2
    return params

def h800_setting7_batch64():
    params = basic_params()
    params.hidden_size = 800
    params.batch_size = 64
    params.latent_type_with_a = 8
    return params

def h800_setting7_batch128():
    params = basic_params()
    params.hidden_size = 800
    params.batch_size = 128
    params.latent_type_with_a = 20
    return params

def h800_setting7_batch256():
    params = basic_params()
    params.hidden_size = 800
    params.batch_size = 256
    params.latent_type_with_a = 50
    return params

# 

def h100_setting7_m1_batch128():
    params = basic_params()
    params.hidden_size = 100
    params.batch_size = 128
    params.latent_type_with_a = 1
    return params

def h100_setting7_m2_batch128():
    params = basic_params()
    params.hidden_size = 100
    params.batch_size = 128
    params.latent_type_with_a = 2
    return params

def h100_setting7_m4_batch128():
    params = basic_params()
    params.hidden_size = 100
    params.batch_size = 128
    params.latent_type_with_a = 4
    return params

def h100_setting7_m8_batch128():
    params = basic_params()
    params.hidden_size = 100
    params.batch_size = 128
    params.latent_type_with_a = 8
    return params

def h100_setting7_m20_batch128():
    params = basic_params()
    params.hidden_size = 100
    params.batch_size = 128
    params.latent_type_with_a = 20
    return params

def h100_setting7_m50_batch128():
    params = basic_params()
    params.hidden_size = 100
    params.batch_size = 128
    params.latent_type_with_a = 50
    return params

def h100_setting7_m100_batch128():
    params = basic_params()
    params.hidden_size = 100
    params.batch_size = 128
    params.latent_type_with_a = 100
    return params

def h100_setting7_m256_batch128():
    params = basic_params()
    params.hidden_size = 100
    params.batch_size = 128
    params.latent_type_with_a = 256
    return params

def h200_setting7_m1_batch128():
    params = basic_params()
    params.hidden_size = 200
    params.batch_size = 128
    params.latent_type_with_a = 1
    return params

def h200_setting7_m2_batch128():
    params = basic_params()
    params.hidden_size = 200
    params.batch_size = 128
    params.latent_type_with_a = 2
    return params

def h200_setting7_m4_batch128():
    params = basic_params()
    params.hidden_size = 200
    params.batch_size = 128
    params.latent_type_with_a = 4
    return params

def h200_setting7_m8_batch128():
    params = basic_params()
    params.hidden_size = 200
    params.batch_size = 128
    params.latent_type_with_a = 8
    return params

def h200_setting7_m20_batch128():
    params = basic_params()
    params.hidden_size = 200
    params.batch_size = 128
    params.latent_type_with_a = 20
    return params

def h200_setting7_m50_batch128():
    params = basic_params()
    params.hidden_size = 200
    params.batch_size = 128
    params.latent_type_with_a = 50
    return params

def h200_setting7_m100_batch128():
    params = basic_params()
    params.hidden_size = 200
    params.batch_size = 128
    params.latent_type_with_a = 100
    return params

def h200_setting7_m256_batch128():
    params = basic_params()
    params.hidden_size = 200
    params.batch_size = 128
    params.latent_type_with_a = 256
    return params

def h350_setting7_m1_batch128():
    params = basic_params()
    params.hidden_size = 350
    params.batch_size = 128
    params.latent_type_with_a = 1
    return params

def h350_setting7_m2_batch128():
    params = basic_params()
    params.hidden_size = 350
    params.batch_size = 128
    params.latent_type_with_a = 2
    return params

def h350_setting7_m4_batch128():
    params = basic_params()
    params.hidden_size = 350
    params.batch_size = 128
    params.latent_type_with_a = 4
    return params

def h350_setting7_m8_batch128():
    params = basic_params()
    params.hidden_size = 350
    params.batch_size = 128
    params.latent_type_with_a = 8
    return params

def h350_setting7_m20_batch128():
    params = basic_params()
    params.hidden_size = 350
    params.batch_size = 128
    params.latent_type_with_a = 20
    return params

def h350_setting7_m50_batch128():
    params = basic_params()
    params.hidden_size = 350
    params.batch_size = 128
    params.latent_type_with_a = 50
    return params

def h350_setting7_m100_batch128():
    params = basic_params()
    params.hidden_size = 350
    params.batch_size = 128
    params.latent_type_with_a = 100
    return params

def h350_setting7_m256_batch128():
    params = basic_params()
    params.hidden_size = 350
    params.batch_size = 128
    params.latent_type_with_a = 256
    return params

def h512_setting7_m1_batch128():
    params = basic_params()
    params.hidden_size = 512
    params.batch_size = 128
    params.latent_type_with_a = 1
    return params

def h512_setting7_m2_batch128():
    params = basic_params()
    params.hidden_size = 512
    params.batch_size = 128
    params.latent_type_with_a = 2
    return params

def h512_setting7_m4_batch128():
    params = basic_params()
    params.hidden_size = 512
    params.batch_size = 128
    params.latent_type_with_a = 4
    return params

def h512_setting7_m8_batch128():
    params = basic_params()
    params.hidden_size = 512
    params.batch_size = 128
    params.latent_type_with_a = 8
    return params

def h512_setting7_m20_batch128():
    params = basic_params()
    params.hidden_size = 512
    params.batch_size = 128
    params.latent_type_with_a = 20
    return params

def h512_setting7_m50_batch128():
    params = basic_params()
    params.hidden_size = 512
    params.batch_size = 128
    params.latent_type_with_a = 50
    return params

def h512_setting7_m100_batch128():
    params = basic_params()
    params.hidden_size = 512
    params.batch_size = 128
    params.latent_type_with_a = 100
    return params

def h512_setting7_m256_batch128():
    params = basic_params()
    params.hidden_size = 512
    params.batch_size = 128
    params.latent_type_with_a = 256
    return params

def h100_setting7_m1_batch256():
    params = basic_params()
    params.hidden_size = 100
    params.batch_size = 256
    params.latent_type_with_a = 1
    return params

def h100_setting7_m2_batch256():
    params = basic_params()
    params.hidden_size = 100
    params.batch_size = 256
    params.latent_type_with_a = 2
    return params

def h100_setting7_m4_batch256():
    params = basic_params()
    params.hidden_size = 100
    params.batch_size = 256
    params.latent_type_with_a = 4
    return params

def h100_setting7_m8_batch256():
    params = basic_params()
    params.hidden_size = 100
    params.batch_size = 256
    params.latent_type_with_a = 8
    return params

def h100_setting7_m20_batch256():
    params = basic_params()
    params.hidden_size = 100
    params.batch_size = 256
    params.latent_type_with_a = 20
    return params

def h100_setting7_m50_batch256():
    params = basic_params()
    params.hidden_size = 100
    params.batch_size = 256
    params.latent_type_with_a = 50
    return params

def h100_setting7_m100_batch256():
    params = basic_params()
    params.hidden_size = 100
    params.batch_size = 256
    params.latent_type_with_a = 100
    return params

def h100_setting7_m256_batch256():
    params = basic_params()
    params.hidden_size = 100
    params.batch_size = 256
    params.latent_type_with_a = 256
    return params

def h200_setting7_m1_batch256():
    params = basic_params()
    params.hidden_size = 200
    params.batch_size = 256
    params.latent_type_with_a = 1
    return params

def h200_setting7_m2_batch256():
    params = basic_params()
    params.hidden_size = 200
    params.batch_size = 256
    params.latent_type_with_a = 2
    return params

def h200_setting7_m4_batch256():
    params = basic_params()
    params.hidden_size = 200
    params.batch_size = 256
    params.latent_type_with_a = 4
    return params

def h200_setting7_m8_batch256():
    params = basic_params()
    params.hidden_size = 200
    params.batch_size = 256
    params.latent_type_with_a = 8
    return params

def h200_setting7_m20_batch256():
    params = basic_params()
    params.hidden_size = 200
    params.batch_size = 256
    params.latent_type_with_a = 20
    return params

def h200_setting7_m50_batch256():
    params = basic_params()
    params.hidden_size = 200
    params.batch_size = 256
    params.latent_type_with_a = 50
    return params

def h200_setting7_m100_batch256():
    params = basic_params()
    params.hidden_size = 200
    params.batch_size = 256
    params.latent_type_with_a = 100
    return params

def h200_setting7_m256_batch256():
    params = basic_params()
    params.hidden_size = 200
    params.batch_size = 256
    params.latent_type_with_a = 256
    return params

def h350_setting7_m1_batch256():
    params = basic_params()
    params.hidden_size = 350
    params.batch_size = 256
    params.latent_type_with_a = 1
    return params

def h350_setting7_m2_batch256():
    params = basic_params()
    params.hidden_size = 350
    params.batch_size = 256
    params.latent_type_with_a = 2
    return params

def h350_setting7_m4_batch256():
    params = basic_params()
    params.hidden_size = 350
    params.batch_size = 256
    params.latent_type_with_a = 4
    return params

def h350_setting7_m8_batch256():
    params = basic_params()
    params.hidden_size = 350
    params.batch_size = 256
    params.latent_type_with_a = 8
    return params

def h350_setting7_m20_batch256():
    params = basic_params()
    params.hidden_size = 350
    params.batch_size = 256
    params.latent_type_with_a = 20
    return params

def h350_setting7_m50_batch256():
    params = basic_params()
    params.hidden_size = 350
    params.batch_size = 256
    params.latent_type_with_a = 50
    return params

def h350_setting7_m100_batch256():
    params = basic_params()
    params.hidden_size = 350
    params.batch_size = 256
    params.latent_type_with_a = 100
    return params

def h350_setting7_m256_batch256():
    params = basic_params()
    params.hidden_size = 350
    params.batch_size = 256
    params.latent_type_with_a = 256
    return params

def h512_setting7_m1_batch256():
    params = basic_params()
    params.hidden_size = 512
    params.batch_size = 256
    params.latent_type_with_a = 1
    return params

def h512_setting7_m2_batch256():
    params = basic_params()
    params.hidden_size = 512
    params.batch_size = 256
    params.latent_type_with_a = 2
    return params

def h512_setting7_m4_batch256():
    params = basic_params()
    params.hidden_size = 512
    params.batch_size = 256
    params.latent_type_with_a = 4
    return params

def h512_setting7_m8_batch256():
    params = basic_params()
    params.hidden_size = 512
    params.batch_size = 256
    params.latent_type_with_a = 8
    return params

def h512_setting7_m20_batch256():
    params = basic_params()
    params.hidden_size = 512
    params.batch_size = 256
    params.latent_type_with_a = 20
    return params

def h512_setting7_m50_batch256():
    params = basic_params()
    params.hidden_size = 512
    params.batch_size = 256
    params.latent_type_with_a = 50
    return params

def h512_setting7_m100_batch256():
    params = basic_params()
    params.hidden_size = 512
    params.batch_size = 256
    params.latent_type_with_a = 100
    return params

def h512_setting7_m256_batch256():
    params = basic_params()
    params.hidden_size = 512
    params.batch_size = 256
    params.latent_type_with_a = 256
    return params

def h100_setting7_m1_batch512():
    params = basic_params()
    params.hidden_size = 100
    params.batch_size = 512
    params.latent_type_with_a = 1
    return params

def h100_setting7_m2_batch512():
    params = basic_params()
    params.hidden_size = 100
    params.batch_size = 512
    params.latent_type_with_a = 2
    return params

def h100_setting7_m4_batch512():
    params = basic_params()
    params.hidden_size = 100
    params.batch_size = 512
    params.latent_type_with_a = 4
    return params

def h100_setting7_m8_batch512():
    params = basic_params()
    params.hidden_size = 100
    params.batch_size = 512
    params.latent_type_with_a = 8
    return params

def h100_setting7_m20_batch512():
    params = basic_params()
    params.hidden_size = 100
    params.batch_size = 512
    params.latent_type_with_a = 20
    return params

def h100_setting7_m50_batch512():
    params = basic_params()
    params.hidden_size = 100
    params.batch_size = 512
    params.latent_type_with_a = 50
    return params

def h100_setting7_m100_batch512():
    params = basic_params()
    params.hidden_size = 100
    params.batch_size = 512
    params.latent_type_with_a = 100
    return params

def h100_setting7_m256_batch512():
    params = basic_params()
    params.hidden_size = 100
    params.batch_size = 512
    params.latent_type_with_a = 256
    return params

def h200_setting7_m1_batch512():
    params = basic_params()
    params.hidden_size = 200
    params.batch_size = 512
    params.latent_type_with_a = 1
    return params

def h200_setting7_m2_batch512():
    params = basic_params()
    params.hidden_size = 200
    params.batch_size = 512
    params.latent_type_with_a = 2
    return params

def h200_setting7_m4_batch512():
    params = basic_params()
    params.hidden_size = 200
    params.batch_size = 512
    params.latent_type_with_a = 4
    return params

def h200_setting7_m8_batch512():
    params = basic_params()
    params.hidden_size = 200
    params.batch_size = 512
    params.latent_type_with_a = 8
    return params

def h200_setting7_m20_batch512():
    params = basic_params()
    params.hidden_size = 200
    params.batch_size = 512
    params.latent_type_with_a = 20
    return params

def h200_setting7_m50_batch512():
    params = basic_params()
    params.hidden_size = 200
    params.batch_size = 512
    params.latent_type_with_a = 50
    return params

def h200_setting7_m100_batch512():
    params = basic_params()
    params.hidden_size = 200
    params.batch_size = 512
    params.latent_type_with_a = 100
    return params

def h200_setting7_m256_batch512():
    params = basic_params()
    params.hidden_size = 200
    params.batch_size = 512
    params.latent_type_with_a = 256
    return params

def h350_setting7_m1_batch512():
    params = basic_params()
    params.hidden_size = 350
    params.batch_size = 512
    params.latent_type_with_a = 1
    return params

def h350_setting7_m2_batch512():
    params = basic_params()
    params.hidden_size = 350
    params.batch_size = 512
    params.latent_type_with_a = 2
    return params

def h350_setting7_m4_batch512():
    params = basic_params()
    params.hidden_size = 350
    params.batch_size = 512
    params.latent_type_with_a = 4
    return params

def h350_setting7_m8_batch512():
    params = basic_params()
    params.hidden_size = 350
    params.batch_size = 512
    params.latent_type_with_a = 8
    return params

def h350_setting7_m20_batch512():
    params = basic_params()
    params.hidden_size = 350
    params.batch_size = 512
    params.latent_type_with_a = 20
    return params

def h350_setting7_m50_batch512():
    params = basic_params()
    params.hidden_size = 350
    params.batch_size = 512
    params.latent_type_with_a = 50
    return params

def h350_setting7_m100_batch512():
    params = basic_params()
    params.hidden_size = 350
    params.batch_size = 512
    params.latent_type_with_a = 100
    return params

def h350_setting7_m256_batch512():
    params = basic_params()
    params.hidden_size = 350
    params.batch_size = 512
    params.latent_type_with_a = 256
    return params

def h512_setting7_m1_batch512():
    params = basic_params()
    params.hidden_size = 512
    params.batch_size = 512
    params.latent_type_with_a = 1
    return params

def h512_setting7_m2_batch512():
    params = basic_params()
    params.hidden_size = 512
    params.batch_size = 512
    params.latent_type_with_a = 2
    return params

def h512_setting7_m4_batch512():
    params = basic_params()
    params.hidden_size = 512
    params.batch_size = 512
    params.latent_type_with_a = 4
    return params

def h512_setting7_m8_batch512():
    params = basic_params()
    params.hidden_size = 512
    params.batch_size = 512
    params.latent_type_with_a = 8
    return params

def h512_setting7_m20_batch512():
    params = basic_params()
    params.hidden_size = 512
    params.batch_size = 512
    params.latent_type_with_a = 20
    return params

def h512_setting7_m50_batch512():
    params = basic_params()
    params.hidden_size = 512
    params.batch_size = 512
    params.latent_type_with_a = 50
    return params

def h512_setting7_m100_batch512():
    params = basic_params()
    params.hidden_size = 512
    params.batch_size = 512
    params.latent_type_with_a = 100
    return params

def h512_setting7_m256_batch512():
    params = basic_params()
    params.hidden_size = 512
    params.batch_size = 512
    params.latent_type_with_a = 256
    return params


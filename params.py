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
        
        num_heads = 1,
        context_depth = 512,
        
        # Memory network related parameters
        use_memorynet = 2,

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



def h200_setting5_batch64():
    params = basic_params()
    params.hidden_size = 200
    params.batch_size = 64
    params.latent_type_with_s = 1
     
    params.beam_width = 0
    return params

def h200_setting5_batch128():
    params = basic_params()
    params.hidden_size = 200
    params.batch_size = 128
    params.latent_type_with_s = 1
     
    return params

def h200_setting5_batch256():
    params = basic_params()
    params.hidden_size = 200
    params.batch_size = 256
    params.latent_type_with_s = 1

    return params

def h200_setting5_batch400():
    params = basic_params()
    params.hidden_size = 200
    params.batch_size = 400
    params.latent_type_with_s = 1
     
    return params

def h250_setting5_batch64():
    params = basic_params()
    params.hidden_size = 250
    params.batch_size = 64
    params.latent_type_with_s = 1
     
    return params

def h250_setting5_batch128():
    params = basic_params()
    params.hidden_size = 250
    params.batch_size = 128
    params.latent_type_with_s = 1
     
    return params

def h250_setting5_batch256():
    params = basic_params()
    params.hidden_size = 250
    params.batch_size = 256
    params.latent_type_with_s = 1
     
    return params

def h250_setting5_batch400():
    params = basic_params()
    params.hidden_size = 250
    params.batch_size = 400
    params.latent_type_with_s = 1
     
    return params

def h300_setting5_batch64():
    params = basic_params()
    params.hidden_size = 300
    params.batch_size = 64
    params.latent_type_with_s = 1
     
    return params

def h300_setting5_batch128():
    params = basic_params()
    params.hidden_size = 300
    params.batch_size = 128
    params.latent_type_with_s = 1
     
    return params

def h300_setting5_batch256():
    params = basic_params()
    params.hidden_size = 300
    params.batch_size = 256
    params.latent_type_with_s = 1
     
    return params

def h300_setting5_batch400():
    params = basic_params()
    params.hidden_size = 300
    params.batch_size = 400
    params.latent_type_with_s = 1
     
    return params

def h350_setting5_batch64():
    params = basic_params()
    params.hidden_size = 350
    params.batch_size = 64
    params.latent_type_with_s = 1
     
    return params

def h350_setting5_batch128():
    params = basic_params()
    params.hidden_size = 350
    params.batch_size = 128
    params.latent_type_with_s = 1
     
    return params

def h350_setting5_batch256():
    params = basic_params()
    params.hidden_size = 350
    params.batch_size = 256
    params.latent_type_with_s = 1
     
    return params

def h350_setting5_batch400():
    params = basic_params()
    params.hidden_size = 350
    params.batch_size = 400
    params.latent_type_with_s = 1
     
    return params

def h400_setting5_batch64():
    params = basic_params()
    params.hidden_size = 400
    params.batch_size = 64
    params.latent_type_with_s = 1
     
    return params

def h400_setting5_batch128():
    params = basic_params()
    params.hidden_size = 400
    params.batch_size = 128
    params.latent_type_with_s = 1
     
    return params

def h400_setting5_batch256():
    params = basic_params()
    params.hidden_size = 400
    params.batch_size = 256
    params.latent_type_with_s = 1
     
    return params

def h400_setting5_batch400():
    params = basic_params()
    params.hidden_size = 400
    params.batch_size = 400
    params.latent_type_with_s = 1
     
    return params

def h450_setting5_batch64():
    params = basic_params()
    params.hidden_size = 450
    params.batch_size = 64
    params.latent_type_with_s = 1
     
    return params

def h450_setting5_batch128():
    params = basic_params()
    params.hidden_size = 450
    params.batch_size = 128
    params.latent_type_with_s = 1
     
    return params

def h450_setting5_batch256():
    params = basic_params()
    params.hidden_size = 450
    params.batch_size = 256
    params.latent_type_with_s = 1
     
    return params

def h450_setting5_batch400():
    params = basic_params()
    params.hidden_size = 450
    params.batch_size = 400
    params.latent_type_with_s = 1
     
    return params

def h512_setting5_batch64():
    params = basic_params()
    params.hidden_size = 512
    params.batch_size = 64
    params.latent_type_with_s = 1
     
    return params

def h512_setting5_batch128():
    params = basic_params()
    params.hidden_size = 512
    params.batch_size = 128
    params.latent_type_with_s = 1
     
    return params

def h512_setting5_batch256():
    params = basic_params()
    params.hidden_size = 512
    params.batch_size = 256
    params.latent_type_with_s = 1
     
    return params

def h512_setting5_batch400():
    params = basic_params()
    params.hidden_size = 512
    params.batch_size = 400
    params.latent_type_with_s = 1
     
    return params

def h800_setting5_batch64():
    params = basic_params()
    params.hidden_size = 800
    params.batch_size = 64
    params.latent_type_with_s = 1
     
    return params

def h800_setting5_batch128():
    params = basic_params()
    params.hidden_size = 800
    params.batch_size = 128
    params.latent_type_with_s = 1
     
    return params

def h800_setting5_batch256():
    params = basic_params()
    params.hidden_size = 800
    params.batch_size = 256
    params.latent_type_with_s = 1
     
    return params

def h800_setting5_batch400():
    params = basic_params()
    params.hidden_size = 800
    params.batch_size = 400
    params.latent_type_with_s = 1
     
    return params



##

def rnn2_attn2():
    params = basic_params()
    params.hidden_size = 250
    params.batch_size = 128
     
    params.rnn_dropout = 0.2
    params.attn_dropout = 0.2
    return params
    
def rnn2_attn4():
    params = basic_params()
    params.hidden_size = 250
    params.batch_size = 128
     
    params.rnn_dropout = 0.2
    params.attn_dropout = 0.4
    return params
    
def rnn2_attn6():
    params = basic_params()
    params.hidden_size = 250
    params.batch_size = 128
     
    params.rnn_dropout = 0.2
    params.attn_dropout = 0.6
    return params
    
def rnn4_attn2():
    params = basic_params()
    params.hidden_size = 250
    params.batch_size = 128
     
    params.rnn_dropout = 0.4
    params.attn_dropout = 0.2
    return params
    
def rnn4_attn4():
    params = basic_params()
    params.hidden_size = 250
    params.batch_size = 128
     
    params.rnn_dropout = 0.4
    params.attn_dropout = 0.4
    return params
    
def rnn4_attn6():
    params = basic_params()
    params.hidden_size = 250
    params.batch_size = 128
     
    params.rnn_dropout = 0.4
    params.attn_dropout = 0.6
    return params
    
def rnn6_attn2():
    params = basic_params()
    params.hidden_size = 250
    params.batch_size = 128
     
    params.rnn_dropout = 0.6
    params.attn_dropout = 0.2
    return params
    
def rnn6_attn4():
    params = basic_params()
    params.hidden_size = 250
    params.batch_size = 128
     
    params.rnn_dropout = 0.6
    params.attn_dropout = 0.4
    return params
    
def rnn6_attn6():
    params = basic_params()
    params.hidden_size = 250
    params.batch_size = 128
     
    params.rnn_dropout = 0.6
    params.attn_dropout = 0.6
    return params


def final_h200_setting5_batch128():
    params = basic_params()
    params.hidden_size = 200
    params.batch_size = 128
    return params

def final_h200_setting5_batch256():
    params = basic_params()
    params.hidden_size = 200
    params.batch_size = 256
    return params

def final_h300_setting5_batch128():
    params = basic_params()
    params.hidden_size = 300
    params.batch_size = 128
    return params

def final_h300_setting5_batch256():
    params = basic_params()
    params.hidden_size = 300
    params.batch_size = 256
    return params

def final_h350_setting5_batch128():
    params = basic_params()
    params.hidden_size = 350
    params.batch_size = 128
    return params

def final_h350_setting5_batch256():
    params = basic_params()
    params.hidden_size = 350
    params.batch_size = 256
    return params

def final_h450_setting5_batch128():
    params = basic_params()
    params.hidden_size = 450
    params.batch_size = 128
    return params

def final_h450_setting5_batch256():
    params = basic_params()
    params.hidden_size = 450
    params.batch_size = 256
    return params

def final_h512_setting5_batch128():
    params = basic_params()
    params.hidden_size = 512
    params.batch_size = 128
    return params

def final_h512_setting5_batch256():
    params = basic_params()
    params.hidden_size = 512
    params.batch_size = 256
    return params


def ner_h200_setting5_batch128():
    params = basic_params()
    params.hidden_size = 200
    params.batch_size = 128
    return params

def ner_h200_setting5_batch256():
    params = basic_params()
    params.hidden_size = 200
    params.batch_size = 256
    return params

def ner_h250_setting5_batch128():
    params = basic_params()
    params.hidden_size = 250
    params.batch_size = 128
    return params

def ner_h250_setting5_batch256():
    params = basic_params()
    params.hidden_size = 250
    params.batch_size = 256
    return params

def ner_h300_setting5_batch128():
    params = basic_params()
    params.hidden_size = 300
    params.batch_size = 128
    return params

def ner_h300_setting5_batch256():
    params = basic_params()
    params.hidden_size = 300
    params.batch_size = 256
    return params

def ner_h350_setting5_batch128():
    params = basic_params()
    params.hidden_size = 350
    params.batch_size = 128
    return params

def ner_h350_setting5_batch256():
    params = basic_params()
    params.hidden_size = 350
    params.batch_size = 256
    return params

def ner_h400_setting5_batch128():
    params = basic_params()
    params.hidden_size = 400
    params.batch_size = 128
    return params

def ner_h400_setting5_batch256():
    params = basic_params()
    params.hidden_size = 400
    params.batch_size = 256
    return params

def ner_h450_setting5_batch128():
    params = basic_params()
    params.hidden_size = 450
    params.batch_size = 128
    return params

def ner_h450_setting5_batch256():
    params = basic_params()
    params.hidden_size = 450
    params.batch_size = 256
    return params

def ner_h512_setting5_batch128():
    params = basic_params()
    params.hidden_size = 512
    params.batch_size = 128
    return params

def ner_h512_setting5_batch256():
    params = basic_params()
    params.hidden_size = 512
    params.batch_size = 256
    return params




def ner_h350_setting5_batch128_1():
    params = basic_params()
    params.hidden_size = 350
    params.batch_size = 128
    return params
    
def ner_h350_setting5_batch128_2():
    params = basic_params()
    params.hidden_size = 350
    params.batch_size = 128
    return params

def ner_h350_setting5_batch128_3():
    params = basic_params()
    params.hidden_size = 350
    params.batch_size = 128
    return params
    
def ner_h350_setting5_batch128_4():
    params = basic_params()
    params.hidden_size = 350
    params.batch_size = 128
    return params

def ner_h350_setting5_batch128_5():
    params = basic_params()
    params.hidden_size = 350
    params.batch_size = 128
    return params
    
def ner_h350_setting5_batch128_6():
    params = basic_params()
    params.hidden_size = 350
    params.batch_size = 128
    return params

def ner_h350_setting5_batch128_7():
    params = basic_params()
    params.hidden_size = 350
    params.batch_size = 128
    return params
    
def ner_h350_setting5_batch128_8():
    params = basic_params()
    params.hidden_size = 350
    params.batch_size = 128
    return params

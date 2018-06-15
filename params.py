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
        
        maxlen_s = 60,
        maxlen_q_train = 32,
        maxlen_q_dev = 27,
        maxlen_q_test = 27,
            
        rnn_dropout = 0.4,
        
        start_token = 1, # <GO> index
        end_token = 2, # <EOS> index
        
        num_heads = 1,
        context_depth = 512,
        attn_dropout = 0.4,

        attn = 'normed_bahdanau',

        # learning parameters
        batch_size = 64,
        learning_rate = 0.001,
        decay_step = None,
        decay_rate = 0.5,

        beam_width = 10,
        length_penalty_weight = 2.1
        )
    
def other_params():
    hparams = basic_params()
    hparams.voca_size = 45004
    hparams.embedding = '../qa_generation/data/processed/mpqg_substitute_a_vocab45_include_a/glove840b_mpqg_vocab300.npy'
    hparams.embedding_trainable = False
    hparams.hidden_size = 512
    hparams.encoder_layer = 1
    hparams.decoder_layer = 1

    hparams.num_heads = 1
    hparams.context_depth = 800
    
    hparams.batch_size = 64

    return hparams

def other_params1():
    hparams = basic_params()
    hparams.voca_size = 34004
    embedding = '../qa_generation/data/processed/full_mpqg_substitute_a_vocab_include_a/glove840b_mpqg_vocab300.npy'

    hparams.maxlen_q_train = 62
    hparams.maxlen_q_dev = 36
    hparams.maxlen_q_test = 38
    hparams.embedding_trainable = False
    hparams.hidden_size = 800
    hparams.encoder_layer = 1
    hparams.decoder_layer = 1

    hparams.batch_size = 64
    return hparams

import tensorflow as tf

def basic_params():
    '''A set of basic hyperparameters'''
    return tf.contrib.training.HParams(
        dtype = tf.float32,
        voca_size = 34004,
        embedding = '../qa_generation/data/squad/processed/qa_from_s/glove840b_qafroms_vocab300.npy',
        embedding_trainable = False,
        hidden_size = 512,
        encoder_layer = 1,
        decoder_layer = 1,
        
        maxlen_s = 60,
        maxlen_q_train = 32,
        maxlen_q_dev = 27,
            
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
        length_penalty_weight = 1.8
        )
    
def other_params():
    hparams = basic_params()
    hparams.embedding_trainable = False
    hparams.hidden_size = 800
    hparams.encoder_layer = 1
    hparams.decoder_layer = 1

    hparams.num_heads = 1
    hparams.context_depth = 800
    
    hparams.batch_size = 256

    return hparams

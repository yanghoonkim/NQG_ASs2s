import tensorflow as tf
def basic_params():
    '''A set of basic hyperparameters'''
    return tf.contrib.training.HParams(
        dtype = tf.float32,
        voca_size = 30004,
        embedding = None, #'data/squad/processed/qa_from_s/glove840b_squad_vocab300.npy',
        embedding_trainable = True,
        hidden_size = 600,
        encoder_layer = 1,
        decoder_layer = 1,
        
        maxlen_s = 61,
        maxlen_q_train = 31,
        maxlen_q_dev = 26,

        rnn_dropout = 0.3,
        regularization = 0, # if zero, no regularization

        start_token = 1, # <GO> index
        fake_end_token = 99999, # random number bigger than voca size

        # learning parameters
        batch_size = 128,
        loss_balance = 1,
        learning_rate = 0.001,
        decay_step = None,
        decay_rate = 0.5
        
    )

def second_params():
    '''A set of basic hyperparameters'''
    return tf.contrib.training.HParams(
        dtype = tf.float32,
        voca_size = 30004,
        embedding = 'data/squad/processed/xinyadu_processed/glove840b_xinyadu_vocab300.npy',
        embedding_trainable = False,
        hidden_size = 600,
        encoder_layer = 1,
        decoder_layer = 1,
        
        maxlen_s = 61,
        maxlen_q_train = 31,
        maxlen_q_dev = 26,

        rnn_dropout = 0.3,
        regularization = 0, # if zero, no regularization

        start_token = 1, # <GO> index
        fake_end_token = 99999, # random number bigger than voca size

        # learning parameters
        batch_size = 128,
        loss_balance = 1,
        learning_rate = 0.001,
        decay_step = 3000,
        decay_rate = 0.5
        )

def third_params():
    hparams = second_params()
    hparams.embedding_trainable = True
    return hparams

def params_4():
    hparams = second_params()
    hparams.decay_step = None
    hparams.hidden_size = 200
    return hparams

def params_5():
    hparams = second_params()
    hparams.decay_step = None
    hparams.hidden_size = 200
    hparams.encoder_layer = 2
    hparams.decoder_layer = 2
    return hparams

def params_6():
    hparams = second_params()
    hparams.decay_step = None
    hparams.hidden_size = 300
    hparams.encoder_layer = 1
    hparams.decoder_layer = 1
    hparams.decay_step = None
    return hparams

def params_7():
    hparams = second_params()
    hparams.decay_step = None
    hparams.hidden_size = 500
    hparams.encoder_layer = 1
    hparams.decoder_layer = 1
    hparams.decay_step = None
    hparams.batch_size = 256
    return hparams
  
def params_8():
    hparams = second_params()
    hparams.decay_step = None
    hparams.hidden_size = 500
    hparams.encoder_layer = 2
    hparams.decoder_layer = 2
    hparams.decay_step = None
    hparams.batch_size = 512
    return hparams
       
    
def other_params():
    hparams = basic_params()
    hparams.voca_size = 30004
    hparams.embedding = None
    hparams.embedding_trainable = True
    hparams.hidden_size = 300
    hparams.encoder_layer = 1
    haprams.decoder_layer = 1

    hparams.rnn_dropout = 0.3

    hparams.batch_size = 128

    hparams.add_hparam('lexicon_effect', 0.0) # lexicon coefficient
    hparams.add_hparam('decay', 0.4) # learning rate decay factor
    return hparams

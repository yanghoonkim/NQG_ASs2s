import tensorflow as tf

def basic_params():
    '''A set of basic hyperparameters'''
    return tf.contrib.training.HParams(
        dtype = tf.float32,
        voca_size = 34004,
        embedding = 'data/squad/processed/xinyadu_processed/glove840b_xinyadu_vocab300.npy',
        embedding_trainable = False,
        hidden_size = 600,
        encoder_layer = 2,
        decoder_layer = 2,
        
        maxlen_s = 61,
        maxlen_q_train = 31,
        maxlen_q_dev = 26,

        rnn_dropout = 0.3,
        regularization = 0, # if zero, no regularization

        start_token = 1, # <GO> index
        end_token = 2, # <EOS> index

        # learning parameters
        batch_size = 64,
        loss_balance = 1,
        learning_rate = 0.001,
        decay_step = None,
        decay_rate = 0.5
        )
    
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

    hparams.add_hparam('decay', 0.4) # learning rate decay factor
    return hparams

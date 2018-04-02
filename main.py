import argparse
import pickle as pkl
import numpy as np
import tensorflow as tf

import params
import model as model

FLAGS = None

def write_result(predict_results):
    print 'Load dic file...'
    with open('data/squad/processed/xinyadu_processed/vocab_xinyadu.dic') as dic:
        dic_file = pkl.load(dic)
    reversed_dic = dict((y,x) for x,y in dic_file.iteritems())
    
    print 'Writing into file...'
    with open(FLAGS.pred_dir, 'w') as f:
        while True:
            try : 
                output = predict_results.next()
                indices = [reversed_dic[index] for index in output['question']]
                sentence = ' '.join(indices) + '\n'
                f.write(sentence)

            except StopIteration:
                break

def print_result(predict_results):
    with open('data/squad/processed/xinyadu_processed/vocab_xinyadu.dic') as dic:
        dic_file = pkl.load(dic)
    reversed_dic = pkl.load(dic)

    while True:
        try:
            output = predict_results.next()
            indices = [reversed_dic[index] for index in output['question']]
            sentence = ' '.join(indices)
            print sentence
        except StopIteration:
            break
    
def main(unused):
    
    # Enable logging for tf.estimator
    tf.logging.set_verbosity(tf.logging.INFO)
    
    # Config
    config = tf.contrib.learn.RunConfig(
            model_dir = FLAGS.model_dir, 
            keep_checkpoint_max = 10, 
            save_checkpoints_steps = 100)
    
    # Load parameters
    model_params = getattr(params, FLAGS.params)().values()

    # Define estimator
    nn = tf.estimator.Estimator(model_fn=model.q_generation, config = config, params=model_params)
    
    # Load training data
    train_sentence = np.load(FLAGS.train_sentence) # train_data
    train_question = np.load(FLAGS.train_question) # train_label
    train_sentence_length = np.load(FLAGS.train_sentence_length)
    train_question_length = np.load(FLAGS.train_question_length)

    # Data shuffling for training data
    permutation = np.random.permutation(len(train_sentence))
    train_sentence = train_sentence[permutation]
    train_question = train_question[permutation]
    train_sentence_length = train_sentence_length[permutation]
    train_question_length = train_question_length[permutation]

    # Training input function for estimator
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"s": train_sentence, 'q': train_question,
            'len_s': train_sentence_length, 'len_q': train_question_length},
        y=train_sentence, # useless value
        batch_size = model_params['batch_size'],
        num_epochs=None,
        shuffle=True)
    
    # Load evaluation data
    eval_sentence = np.load(FLAGS.eval_sentence)
    eval_question = np.load(FLAGS.eval_question)
    eval_sentence_length = np.load(FLAGS.eval_sentence_length)
    eval_question_length = np.load(FLAGS.eval_question_length)


    # Evaluation input function for estimator
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x = {"s": eval_sentence, 'q': eval_question,
            'len_s': eval_sentence_length, 'len_q': eval_question_length},
        y = None,
        batch_size = model_params['batch_size'],
        num_epochs=1,
        shuffle=False)  
    
    # Calculate step size
    total_steps = 90000/model_params['batch_size'] * FLAGS.num_epochs

    # define experiment
    exp_nn = tf.contrib.learn.Experiment(
            estimator = nn, 
            train_input_fn = train_input_fn, 
            eval_input_fn = eval_input_fn,
            train_steps = total_steps,
            min_eval_frequency = 100)

    # train and evaluate
    if FLAGS.mode == 'train':
        exp_nn.train_and_evaluate()
    
    elif FLAGS.mode == 'eval':
        exp_nn.evaluate(delay_secs = 0)

    else: # 'pred'
        # load preprocessed prediction data
        # Still, pred data == eval data

        # prediction input function for estimator
        pred_input_fn = tf.estimator.inputs.numpy_input_fn(
                x = {"s" : eval_sentence, 'q': eval_question,
                    'len_s': eval_sentence_length, 'len_q': eval_question_length},
                y = None,
                batch_size = model_params['batch_size'],
                num_epochs = 1,
                shuffle = False)

        # prediction
        predict_results = nn.predict(input_fn = pred_input_fn)
        # write result(question) into file
        write_result(predict_results)
        #print_result(predict_results)

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type = str, help = 'train, eval')
    parser.add_argument('--train_sentence', type = str, default= '', help = 'path to the training sentence.')
    parser.add_argument('--train_question', type = str, default = '', help = 'path to the training question.')
    parser.add_argument('--train_sentence_length', type = str, default = '')
    parser.add_argument('--train_question_length', type = str, default = '')
    parser.add_argument('--eval_sentence', type = str, default = '', help = 'path to the evaluation sentence. ')
    parser.add_argument('--eval_question', type = str, default = '', help = 'path to the evaluation question.')
    parser.add_argument('--eval_sentence_length', type = str, default = '')
    parser.add_argument('--eval_question_length', type = str, default = '')
    parser.add_argument('--model_dir', type = str, help = 'path to save the model')
    parser.add_argument('--pred_dir', type = str, help = 'path to save the predictions')
    parser.add_argument('--params', type = str, help = 'parameter setting')
    parser.add_argument('--steps', type = int, default = 200000, help = 'training step size')
    parser.add_argument('--num_epochs', default = 10, help = 'training epoch size')
    FLAGS = parser.parse_args()

    tf.app.run(main)

#########################################################################
# File Name: run_attention.sh
# Author: ad26kt
# mail: ad26kt@gmail.com
# Created Time: Mon 09 Oct 2017 05:07:43 PM KST
#########################################################################
#!/bin/bash

# Definition of target dataset
squad(){
	TRAIN_SENTENCE='data/processed/mpqg_substitute_a_vocab_include_a/train_sentence.npy'
	TRAIN_QUESTION='data/processed/mpqg_substitute_a_vocab_include_a/train_question.npy'
	TRAIN_ANSWER='data/processed/mpqg_substitute_a_vocab_include_a/train_answer.npy'
	TRAIN_LENGTH_S='data/processed/mpqg_substitute_a_vocab_include_a/train_length_sentence.npy'
	TRAIN_LENGTH_Q='data/processed/mpqg_substitute_a_vocab_include_a/train_length_question.npy'
	TRAIN_LENGTH_A='data/processed/mpqg_substitute_a_vocab_include_a/train_length_answer.npy'
	DEV_SENTENCE='data/processed/mpqg_substitute_a_vocab_include_a/dev_sentence.npy'
	DEV_QUESTION='data/processed/mpqg_substitute_a_vocab_include_a/dev_question.npy'
	DEV_ANSWER='data/processed/mpqg_substitute_a_vocab_include_a/dev_answer.npy'
	DEV_LENGTH_S='data/processed/mpqg_substitute_a_vocab_include_a/dev_length_sentence.npy'
	DEV_LENGTH_Q='data/processed/mpqg_substitute_a_vocab_include_a/dev_length_question.npy'
	DEV_LENGTH_A='data/processed/mpqg_substitute_a_vocab_include_a/dev_length_answer.npy'
	TEST_SENTENCE='data/processed/mpqg_substitute_a_vocab_include_a/test_sentence.npy'
	TEST_ANSWER='data/processed/mpqg_substitute_a_vocab_include_a/test_answer.npy'
	TEST_LENGTH_S='data/processed/mpqg_substitute_a_vocab_include_a/test_length_sentence.npy'
	TEST_LENGTH_A='data/processed/mpqg_substitute_a_vocab_include_a/test_length_answer.npy'
	EMBEDDING='data/processed/mpqg_substitute_a_vocab_include_a/glove840b_vocab300.npy'
	DICTIONARY='data/processed/mpqg_substitute_a_vocab_include_a/vocab.dic'
	PARAMS=basic_params
	PRED_DIR='result/predictions.txt'
}


PARAMS=basic_params
MODEL_DIR=./store_model/$PARAMS
NUM_EPOCHS=10
python main.py --model_dir=$MODEL_DIR --params=$PARAMS --num_epochs=$NUM_EPOCHS

PARAMS=h200_batch64
MODEL_DIR=./store_model/$PARAMS
NUM_EPOCHS=10
python main.py --model_dir=$MODEL_DIR --params=$PARAMS --num_epochs=$NUM_EPOCHS

PARAMS=h512_batch128
MODEL_DIR=./store_model/$PARAMS
NUM_EPOCHS=10
python main.py --model_dir=$MODEL_DIR --params=$PARAMS --num_epochs=$NUM_EPOCHS

#########################################################################
# File Name: run_attention.sh
# Author: ad26kt
# mail: ad26kt@gmail.com
# Created Time: Mon 09 Oct 2017 05:07:43 PM KST
#########################################################################
#!/bin/bash
train(){
	MODE='train'
}

eval(){
	MODE='eval'
}

pred(){
	MODE='pred'
}


xinyadu_glove(){
	TRAIN_SENTENCE='data/squad/processed/xinyadu_processed/train_sentence.npy'
	TRAIN_QUESTION='data/squad/processed/xinyadu_processed/train_question.npy'
	TRAIN_LENGTH_S='data/squad/processed/xinyadu_processed/train_length_sentence.npy'
	TRAIN_LENGTH_Q='data/squad/processed/xinyadu_processed/train_length_question.npy'
	DEV_SENTENCE='data/squad/processed/xinyadu_processed/dev_sentence.npy'
	DEV_QUESTION='data/squad/processed/xinyadu_processed/dev_question.npy'
	DEV_LENGTH_S='data/squad/processed/xinyadu_processed/dev_length_sentence.npy'
	DEV_LENGTH_Q='data/squad/processed/xinyadu_processed/dev_length_question.npy'
	TEST_SENTENCE='data/squad/processed/xinyadu_processed/test_sentence.npy'
	TEST_LENGTH_S='data/squad/processed/xinyadu_processed/test_length_sentence.npy'
	PRED_DIR='result/xinyadu_glove.txt'
	PARAMS=basic_params
}

# Pass the first argument as the name of dataset
# Pass the second argument as mode
# Pass the third argument to control GPU usage
$1
$2

TRAIN_STEPS=200000
NUM_EPOCHS=15
MODEL_DIR=./store_model/$3

python main.py \
	--mode=$MODE \
	--train_sentence=$TRAIN_SENTENCE \
	--train_question=$TRAIN_QUESTION \
	--train_sentence_length=$TRAIN_LENGTH_S \
	--train_question_length=$TRAIN_LENGTH_Q \
	--eval_sentence=$DEV_SENTENCE \
	--eval_question=$DEV_QUESTION \
	--eval_sentence_length=$DEV_LENGTH_S \
	--eval_question_length=$DEV_LENGTH_Q \
	--test_sentence=$TEST_SENTENCE \
	--test_sentence_length=$TEST_LENGTH_S \
	--model_dir=$MODEL_DIR \
	--pred_dir=$PRED_DIR \
	--params=$PARAMS \
	--num_epochs=$NUM_EPOCHS

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


squad(){
	TRAIN_SENTENCE='../qa_generation/data/processed/mpqg_substitute_a_vocab_include_a/train_sentence.npy'
	TRAIN_QUESTION='../qa_generation/data/processed/mpqg_substitute_a_vocab_include_a/train_question.npy'
	TRAIN_ANSWER='../qa_generation/data/processed/mpqg_substitute_a_vocab_include_a/train_answer.npy'
	TRAIN_LENGTH_S='../qa_generation/data/processed/mpqg_substitute_a_vocab_include_a/train_length_sentence.npy'
	TRAIN_LENGTH_Q='../qa_generation/data/processed/mpqg_substitute_a_vocab_include_a/train_length_question.npy'
	TRAIN_LENGTH_A='../qa_generation/data/processed/mpqg_substitute_a_vocab_include_a/train_length_answer.npy'
	DEV_SENTENCE='../qa_generation/data/processed/mpqg_substitute_a_vocab_include_a/dev_sentence.npy'
	DEV_QUESTION='../qa_generation/data/processed/mpqg_substitute_a_vocab_include_a/dev_question.npy'
	DEV_ANSWER='../qa_generation/data/processed/mpqg_substitute_a_vocab_include_a/dev_answer.npy'
	DEV_LENGTH_S='../qa_generation/data/processed/mpqg_substitute_a_vocab_include_a/dev_length_sentence.npy'
	DEV_LENGTH_Q='../qa_generation/data/processed/mpqg_substitute_a_vocab_include_a/dev_length_question.npy'
	DEV_LENGTH_A='../qa_generation/data/processed/mpqg_substitute_a_vocab_include_a/dev_length_answer.npy'
	TEST_SENTENCE='../qa_generation/data/processed/mpqg_substitute_a_vocab_include_a/test_sentence.npy'
	TEST_ANSWER='../qa_generation/data/processed/mpqg_substitute_a_vocab_include_a/test_answer.npy'
	TEST_LENGTH_S='../qa_generation/data/processed/mpqg_substitute_a_vocab_include_a/test_length_sentence.npy'
	TEST_LENGTH_A='../qa_generation/data/processed/mpqg_substitute_a_vocab_include_a/test_length_answer.npy'
	DICTIONARY='../qa_generation/data/processed/mpqg_substitute_a_vocab_include_a/vocab_mpqg.dic'
	PRED_DIR='result/squad.txt'
	PARAMS=basic_params
}

squadd(){
	TRAIN_SENTENCE='../qa_generation/data/processed/mpqg_substitute_a_vocab45_include_a/train_sentence.npy'
	TRAIN_QUESTION='../qa_generation/data/processed/mpqg_substitute_a_vocab45_include_a/train_question.npy'
	TRAIN_ANSWER='../qa_generation/data/processed/mpqg_substitute_a_vocab45_include_a/train_answer.npy'
	TRAIN_LENGTH_S='../qa_generation/data/processed/mpqg_substitute_a_vocab45_include_a/train_length_sentence.npy'
	TRAIN_LENGTH_Q='../qa_generation/data/processed/mpqg_substitute_a_vocab45_include_a/train_length_question.npy'
	TRAIN_LENGTH_A='../qa_generation/data/processed/mpqg_substitute_a_vocab45_include_a/train_length_answer.npy'
	DEV_SENTENCE='../qa_generation/data/processed/mpqg_substitute_a_vocab45_include_a/dev_sentence.npy'
	DEV_QUESTION='../qa_generation/data/processed/mpqg_substitute_a_vocab45_include_a/dev_question.npy'
	DEV_ANSWER='../qa_generation/data/processed/mpqg_substitute_a_vocab45_include_a/dev_answer.npy'
	DEV_LENGTH_S='../qa_generation/data/processed/mpqg_substitute_a_vocab45_include_a/dev_length_sentence.npy'
	DEV_LENGTH_Q='../qa_generation/data/processed/mpqg_substitute_a_vocab45_include_a/dev_length_question.npy'
	DEV_LENGTH_A='../qa_generation/data/processed/mpqg_substitute_a_vocab45_include_a/dev_length_answer.npy'
	TEST_SENTENCE='../qa_generation/data/processed/mpqg_substitute_a_vocab45_include_a/test_sentence.npy'
	TEST_ANSWER='../qa_generation/data/processed/mpqg_substitute_a_vocab45_include_a/test_answer.npy'
	TEST_LENGTH_S='../qa_generation/data/processed/mpqg_substitute_a_vocab45_include_a/test_length_sentence.npy'
	TEST_LENGTH_A='../qa_generation/data/processed/mpqg_substitute_a_vocab45_include_a/test_length_answer.npy'
	DICTIONARY='../qa_generation/data/processed/mpqg_substitute_a_vocab45_include_a/vocab_mpqg.dic'
	PRED_DIR='result/squadd.txt'
	PARAMS=other_params
}


squaddd(){
	TRAIN_SENTENCE='../qa_generation/data/processed/full_mpqg_substitute_a_vocab_include_a/train_sentence.npy'
	TRAIN_QUESTION='../qa_generation/data/processed/full_mpqg_substitute_a_vocab_include_a/train_question.npy'
	TRAIN_ANSWER='../qa_generation/data/processed/full_mpqg_substitute_a_vocab_include_a/train_answer.npy'
	TRAIN_LENGTH_S='../qa_generation/data/processed/full_mpqg_substitute_a_vocab_include_a/train_length_sentence.npy'
	TRAIN_LENGTH_Q='../qa_generation/data/processed/full_mpqg_substitute_a_vocab_include_a/train_length_question.npy'
	TRAIN_LENGTH_A='../qa_generation/data/processed/full_mpqg_substitute_a_vocab_include_a/train_length_answer.npy'
	DEV_SENTENCE='../qa_generation/data/processed/full_mpqg_substitute_a_vocab_include_a/dev_sentence.npy'
	DEV_QUESTION='../qa_generation/data/processed/full_mpqg_substitute_a_vocab_include_a/dev_question.npy'
	DEV_ANSWER='../qa_generation/data/processed/full_mpqg_substitute_a_vocab_include_a/dev_answer.npy'
	DEV_LENGTH_S='../qa_generation/data/processed/full_mpqg_substitute_a_vocab_include_a/dev_length_sentence.npy'
	DEV_LENGTH_Q='../qa_generation/data/processed/full_mpqg_substitute_a_vocab_include_a/dev_length_question.npy'
	DEV_LENGTH_A='../qa_generation/data/processed/full_mpqg_substitute_a_vocab_include_a/dev_length_answer.npy'
	TEST_SENTENCE='../qa_generation/data/processed/full_mpqg_substitute_a_vocab_include_a/test_sentence.npy'
	TEST_ANSWER='../qa_generation/data/processed/full_mpqg_substitute_a_vocab_include_a/test_answer.npy'
	TEST_LENGTH_S='../qa_generation/data/processed/full_mpqg_substitute_a_vocab_include_a/test_length_sentence.npy'
	TEST_LENGTH_A='../qa_generation/data/processed/full_mpqg_substitute_a_vocab_include_a/test_length_answer.npy'
	DICTIONARY='../qa_generation/data/processed/full_mpqg_substitute_a_vocab_include_a/vocab_mpqg.dic'
	PRED_DIR='result/squaddd.txt'
	PARAMS=other_params1
}

squad1(){
	TRAIN_SENTENCE='../qa_generation/data/processed/full_mpqg_substitute_a_vocab_include_a/train_sentence.npy'
	TRAIN_QUESTION='../qa_generation/data/processed/full_mpqg_substitute_a_vocab_include_a/train_question.npy'
	TRAIN_ANSWER='../qa_generation/data/processed/full_mpqg_substitute_a_vocab_include_a/train_answer.npy'
	TRAIN_LENGTH_S='../qa_generation/data/processed/full_mpqg_substitute_a_vocab_include_a/train_length_sentence.npy'
	TRAIN_LENGTH_Q='../qa_generation/data/processed/full_mpqg_substitute_a_vocab_include_a/train_length_question.npy'
	TRAIN_LENGTH_A='../qa_generation/data/processed/full_mpqg_substitute_a_vocab_include_a/train_length_answer.npy'
	DEV_SENTENCE='../qa_generation/data/processed/full_mpqg_substitute_a_vocab_include_a/dev_sentence.npy'
	DEV_QUESTION='../qa_generation/data/processed/full_mpqg_substitute_a_vocab_include_a/dev_question.npy'
	DEV_ANSWER='../qa_generation/data/processed/full_mpqg_substitute_a_vocab_include_a/dev_answer.npy'
	DEV_LENGTH_S='../qa_generation/data/processed/full_mpqg_substitute_a_vocab_include_a/dev_length_sentence.npy'
	DEV_LENGTH_Q='../qa_generation/data/processed/full_mpqg_substitute_a_vocab_include_a/dev_length_question.npy'
	DEV_LENGTH_A='../qa_generation/data/processed/full_mpqg_substitute_a_vocab_include_a/dev_length_answer.npy'
	TEST_SENTENCE='../qa_generation/data/processed/full_mpqg_substitute_a_vocab_include_a/test_sentence.npy'
	TEST_ANSWER='../qa_generation/data/processed/full_mpqg_substitute_a_vocab_include_a/test_answer.npy'
	TEST_LENGTH_S='../qa_generation/data/processed/full_mpqg_substitute_a_vocab_include_a/test_length_sentence.npy'
	TEST_LENGTH_A='../qa_generation/data/processed/full_mpqg_substitute_a_vocab_include_a/test_length_answer.npy'
	DICTIONARY='../qa_generation/data/processed/full_mpqg_substitute_a_vocab_include_a/vocab_mpqg.dic'
	PRED_DIR='result/squad1.txt'
	PARAMS=other_params1
}


# Pass the first argument as the name of dataset
# Pass the second argument as mode
# Pass the third argument to name the weight set
# Pass the fourth arqugment to adjust training epoch
$1
$2

TRAIN_STEPS=200000
MODEL_DIR=./store_model/$3
NUM_EPOCHS=$4

python main.py \
	--mode=$MODE \
	--train_sentence=$TRAIN_SENTENCE \
	--train_question=$TRAIN_QUESTION \
	--train_answer=$TRAIN_ANSWER \
	--train_sentence_length=$TRAIN_LENGTH_S \
	--train_question_length=$TRAIN_LENGTH_Q \
	--train_answer_length=$TRAIN_LENGTH_A \
	--eval_sentence=$DEV_SENTENCE \
	--eval_question=$DEV_QUESTION \
	--eval_answer=$DEV_ANSWER \
	--eval_sentence_length=$DEV_LENGTH_S \
	--eval_question_length=$DEV_LENGTH_Q \
	--eval_answer_length=$DEV_LENGTH_A \
	--test_sentence=$TEST_SENTENCE \
	--test_answer=$TEST_ANSWER \
	--test_sentence_length=$TEST_LENGTH_S \
	--test_answer_length=$TEST_LENGTH_A \
	--dictionary=$DICTIONARY \
	--model_dir=$MODEL_DIR \
	--pred_dir=$PRED_DIR \
	--params=$PARAMS \
	--num_epochs=$NUM_EPOCHS

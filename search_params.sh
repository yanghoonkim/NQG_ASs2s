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

TRAIN_STEPS=200000
NUM_EPOCHS=8

<<'THISTIME'
PARAMS=h250_divert_s1_batch256
MODEL_DIR=./store_model/$PARAMS
python main.py --model_dir=$MODEL_DIR --params=$PARAMS

PARAMS=h250_divert_s2_batch256
MODEL_DIR=./store_model/$PARAMS
python main.py --model_dir=$MODEL_DIR --params=$PARAMS

PARAMS=h250_divert_s8_batch256
MODEL_DIR=./store_model/$PARAMS
python main.py --model_dir=$MODEL_DIR --params=$PARAMS

PARAMS=h250_divert_s20_batch256
MODEL_DIR=./store_model/$PARAMS
python main.py --model_dir=$MODEL_DIR --params=$PARAMS

PARAMS=h250_divert_s50_batch256
MODEL_DIR=./store_model/$PARAMS
python main.py --model_dir=$MODEL_DIR --params=$PARAMS

PARAMS=h250_divert_s100_batch256
MODEL_DIR=./store_model/$PARAMS
python main.py --model_dir=$MODEL_DIR --params=$PARAMS

PARAMS=h250_divert_s256_batch256
MODEL_DIR=./store_model/$PARAMS
python main.py --model_dir=$MODEL_DIR --params=$PARAMS

#

PARAMS=h350_divert_s1_batch256
MODEL_DIR=./store_model/$PARAMS
python main.py --model_dir=$MODEL_DIR --params=$PARAMS

PARAMS=h350_divert_s2_batch256
MODEL_DIR=./store_model/$PARAMS
python main.py --model_dir=$MODEL_DIR --params=$PARAMS

PARAMS=h350_divert_s8_batch256
MODEL_DIR=./store_model/$PARAMS
python main.py --model_dir=$MODEL_DIR --params=$PARAMS

PARAMS=h350_divert_s20_batch256
MODEL_DIR=./store_model/$PARAMS
python main.py --model_dir=$MODEL_DIR --params=$PARAMS

PARAMS=h350_divert_s50_batch256
MODEL_DIR=./store_model/$PARAMS
python main.py --model_dir=$MODEL_DIR --params=$PARAMS

PARAMS=h350_divert_s100_batch256
MODEL_DIR=./store_model/$PARAMS
python main.py --model_dir=$MODEL_DIR --params=$PARAMS

PARAMS=h350_divert_s256_batch256
MODEL_DIR=./store_model/$PARAMS
python main.py --model_dir=$MODEL_DIR --params=$PARAMS

##

PARAMS=h512_divert_s1_batch256
MODEL_DIR=./store_model/$PARAMS
python main.py --model_dir=$MODEL_DIR --params=$PARAMS

PARAMS=h512_divert_s2_batch256
MODEL_DIR=./store_model/$PARAMS
python main.py --model_dir=$MODEL_DIR --params=$PARAMS

PARAMS=h512_divert_s8_batch256
MODEL_DIR=./store_model/$PARAMS
python main.py --model_dir=$MODEL_DIR --params=$PARAMS

PARAMS=h512_divert_s20_batch256
MODEL_DIR=./store_model/$PARAMS
python main.py --model_dir=$MODEL_DIR --params=$PARAMS

PARAMS=h512_divert_s50_batch256
MODEL_DIR=./store_model/$PARAMS
python main.py --model_dir=$MODEL_DIR --params=$PARAMS

PARAMS=h512_divert_s100_batch256
MODEL_DIR=./store_model/$PARAMS
python main.py --model_dir=$MODEL_DIR --params=$PARAMS
THISTIME

PARAMS=h512_divert_s256_batch256
MODEL_DIR=./store_model/$PARAMS
python main.py --model_dir=$MODEL_DIR --params=$PARAMS


<<'COMMENT'
PARAMS=h350_memory_setting5_batch256
MODEL_DIR=./store_model/$PARAMS
python main.py --model_dir=$MODEL_DIR --params=$PARAMS

PARAMS=h512_memory_setting5_batch256
MODEL_DIR=./store_model/$PARAMS
python main.py --model_dir=$MODEL_DIR --params=$PARAMS

PARAMS=h250_memory_setting5_batch128
MODEL_DIR=./store_model/$PARAMS
python main.py --model_dir=$MODEL_DIR --params=$PARAMS

PARAMS=h250_memory_setting5_batch256
MODEL_DIR=./store_model/$PARAMS
python main.py --model_dir=$MODEL_DIR --params=$PARAMS

PARAMS=h512_memory_setting5_batch128
MODEL_DIR=./store_model/h512_memory_setting5_batch128_
python main.py --model_dir=$MODEL_DIR --params=$PARAMS


PARAMS=h300_memory_setting5_batch64
MODEL_DIR=./store_model/$PARAMS
python main.py --model_dir=$MODEL_DIR --params=$PARAMS 

PARAMS=h300_memory_setting5_batch32
MODEL_DIR=./store_model/$PARAMS
python main.py --mode='pred ' --model_dir=$MODEL_DIR --params=$PARAMS

PARAMS=h512_memory_setting5_batch64
MODEL_DIR=./store_model/$PARAMS
python main.py --model_dir=$MODEL_DIR --params=$PARAMS

PARAMS=h512_memory_setting5_batch128
MODEL_DIR=./store_model/$PARAMS
python main.py --mode='pred' --model_dir=$MODEL_DIR --params=$PARAMS


PARAMS=h1024_memory_setting5_batch64
MODEL_DIR=./store_model/$PARAMS
python main.py --model_dir=$MODEL_DIR --params=$PARAMS


###
PARAMS=h200_memory_setting5_batch64_2layer
MODEL_DIR=./store_model/$PARAMS
python main.py --model_dir=$MODEL_DIR --params=$PARAMS 

PARAMS=h512_memory_setting5_batch128
MODEL_DIR=./store_model/h512_memory_setting5_batch128
python main.py --model_dir=$MODEL_DIR --params=$PARAMS

PARAMS=h350_memory_setting5_batch128
MODEL_DIR=./store_model/$PARAMS
python main.py --model_dir=$MODEL_DIR --params=$PARAMS

PARAMS=h800_memory_setting5_batch128
MODEL_DIR=./store_model/$PARAMS
python main.py --model_dir=$MODEL_DIR --params=$PARAMS


PARAMS=h350_memory_setting5_batch64_2layer
MODEL_DIR=./store_model/$PARAMS
python main.py --model_dir=$MODEL_DIR --params=$PARAMS


PARAMS=h512_memory_setting5_batch64_2layer
MODEL_DIR=./store_model/$PARAMS
python main.py --model_dir=$MODEL_DIR --params=$PARAMS

PARAMS=h512_memory_setting5_batch64_212layer
MODEL_DIR=./store_model/$PARAMS
python main.py --model_dir=$MODEL_DIR --params=$PARAMS


PARAMS=h512_memory_setting4_batch64_112layer
MODEL_DIR=./store_model/$PARAMS
python main.py --model_dir=$MODEL_DIR --params=$PARAMS

PARAMS=h512_memory_setting5_batch64_221layer
MODEL_DIR=./store_model/$PARAMS
python main.py --model_dir=$MODEL_DIR --params=$PARAMS

PARAMS=h512_memory_setting5_batch64_121layer
MODEL_DIR=./store_model/$PARAMS
python main.py --model_dir=$MODEL_DIR --params=$PARAMS

PARAMS=h800_memory_a_60
MODEL_DIR=./store_model/$PARAMS
python main.py --model_dir=$MODEL_DIR --params=$PARAMS

PARAMS=h800_memory_s_1
MODEL_DIR=./store_model/$PARAMS
python main.py --model_dir=$MODEL_DIR --params=$PARAMS

PARAMS=h800_memory_s_5
MODEL_DIR=./store_model/$PARAMS
python main.py --model_dir=$MODEL_DIR --params=$PARAMS

PARAMS=h800_memory_s_20
MODEL_DIR=./store_model/$PARAMS
python main.py --model_dir=$MODEL_DIR --params=$PARAMS
COMMENT















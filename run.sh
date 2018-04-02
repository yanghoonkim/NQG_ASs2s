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
	TRAIN_SENTENCE='data/squad/processed/qa_from_s/train_sentence.npy'
	TRAIN_QUESTION='data/squad/processed/qa_from_s/train_question.npy'
	TRAIN_LENGTH_S='data/squad/processed/qa_from_s/train_length_sentence.npy'
	TRAIN_LENGTH_Q='data/squad/processed/qa_from_s/train_length_question.npy'
	DEV_SENTENCE='data/squad/processed/qa_from_s/dev_sentence.npy'
	DEV_QUESTION='data/squad/processed/qa_from_s/dev_question.npy'
	DEV_LENGTH_S='data/squad/processed/qa_from_s/dev_length_sentence.npy'
	DEV_LENGTH_Q='data/squad/processed/qa_from_s/dev_length_question.npy'
	PRED_DIR='result/question.txt'
	PARAMS=basic_params
}

xinyadu(){
	TRAIN_SENTENCE='data/squad/processed/xinyadu_processed/train_sentence.npy'
	TRAIN_QUESTION='data/squad/processed/xinyadu_processed/train_question.npy'
	TRAIN_LENGTH_S='data/squad/processed/xinyadu_processed/train_length_sentence.npy'
	TRAIN_LENGTH_Q='data/squad/processed/xinyadu_processed/train_length_question.npy'
	DEV_SENTENCE='data/squad/processed/xinyadu_processed/dev_sentence.npy'
	DEV_QUESTION='data/squad/processed/xinyadu_processed/dev_question.npy'
	DEV_LENGTH_S='data/squad/processed/xinyadu_processed/dev_length_sentence.npy'
	DEV_LENGTH_Q='data/squad/processed/xinyadu_processed/dev_length_question.npy'
	PRED_DIR='result/question.txt'
	PARAMS=basic_params
}

xinyadu_separate(){
	TRAIN_SENTENCE='data/squad/processed/xinyadu_separate_processed/train_sentence.npy'
	TRAIN_QUESTION='data/squad/processed/xinyadu_separate_processed/train_question.npy'
	TRAIN_LENGTH_S='data/squad/processed/xinyadu_separate_processed/train_length_sentence.npy'
	TRAIN_LENGTH_Q='data/squad/processed/xinyadu_separate_processed/train_length_question.npy'
	DEV_SENTENCE='data/squad/processed/xinyadu_separate_processed/dev_sentence.npy'
	DEV_QUESTION='data/squad/processed/xinyadu_separate_processed/dev_question.npy'
	DEV_LENGTH_S='data/squad/processed/xinyadu_separate_processed/dev_length_sentence.npy'
	DEV_LENGTH_Q='data/squad/processed/xinyadu_separate_processed/dev_length_question.npy'
	PRED_DIR='result/question.txt'
	PARAMS=basic_params
}

# for xinyadu_glove_300/600
xinyadu_glove(){ 
	TRAIN_SENTENCE='data/squad/processed/xinyadu_processed/train_sentence.npy'
	TRAIN_QUESTION='data/squad/processed/xinyadu_processed/train_question.npy'
	TRAIN_LENGTH_S='data/squad/processed/xinyadu_processed/train_length_sentence.npy'
	TRAIN_LENGTH_Q='data/squad/processed/xinyadu_processed/train_length_question.npy'
	DEV_SENTENCE='data/squad/processed/xinyadu_processed/dev_sentence.npy'
	DEV_QUESTION='data/squad/processed/xinyadu_processed/dev_question.npy'
	DEV_LENGTH_S='data/squad/processed/xinyadu_processed/dev_length_sentence.npy'
	DEV_LENGTH_Q='data/squad/processed/xinyadu_processed/dev_length_question.npy'
	PRED_DIR='result/question.txt'
	PARAMS=second_params
}
# for xinyadu_glove_trainable_300/600
xinyadu_glove_trainable(){
	TRAIN_SENTENCE='data/squad/processed/xinyadu_processed/train_sentence.npy'
	TRAIN_QUESTION='data/squad/processed/xinyadu_processed/train_question.npy'
	TRAIN_LENGTH_S='data/squad/processed/xinyadu_processed/train_length_sentence.npy'
	TRAIN_LENGTH_Q='data/squad/processed/xinyadu_processed/train_length_question.npy'
	DEV_SENTENCE='data/squad/processed/xinyadu_processed/dev_sentence.npy'
	DEV_QUESTION='data/squad/processed/xinyadu_processed/dev_question.npy'
	DEV_LENGTH_S='data/squad/processed/xinyadu_processed/dev_length_sentence.npy'
	DEV_LENGTH_Q='data/squad/processed/xinyadu_processed/dev_length_question.npy'
	PRED_DIR='result/question.txt'
	PARAMS=third_params
}

xinyadu_glove_200(){ 
	TRAIN_SENTENCE='data/squad/processed/xinyadu_processed/train_sentence.npy'
	TRAIN_QUESTION='data/squad/processed/xinyadu_processed/train_question.npy'
	TRAIN_LENGTH_S='data/squad/processed/xinyadu_processed/train_length_sentence.npy'
	TRAIN_LENGTH_Q='data/squad/processed/xinyadu_processed/train_length_question.npy'
	DEV_SENTENCE='data/squad/processed/xinyadu_processed/dev_sentence.npy'
	DEV_QUESTION='data/squad/processed/xinyadu_processed/dev_question.npy'
	DEV_LENGTH_S='data/squad/processed/xinyadu_processed/dev_length_sentence.npy'
	DEV_LENGTH_Q='data/squad/processed/xinyadu_processed/dev_length_question.npy'
	PRED_DIR='result/question.txt'
	PARAMS=params_4
}

xinyadu_glove_200_2layer(){ 
	TRAIN_SENTENCE='data/squad/processed/xinyadu_processed/train_sentence.npy'
	TRAIN_QUESTION='data/squad/processed/xinyadu_processed/train_question.npy'
	TRAIN_LENGTH_S='data/squad/processed/xinyadu_processed/train_length_sentence.npy'
	TRAIN_LENGTH_Q='data/squad/processed/xinyadu_processed/train_length_question.npy'
	DEV_SENTENCE='data/squad/processed/xinyadu_processed/dev_sentence.npy'
	DEV_QUESTION='data/squad/processed/xinyadu_processed/dev_question.npy'
	DEV_LENGTH_S='data/squad/processed/xinyadu_processed/dev_length_sentence.npy'
	DEV_LENGTH_Q='data/squad/processed/xinyadu_processed/dev_length_question.npy'
	PRED_DIR='result/question.txt'
	PARAMS=params_5
}
xinyadu_glove_300_bi(){ 
	TRAIN_SENTENCE='data/squad/processed/xinyadu_processed/train_sentence.npy'
	TRAIN_QUESTION='data/squad/processed/xinyadu_processed/train_question.npy'
	TRAIN_LENGTH_S='data/squad/processed/xinyadu_processed/train_length_sentence.npy'
	TRAIN_LENGTH_Q='data/squad/processed/xinyadu_processed/train_length_question.npy'
	DEV_SENTENCE='data/squad/processed/xinyadu_processed/dev_sentence.npy'
	DEV_QUESTION='data/squad/processed/xinyadu_processed/dev_question.npy'
	DEV_LENGTH_S='data/squad/processed/xinyadu_processed/dev_length_sentence.npy'
	DEV_LENGTH_Q='data/squad/processed/xinyadu_processed/dev_length_question.npy'
	PRED_DIR='result/xinyadu_glove_300_bi.txt'
	PARAMS=params_6
}

xinyadu_glove_500_bi(){ 
	TRAIN_SENTENCE='data/squad/processed/xinyadu_processed/train_sentence.npy'
	TRAIN_QUESTION='data/squad/processed/xinyadu_processed/train_question.npy'
	TRAIN_LENGTH_S='data/squad/processed/xinyadu_processed/train_length_sentence.npy'
	TRAIN_LENGTH_Q='data/squad/processed/xinyadu_processed/train_length_question.npy'
	DEV_SENTENCE='data/squad/processed/xinyadu_processed/dev_sentence.npy'
	DEV_QUESTION='data/squad/processed/xinyadu_processed/dev_question.npy'
	DEV_LENGTH_S='data/squad/processed/xinyadu_processed/dev_length_sentence.npy'
	DEV_LENGTH_Q='data/squad/processed/xinyadu_processed/dev_length_question.npy'
	PRED_DIR='result/xinyadu_glove_500_bi.txt'
	PARAMS=params_7
}

xinyadu_glove_500_2layer_bi(){ 
	TRAIN_SENTENCE='data/squad/processed/xinyadu_processed/train_sentence.npy'
	TRAIN_QUESTION='data/squad/processed/xinyadu_processed/train_question.npy'
	TRAIN_LENGTH_S='data/squad/processed/xinyadu_processed/train_length_sentence.npy'
	TRAIN_LENGTH_Q='data/squad/processed/xinyadu_processed/train_length_question.npy'
	DEV_SENTENCE='data/squad/processed/xinyadu_processed/dev_sentence.npy'
	DEV_QUESTION='data/squad/processed/xinyadu_processed/dev_question.npy'
	DEV_LENGTH_S='data/squad/processed/xinyadu_processed/dev_length_sentence.npy'
	DEV_LENGTH_Q='data/squad/processed/xinyadu_processed/dev_length_question.npy'
	PRED_DIR='result/xinyadu_glove_500_2layer_bi.txt'
	PARAMS=params_8
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
	--model_dir=$MODEL_DIR \
	--pred_dir=$PRED_DIR \
	--params=$PARAMS \
	--steps=$TRAIN_STEPS\
	--num_epochs=$NUM_EPOCHS

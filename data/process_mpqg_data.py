import json
import os
import io
import pickle as pkl
import numpy as np
import string
from collections import defaultdict

print '########## Data processing start ##########\n'

# SETTINGS >>>



# Use all data or cut by length
cut_by_length = True

# Maximum length for train/dev
cut_s_train = 60
cut_q_train = 30

cut_s_dev = 60
cut_q_dev = 25

# Original data path
TRAIN_FILE = 'mpqg_data/train_sent_pre.json'
DEV_FILE = 'mpqg_data/dev_sent_pre.json'
TEST_FILE=  'mpqg_data/test_sent_pre.json'

output_dir = 'processed/mpqg_substitute_a_vocab_include_a/'            
text_dir = os.path.join(output_dir, 'filtered_txt/')


# Training txt saving path
sentence_txt_train_origin = 'train_sentence_origin.txt'
sentence_txt_train = 'train_sentence.txt'
question_txt_train = 'train_question.txt'
answer_txt_train = 'train_answer.txt'

# Dev txt saving path
sentence_txt_dev_origin = 'dev_sentence_origin.txt'
sentence_txt_dev = 'dev_sentence.txt'
question_txt_dev = 'dev_question.txt'
answer_txt_dev = 'dev_answer.txt'

# Test txt saving path
sentence_txt_test_origin = 'test_sentence_origin.txt'
sentence_txt_test = 'test_sentence.txt'
question_txt_test = 'test_question.txt'
answer_txt_test = 'test_answer.txt'


# Vocabulary related 
dic_size = 34000
vocab_dir = os.path.join(output_dir, 'vocab.dic')
vocab_include_answer = True


# Processed training data saving path
sentence_outfile_train = os.path.join(output_dir, 'train_sentence.npy')
question_outfile_train = os.path.join(output_dir, 'train_question.npy')
answer_outfile_train = os.path.join(output_dir, 'train_answer.npy')
length_sentence_outfile_train = os.path.join(output_dir, 'train_length_sentence.npy')
length_question_outfile_train = os.path.join(output_dir, 'train_length_question.npy')
length_answer_outfile_train = os.path.join(output_dir, 'train_length_answer.npy')

# Processed dev data saving path
sentence_outfile_dev = os.path.join(output_dir, 'dev_sentence.npy')
question_outfile_dev = os.path.join(output_dir, 'dev_question.npy')
answer_outfile_dev = os.path.join(output_dir, 'dev_answer.npy')
length_sentence_outfile_dev = os.path.join(output_dir, 'dev_length_sentence.npy')
length_question_outfile_dev = os.path.join(output_dir, 'dev_length_question.npy')
length_answer_outfile_dev = os.path.join(output_dir, 'dev_length_answer.npy')

# Processed test data saving path
sentence_outfile_test = os.path.join(output_dir, 'test_sentence.npy')
question_outfile_test = os.path.join(output_dir, 'test_question.npy')
answer_outfile_test = os.path.join(output_dir, 'test_answer.npy')
length_sentence_outfile_test = os.path.join(output_dir, 'test_length_sentence.npy')
length_question_outfile_test = os.path.join(output_dir, 'test_length_question.npy')
length_answer_outfile_test = os.path.join(output_dir, 'test_length_answer.npy')



# SETTINGS <<<



# DATA EXTRACTION FROM JSON FILE >>>



def data_extract(data_path):
    with open(data_path) as data:
        f = json.load(data)
    
    tok_sentence = list()
    tok_question = list()
    tok_answer = list()
    

    # f is a list
    for line in f:
        sentence = line['annotation1']['toks']
        question = line['annotation2']['toks']
        answer = line['annotation3']['toks']
        if len(sentence) != 0 and len(question) != 0 and len(answer) != 0:   
            tok_sentence.append(sentence.split())
            tok_question.append(question.split())
            tok_answer.append(answer.split())
    
    return tok_sentence, tok_question, tok_answer


# Training data tokenization
print '>>> Extracting data from json file...',
train_tok = data_extract(TRAIN_FILE)

sentence_train = [[word.lower() for word in line] for line in train_tok[0]]
question_train = [[word.lower() for word in line] for line in train_tok[1]]
answer_train = [[word.lower() for word in line] for line in train_tok[2]]
print 'Complete\n'

maxlen_s_train = max([len(sentence) for sentence in sentence_train])
print '---------- Data Statistics ----------'
print 'Training sentences max-length : %d'%maxlen_s_train

maxlen_q_train = max([len(sentence) for sentence in question_train])
print 'Training questions max-length : %d'%maxlen_q_train


# Dev data tokenization
dev_tok = data_extract(DEV_FILE)

sentence_dev = [[word.lower() for word in line] for line in dev_tok[0]]
question_dev = [[word.lower() for word in line] for line in dev_tok[1]]
answer_dev = [[word.lower() for word in line] for line in dev_tok[2]]

maxlen_s_dev = max([len(sentence) for sentence in sentence_dev])
print 'Dev sentences max-length : %d'%maxlen_s_dev

maxlen_q_dev = max([len(sentence) for sentence in question_dev])
print 'Dev questions max-length : %d'%maxlen_q_dev


# Test data tokenization
test_tok = data_extract(TEST_FILE)

sentence_test = [[word.lower() for word in line] for line in test_tok[0]]
question_test = [[word.lower() for word in line] for line in test_tok[1]]
answer_test = [[word.lower() for word in line] for line in test_tok[2]]

maxlen_s_test = max([len(sentence) for sentence in sentence_test])
print 'Test sentences max-length : %d'%maxlen_s_test

maxlen_q_test = max([len(sentence) for sentence in question_test])
print 'Test questions max-length : %d\n'%maxlen_q_test



# DATA EXTRACTION FROM JSON FILE <<<



# FILTERING WITH MAXLEN >>>



if cut_by_length:
    maxlen_s_train = cut_s_train
    maxlen_q_train = cut_q_train
    maxlen_s_dev = cut_s_dev
    maxlen_q_dev = cut_q_dev

print '---------- Length Cut ----------'
print 'Training sentences: %d'%maxlen_s_train
print 'Training questions: %d'%maxlen_q_train
print 'Dev sentences: %d'%maxlen_s_dev
print 'Dev questions: %d'%maxlen_q_dev
print 'Test sentences: %d'%maxlen_s_test
print 'Test questions: %d\n'%maxlen_q_test


# Substitute answer with <a> token
def substitute_answer(sentence, answer):
    replaced_sentence = list()
    
    for i, line in enumerate(sentence):
        concat_sentence = ' '.join(line)
        concat_answer = ' '.join(answer[i])
        substitute_sentence = string.replace(concat_sentence, concat_answer, '<a>')
        replaced_sentence.append(substitute_sentence.split())
    
    return replaced_sentence

replaced_sentence_train = substitute_answer(sentence_train, answer_train)
replaced_sentence_dev = substitute_answer(sentence_dev, answer_dev)
replaced_sentence_test = substitute_answer(sentence_test, answer_test)


def filter_with_maxlen(maxlen_s, maxlen_q, sentence, sentence_origin, question, answer, cut_by_length):
    if cut_by_length:
        temp_sentence = list()
        temp_sentence_origin = list()
        temp_question = list()
        temp_answer = list()
        for i, line in enumerate(sentence):
            if (len(line) <= maxlen_s):
                temp_sentence.append(line)
                temp_sentence_origin.append(sentence_origin[i])
                temp_question.append(question[i])
                temp_answer.append(answer[i])


        # Filtering with maxlen(question)
        filtered_sentence = list()
        filtered_sentence_origin = list()
        filtered_question = list()
        filtered_answer = list()
        for i, line in enumerate(temp_question):
            if len(line) <= maxlen_q:
                filtered_sentence.append(temp_sentence[i])
                filtered_sentence_origin.append(temp_sentence_origin[i])
                filtered_question.append(line)
                filtered_answer.append(temp_answer[i])

        return filtered_sentence, filtered_sentence_origin, filtered_question, filtered_answer
    
    else:
        return sentence, sentence_origin, question, answer

# Filtering training data
filtered_sentence_train, filtered_sentence_origin_train, filtered_question_train, filtered_answer_train = filter_with_maxlen(
    maxlen_s_train, maxlen_q_train, replaced_sentence_train, sentence_train, question_train, answer_train, cut_by_length)        
# Filtering dev data
filtered_sentence_dev, filtered_sentence_origin_dev, filtered_question_dev, filtered_answer_dev = filter_with_maxlen(
    maxlen_s_dev, maxlen_q_dev, replaced_sentence_dev, sentence_dev, question_dev, answer_dev, cut_by_length) 

# Filtering test data
filtered_sentence_test, filtered_sentence_origin_test, filtered_question_test, filtered_answer_test = filter_with_maxlen(
    maxlen_s_test, maxlen_q_test, replaced_sentence_test, sentence_test, question_test, answer_test, cut_by_length) 


# Answer length calculation
maxlen_a_train = max([len(answer) for answer in filtered_answer_train])
maxlen_a_dev = max([len(answer) for answer in filtered_answer_dev])
maxlen_a_test = max([len(answer) for answer in filtered_answer_test])
    

    
# FILTERING WITH MAXLEN <<<



# SAVE TXT FILES FOR EVALUATION >>>



def save_txt(dir_1, dir_2, data):
    with open(os.path.join(dir_1,dir_2), 'w') as f:
        for line in data:
            f.write(' '.join(line).encode('utf-8') + '\n')

if not os.path.exists(text_dir):
    os.makedirs(text_dir)

    
save_txt(text_dir, sentence_txt_train_origin, filtered_sentence_origin_train)
save_txt(text_dir, sentence_txt_train, filtered_sentence_train)
save_txt(text_dir, question_txt_train, filtered_question_train)
save_txt(text_dir, answer_txt_train, filtered_answer_train)

save_txt(text_dir, sentence_txt_dev_origin, filtered_sentence_origin_dev)
save_txt(text_dir, sentence_txt_dev, filtered_sentence_dev)
save_txt(text_dir, question_txt_dev, filtered_question_dev)
save_txt(text_dir, answer_txt_dev, filtered_answer_dev)

save_txt(text_dir, sentence_txt_test_origin, filtered_sentence_origin_test)
save_txt(text_dir, sentence_txt_test, filtered_sentence_test)
save_txt(text_dir, question_txt_test, filtered_question_test)
save_txt(text_dir, answer_txt_test, filtered_answer_test)



# SAVE TXT FILES FOR EVALUATION <<<


# MAKE VOCABULARY WITH ALL FILTERED SENTENCES AND QUESTIONS >>>



if not vocab_include_answer:
    all_sentence = filtered_sentence_train + filtered_question_train
else:
    all_sentence = filtered_sentence_train + filtered_question_train + filtered_answer_train

wordcount = defaultdict(int)
for sentence in all_sentence:
    for word in sentence:
        wordcount[word]+=1
        
sorted_wordlist = [(k, wordcount[k]) for k in sorted(wordcount, key=wordcount.get, reverse=True)]

print '>>> Resizing dictionary with frequent words...',
resized_dic = dict(sorted_wordlist[:dic_size])

word2idx = dict()
word2idx['<PAD>'] = 0
word2idx['<GO>'] = 1
word2idx['<EOS>'] = 2
word2idx['<UNK>'] = 3
idx = 4
for word in resized_dic:
    word2idx[word] = idx
    idx += 1
print 'Complete\n'

# Save dic
print '>>> Saving Dic File...',
with open(vocab_dir, 'w') as f:
    pkl.dump(word2idx, f)
print 'Complete\n'
       
# Process with vocabulary

def process(data, vocab, maxlen, if_go = False):
    if if_go:
        maxlen = maxlen + 2 # include <GO> and <EOS>
    processed_data = list()
    length_data = list()
    for line in data:
        processed_data.append([])
        if if_go:
            processed_data[-1].append(vocab['<GO>'])
        for word in line:
            if word in vocab:
                processed_data[-1].append(vocab[word])
            else:
                processed_data[-1].append(vocab['<UNK>'])
        if if_go:
            processed_data[-1].append(vocab['<EOS>'])
        length_data.append(len(processed_data[-1]))
        processed_data[-1] = processed_data[-1] + [vocab['<PAD>']] * (maxlen - len(processed_data[-1]))
    return processed_data, length_data

print '>>> Processing data with vocabulary...',

# Processing training data
processed_sentence_train, length_sentence_train = process(filtered_sentence_train, word2idx, maxlen_s_train, if_go = False)
processed_question_train, length_question_train = process(filtered_question_train, word2idx, maxlen_q_train, if_go = True)
processed_answer_train, length_answer_train = process(filtered_answer_train, word2idx, maxlen_a_train, if_go = False)

# Processing dev data
processed_sentence_dev, length_sentence_dev = process(filtered_sentence_dev, word2idx, maxlen_s_dev, if_go = False)
processed_question_dev, length_question_dev = process(filtered_question_dev, word2idx, maxlen_q_dev, if_go = True)
processed_answer_dev, length_answer_dev = process(filtered_answer_dev, word2idx, maxlen_a_dev, if_go = False)

# Processing test data
processed_sentence_test, length_sentence_test = process(filtered_sentence_test, word2idx, maxlen_s_dev, if_go = False)
processed_question_test, length_question_test = process(filtered_question_test, word2idx, maxlen_q_dev, if_go = True)
processed_answer_test, length_answer_test = process(filtered_answer_test, word2idx, maxlen_a_test, if_go = False)

print 'Complete\n'



# MAKE VOCABULARY WITH ALL FILTERED SENTENCES AND QUESTIONS <<<



# SAVE PROCESSED DATA >>>



print '>>> Saving processed data...',
np.save(sentence_outfile_train, processed_sentence_train)
np.save(question_outfile_train, processed_question_train)
np.save(answer_outfile_train, processed_answer_train)
np.save(length_sentence_outfile_train, length_sentence_train)
np.save(length_question_outfile_train, length_question_train)
np.save(length_answer_outfile_train, length_answer_train)

np.save(sentence_outfile_dev, processed_sentence_dev)
np.save(question_outfile_dev, processed_question_dev)
np.save(answer_outfile_dev, processed_answer_dev)
np.save(length_sentence_outfile_dev, length_sentence_dev)
np.save(length_question_outfile_dev, length_question_dev)
np.save(length_answer_outfile_dev, length_answer_dev)

np.save(sentence_outfile_test, processed_sentence_test)
np.save(question_outfile_test, processed_question_test)
np.save(answer_outfile_test, processed_answer_test)
np.save(length_sentence_outfile_test, length_sentence_test)
np.save(length_question_outfile_test, length_question_test)
np.save(length_answer_outfile_test, length_answer_test)
print 'Complete\n'



# SAVE PROCESSED DATA <<<

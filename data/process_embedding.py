import numpy as np
import pickle as pkl
import os
from tqdm import tqdm

print '########## Embedding processing start ##########\n'

# SETTINGS >>>



dic_dir = 'processed/mpqg_substitute_a_vocab_include_a/'
dic_name = 'vocab.dic'
embedding_name = 'glove840b_vocab300.npy'

data_dir = 'GloVe/'
glove = 'glove.840B.300d'



# SETTINGS <<<



# LOAD & PROCESS GloVe >>>


if not os.path.exists('processed/'+ glove + '.dic.npy'):
    # Load GloVe
    print '>>> Reading GloVe file...',
    f = open(data_dir + glove + '.txt')
    lines = f.readlines()
    f.close()
    print 'Complete\n'

    # Process GloVe
    print '>>> Processing Glove...'
    embedding = dict()
    for line in tqdm(lines):
        splited = line.split()
        embedding[splited[0]] = map(float, splited[1:])

    # Save processed GloVe as dic file
    print '>>> Saving processed GloVe...'
    np.save('processed/' + glove + '.dic', embedding)
    print 'Complete\n'
else:
    print 'Processed GloVe exists!'
    print '>>> Loading processed GloVe file...'
    embedding = np.load('processed/' + glove + '.dic.npy').item()
    print 'Complete\n'




# LOAD & PROCESS GloVe <<<


# PRODUCE PRE-TRAINED EMBEDDING >>>


# Load vocabulary
print '>>> Loading vocabulary...',
with open(os.path.join(dic_dir, dic_name)) as f:
    vocab = pkl.load(f)
print 'Complete\n'

# Initialize random embedding and extract pre-trained embedding
print '>>> Producing pre-trained embedding...',
embedding_vocab =  np.random.ranf((len(vocab), 300)) -  np.random.ranf((len(vocab), 300))

embedding_vocab[0] = 0.0 # vocab['<PAD>'] = 0
embedding_vocab[1] = embedding['<s>'] # vocab['<GO>'] = 1
embedding_vocab[2] = embedding['EOS'] # vocab['<EOS>'] = 2
embedding_vocab[3] = embedding['UNKNOWN'] # vocab['<UNK>'] = 3

unk_num = 0
for word, idx in vocab.items():
    if word in embedding:
        embedding_vocab[idx] = embedding[word]
    else:
        unk_num += 1
print 'Complete\n'
        
# Save embedding    
print '>>> Saving pre-trained embedding',
np.save(os.path.join(dic_dir, embedding_name), embedding_vocab)
print 'Complete\n'

print '---------- Statistics ----------'
print 'Vocabulary size: %d'%len(embedding_vocab)
print 'Unknown words: %d'%unk_num



# PRODUCE PRE-TRAINED EMBEDDING <<<


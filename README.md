# NQG_ASs2s

** Still updating...**
Implementation of &lt;Improving Neural Question Generation Using Answer Separation> by Yanghoon Kim et al.

**The source code still needs to be modified**


1. **Model**

	- Embedding
	  - Pretrained GloVe embeddings
	  - Randomly initialized embeddings

	- Answer-separated seq2seq
	  - Answer-separated encoder
	  - Answer-separated decoder
	    - Keyword-net
		- Retrieval style word generator
	
	- Post processing
	  - Remove repetition

2. **Dataset**

Processed data provided by [Linfeng Song et al.](https://www.aclweb.org/anthology/N18-2090)

## Requirements

- python 2.7
- numpy
- Tensorflow 1.4

## Usage

1. Data preprocessing

```
# Extract dataset
tar -zxvf data/mpqg_data/nqg_data.tgz -C data/mpqg_data

# Process data
cd data
python process_mpqg_data.py # Several settings can be modified inside the source code (data path, vocab_size, etc)
```

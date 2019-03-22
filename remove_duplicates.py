import argparse

FLAGS = None
def check_and_remove_duplicates(sentence, unit_size, start_idx):
    if unit_size == 1:
        sentence = sentence.split()
        filtered_sentence = ['']
        for word in sentence:
            if word != filtered_sentence[-1]:
                filtered_sentence.append(word)
        filtered_sentence = ' '.join(filtered_sentence[1:]).strip()
        return filtered_sentence
    
    end_idx = len(sentence.split()) - 1
    if (end_idx - start_idx + 1)/ unit_size >= 2:
        sentence_ = sentence.split()
        pre_words = ' '.join(sentence_[:start_idx]) if start_idx != 0 else ''
        
        remain_from = (end_idx - start_idx + 1) % unit_size
        post_words = ' '.join(sentence_[-remain_from:]) if remain_from != 0 else ''

        sequence_list = ['']
        for idx in range(start_idx, end_idx - remain_from, unit_size):
            temp_sequence = ' '.join(sentence_[idx:idx + unit_size])

            if sequence_list[-1] != temp_sequence:
                sequence_list.append(temp_sequence)
        
        middle_sentence = ' '.join(sequence_list).strip()
                            
        filtered_sentence = pre_words + ' ' + middle_sentence + ' ' + post_words
        filtered_sentence = filtered_sentence.strip()
        
        return check_and_remove_duplicates(filtered_sentence, unit_size, start_idx + 1)
    
    else:
        return sentence      
    
def check_and_remove_duplicates_loop(sentence, current_length, sequence_up_to):
    if current_length <= sequence_up_to:
        filtered_sentence = check_and_remove_duplicates(sentence, current_length, 0)
        return check_and_remove_duplicates_loop(filtered_sentence, current_length + 1, sequence_up_to)
    else:
        return sentence 
    
def main():
    FILE_IN = FLAGS.source_file
    FILE_OUT = FLAGS.out_file
    NGRAM = FLAGS.ngram
    
    with open(FILE_IN) as g:
        data = g.readlines()
    with open(FILE_OUT, 'w') as g:
        for line in data:
            g.write(check_and_remove_duplicates_loop(line, 1, NGRAM) + '\n')
                        
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source_file', type = str, default = 'result/predictions.txt')
    parser.add_argument('--out_file', type = str, default = 'result/predictions.rmv')
    parser.add_argument('--ngram', type = int, default = 4)
    FLAGS = parser.parse_args()
    main()

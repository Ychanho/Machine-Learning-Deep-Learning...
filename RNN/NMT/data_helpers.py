import numpy as np
import re
import glob
import collections

def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z가-ퟻ0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()

def load_data(s_file, t_file):
    # Load data from files

    x_text = list(open(s_file, "r", encoding='UTF8').readlines())
    x_text = [s.strip() for s in x_text]
    x_text = np.array([clean_str(sent) for sent in x_text])

    # x_lengths = list(map(len, [sent.split(" ") for sent in x_text]))
    # x_lengths = np.array([length + 2 for length in x_lengths])

    t_text = list(open(t_file, "r", encoding='UTF8').readlines())
    t_text = [s.strip() for s in t_text]
    t_text = np.array([clean_str(sent) for sent in t_text])

    # t_lengths = list(map(len, [sent.split(" ") for sent in t_text]))
    # t_lengths = np.array([length + 2 for length in t_lengths])

    return [x_text, t_text]

def load_en_data(s_file):
    # Load data from files

    x_text = list(open(s_file, "r", encoding='UTF8').readlines())
    x_text = [s.strip() for s in x_text]
    x_text = np.array([clean_str(sent) for sent in x_text])

    x_lengths = np.array(list(map(len, [sent.split(" ") for sent in x_text])))

    return [x_text, x_lengths]

def buildVocab(sentences, vocab_size):
    # Build vocabulary
    words = []
    for sentence in sentences:
        words.extend(sentence.split(" "))

    print("The number of words: ", len(words))
    word_counts = collections.Counter(words)
    # Mapping from index to word
    vocabulary_inv = [x[0] for x in word_counts.most_common(vocab_size)]
    # Mapping from word to index
    vocabulary = {x: i for i, x in enumerate(vocabulary_inv)}
    return [vocabulary, vocabulary_inv]

def text_to_index(text_list, word_to_id):
    text_indices = []
    lengths = []
    for text in text_list:
        words = text.split(" ")
        ids = [1]  # <s>
        for word in words:
            if word in word_to_id:
                word_id = word_to_id[word] # word -> apple, word_id -> 1017
            else:
                word_id = 3  # <unk>
            ids.append(word_id)
        ids.append(2)  # <eos>
        text_indices.append(ids)
        lengths.append(len(ids))

    lengths = np.asarray(lengths, dtype=np.int64)
    return text_indices, lengths

def text_to_index_ko(text_list, word_to_id):
    text_indices = []
    lengths = []
    for text in text_list:
        words = text.split(" ")
        ids = []
        for word in words:
            if word in word_to_id:
                word_id = word_to_id[word]  # word -> apple, word_id -> 1017
            else:
                word_id = 3  # <unk>
            ids.append(word_id)
        text_indices.append(ids)
        lengths.append(len(ids) + 1)

    lengths = np.asarray(lengths, dtype=np.int64)
    return text_indices, lengths


def batch_tensor(batches):
    max_length = max([len(batch) for batch in batches])
    tensor = np.zeros((len(batches), max_length), dtype=np.int64)
    for i, indices in enumerate(batches):
        tensor[i, :len(indices)] = np.asarray(indices, dtype=np.int64)

    return tensor

def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]
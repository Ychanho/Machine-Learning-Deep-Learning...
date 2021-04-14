import numpy as np
import re
import collections

def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
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

def load_data(x_file, t_file):
    # Load data from files

    t_large = []

    x_text = list(open(x_file, "r", encoding='UTF8').readlines())
    x_text = [s.strip() for s in x_text]
    x_text = np.array([clean_str(sent) for sent in x_text])

    lengths = np.array(list(map(len, [sent.split(" ") for sent in x_text])))
    t_text_temp = np.array(list(open(t_file, "r", encoding='UTF8').readlines()))

    maxLabel = t_text_temp.astype(np.int)
    print(maxLabel)
    maxLabel = np.max(maxLabel) + 1
    print("max label: "+str(maxLabel))
    for i, s in enumerate(t_text_temp):
        t = np.zeros(maxLabel)
        t[int(s)] = 1.0
        t_large.append(t)

    t_large = np.array(t_large)

    return [x_text, t_large, lengths]

def buildVocab(sentences, vocab_size):
    # Build vocabulary
    words = []
    for sentence in sentences: words.extend(sentence.split())  # i, am, a, boy, you, are, a, girl
    print("The number of words: ", len(words))
    word_counts = collections.Counter(words)
    # Mapping from index to word
    vocabulary_inv = [x[0] for x in word_counts.most_common(vocab_size)]
    # vocabulary_inv = list(sorted(vocabulary_inv))
    # Mapping from word to index
    vocabulary = {x: i for i, x in enumerate(vocabulary_inv)}  # a: 0, i: 1...
    return [vocabulary, vocabulary_inv]

def text_to_index(text_list, word_to_id, nb_pad):
    text_indices = []
    for text in text_list:
        words = text.split(" ")
        pad = [0 for _ in range(nb_pad) ]
        ids = []
        for word in words: # i, am, a, boy
            if word in word_to_id:
                word_id = word_to_id[word]
            else:
                word_id = 1 # OOV (out-of-vocabulary)
            ids.append(word_id) # 5, 8, 6, 19
        ids = pad + ids # 0, 0, 0, 0, 5, 8, 6, 19
        text_indices.append(ids)
    return text_indices

def train_tensor(batches):
    max_length = max([len(batch) for batch in batches]) # 100
    tensor = np.zeros((len(batches), max_length), dtype=np.int64) #(5000, 100)
    # 0 0 0 0 5 8 6 19 0 0....0 0 0
    # 0 0 0 0 5 7 11 1 1 1....1 1 1
    # 0 0 0 0 0 0 0 0 0 0....0 0 0
    # 0 0 0 0 0 0 0 0 0 0....0 0 0
    #...
    # 0 0 0 0 0 0 0 0 0 0....0 0 0
    for i, indices in enumerate(batches):
        tensor[i, :len(indices)] = np.asarray(indices, dtype=np.int64)
    return tensor, max_length

def test_tensor(batches, max_length):
    tensor = np.zeros((len(batches), max_length), dtype=np.int64)
    for i, indices in enumerate(batches):
        if len(indices) > max_length:
            tensor[i, :max_length] = np.asarray(indices[:max_length], dtype=np.int64)
        else:
            tensor[i, :len(indices)] = np.asarray(indices, dtype=np.int64)

    return tensor

def batch_tensor(batches):
    max_length = max([len(batch) for batch in batches])
    tensor = np.zeros((len(batches), max_length), dtype=np.int64) # 50, 22
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
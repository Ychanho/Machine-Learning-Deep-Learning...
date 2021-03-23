import numpy as np
import re
import glob

def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string) # kwon's -> kwon 's
    string = re.sub(r"\'ve", " \'ve", string) #I've -> i 've
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


def load_data_and_labels(positive_data_file, negative_data_file):
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    # Load data from files
    positive_examples = list(open(positive_data_file, "r", encoding='utf-8').readlines())
    positive_examples = [s.strip() for s in positive_examples]
    negative_examples = list(open(negative_data_file, "r", encoding='utf-8').readlines())
    negative_examples = [s.strip() for s in negative_examples]
    # Split by words
    x_text = positive_examples + negative_examples
    x_text = [clean_str(sent) for sent in x_text]
    # Generate labels
    positive_labels = [[0, 1] for _ in positive_examples]
    negative_labels = [[1, 0] for _ in negative_examples]
    y = np.concatenate([positive_labels, negative_labels], 0)
    return [x_text, y]

def load_imdb_data_and_labels(pos_file, neg_file):
    # Load data from files
    pos_list = glob.glob(pos_file)  # load file list
    pos_final = []  # sentence list

    for pos in pos_list:
        x_text = list(open(pos, "r", encoding='UTF8').readlines())
        x_text = [clean_str(sent) for sent in x_text]
        pos_final = pos_final + x_text

    neg_list = glob.glob(neg_file)
    neg_final = []

    for neg in neg_list:
        x_text = list(open(neg, "r", encoding='UTF8').readlines())
        x_text = [clean_str(sent) for sent in x_text]
        neg_final = neg_final + x_text

    positive_labels = [[0, 1] for _ in pos_final]  # [[0,1], [0,1], [0,1], [0,1], [0,1], [0,1],...]
    negative_labels = [[1, 0] for _ in neg_final]  # [[1,0], [1,0], [1,0], [1,0], [1,0], [1,0],...]

    y = np.concatenate([positive_labels, negative_labels], 0)  # [[0,1], [0,1], [0,1], [0,1], [0,1], [0,1],...[1,0], [1,0], [1,0], [1,0], [1,0], [1,0],...]
    print(pos_final)
    print(len(pos_final))
    print(len(neg_final))
    print(len(y))

    x_final = pos_final + neg_final
    print(len(x_final[0]))
    return [x_final, y]

def load_real_imdb_data_and_labels(text_data_file, score_data_file):
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    # Load data from files
    text_list = list(open(text_data_file, "r", encoding='utf-8').readlines())
    text_list = [s.strip() for s in text_list]
    score_list = list(open(score_data_file, "r", encoding='utf-8').readlines())
    score_list = [s.strip() for s in score_list]
    # Split by words
    x_text = [clean_str(sent) for sent in text_list]
    print(score_list)
    y = []
    for score in score_list:
        if int(score) > 5:
            y.append([0, 1])
        else:
            y.append([1, 0])
    print(y)
    return [x_text, y]

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
import numpy as np
import re
from sklearn.model_selection import train_test_split

def clean_str(string):
    """
    Tokenization/string cleaning 
    """
    string = re.sub(r"[^A-Za-zа-яА-Я0-9(),:!?\'\`]", " ", string)
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


def load_tweets(tweets):
    strings = list(open(tweets, "r").readlines())
    strings = [s.strip() for s in strings]
    all_examples=[clean_str(sent) for sent in strings]
    
    return all_examples

def load_1dsentiment(labels):
    """
    1 if there is a sentiment, 0 if not 
    """
    labels = list(open(labels, "r").readlines())
    y_text = np.array([abs(int(re.sub(r"\n", "", sent))) for sent in labels])
    
    return y_text

def load_2dsentiment(labels):
    """
    1 if there is a sentiment, 0 if not 
    """
    labels = list(open(labels, "r").readlines())
    y_text = np.array([[abs(int(re.sub(r"\n", "", sent))), 1-abs(int(re.sub(r"\n", "", sent)))] for sent in labels])

    return y_text


def load_labels(labels):
    """
    TODO 
    """
    labels = list(open(labels, "r").readlines())
    y_text = np.array([[abs(int(re.sub(r"\n", "", sent))), 1-abs(int(re.sub(r"\n", "", sent)))] for sent in labels])

    return y_text

def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = list(data)
    data_size = len(data)
    num_batches_per_epoch = int(data_size/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = (np.array(data))[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]







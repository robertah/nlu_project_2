from config import *
import numpy as np
import pickle
from collections import Counter


def count_words(df):
    """
    Count words in sentences columns in the given dataframe

    :param df: dataframe containing train / val / test data
    :return: counter of words
    """

    # initialize counter
    words = Counter()

    # get dataframe columns with sentences
    sen_cols = [s for s in df if s.startswith('sen')]

    # add sentences to counter
    for col in sen_cols:
        for i, row in df.iterrows():
            words.update(list(row[col]))
    print("Found", len(words), "words in dataset:")
    #print(words)

    return words


def generate_vocabulary(df):
    """
    Generate vocabulary and save it as pickle

    :param df: dataframe containing train / val / test data
    :param voc_size: vocabulary size specified in config file (None, if do not want to limit vocabulary)
    :return:
    """

    print("Generating vocabulary...")

    words = count_words(df)

    # create vocabulary
    if vocabulary_size is not None:
        vocabulary = dict((x, y) for x, y in words.most_common(vocabulary_size - (1 if sentence_len is None else 2)))
    else:
        vocabulary = words

    # generate index for each word
    for i, k in enumerate(vocabulary.keys()):
        vocabulary[k] = i

    # if sentence length is fixed add pad
    if sentence_len is not None:
        vocabulary.update({pad: len(vocabulary)})

    # if vocabulary size is fixed add unk
    if vocabulary_size is not None:
        vocabulary.update({unk: len(vocabulary)})

    #print("Vocabulary generated: \n", vocabulary)
    with open(vocabulary_pkl, 'wb') as output:
        pickle.dump(vocabulary, output, pickle.HIGHEST_PROTOCOL)
        print("Vocabulary saved as pkl")

    return vocabulary


def load_vocabulary():
    """
    Load existing vocabulary

    :return: vocabulary
    """

    print("Loading vocabulary... ")

    try:
        with open(vocabulary_pkl, 'rb') as handle:
            vocabulary = pickle.load(handle)
        print("Vocabulary loaded")
    except FileNotFoundError:
        print("Vocabulary not found.")

    return vocabulary


def get_words_from_indexes(indexes, vocabulary):
    """
    Get words from indexes in the vocabulary

    :param indexes: list of indexes of words in vocabulary
    :param vocabulary: vocabulary
    :return: words corresponding to given indexes
    """

    # map indexes to words in vocabulary
    vocabulary_reverse = {v: k for k, v in vocabulary.items()}

    # retrieve words corresponding to indexes
    words = [vocabulary_reverse[x] for x in indexes]
    return words


def get_indexes_from_words(words, vocabulary):
    """
    Get indexes from words in the vocabulary

    :param words: list of words in vocabulary
    :param vocabulary: vocabulary
    :return: indexes corresponding to given words
    """
    # retrieve indexes corresponding to words
    indexes = [vocabulary[x] for x in words]
    return indexes


def wrap_sentence(sentence, vocabulary):
    """
    Wrap sentence according to vocabulary and sentence length

    :param sentence: raw sentence
    :param vocabulary: vocabulary built from train set
    :return: wrapped sentence with indexes
    """

    wrapped_sentence = []

    # replace words not in vocabulary with unk
    for word in sentence:
        wrapped_sentence.append(word if word in vocabulary.keys() else unk)

    # if sentence length is fixed
    if sentence_len is not None:
        # pad sentence if it is too short
        while len(wrapped_sentence) <= sentence_len:
            wrapped_sentence.append(pad)
        # trim the sentence if it is too long
        if len(wrapped_sentence) > sentence_len:
            wrapped_sentence = wrapped_sentence[:sentence_len]

    # get word indexes from vocabulary
    wrapped_sentence_w_index = get_indexes_from_words(wrapped_sentence, vocabulary)

    return wrapped_sentence_w_index


def generate_vocab_pos(pos_data):
    """
    Generate a pos_vocabulary file and save it in pkl form
    :param pos_data:
    :return:
    """

    matrix = np.load(pos_data)

    numrows = matrix.shape[0]
    numsen = matrix.shape[1]



    list_pos = list()

    for item in matrix[0:numrows, 1:numsen]:
        for sentence in item[0:numrows]:
            list_pos = list_pos + sentence[:, 1].tolist()
    # print(list(set(list_pos)))

    #creating dictionary with pos_tags, using negative numbers
    pos_dic = dict(enumerate(list(set(list_pos))))
    pos_dictionary = {v: -(k+1) for k, v in pos_dic.items()}

    print(pos_dictionary)

    with open(pos_vocabulary_pkl , 'wb') as output:
        pickle.dump(pos_dictionary, output, pickle.HIGHEST_PROTOCOL)
        print("Pos vocabulary saved as pkl")

    return pos_dictionary


def merge_vocab(vocab1, vocab2):
    """
    Merges two dictionaries in pkl format and saves as new dictionary
    :param vocab1, vocab2: dictionaries to merge in pkl format
    :return: merged dictionary
    """

    with open(vocab1, 'rb') as f:
        data1 = pickle.load(f)

    with open(vocab2, 'rb') as f:
        data2 = pickle.load(f)

    data1.update(data2)

    with open(full_vocabulary_pkl , 'wb') as output:
        pickle.dump(data1, output, pickle.HIGHEST_PROTOCOL)
        print("Full vocabulary saved as pkl")

    return data1


if __name__ == '__main__':

    generate_vocab_pos(train_pos_begin)
    merge_vocab(vocabulary_pkl, pos_vocabulary_pkl)

    with open(full_vocabulary_pkl, 'rb') as f:
        data = pickle.load(f)
    print(data)
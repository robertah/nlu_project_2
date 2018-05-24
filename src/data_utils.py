from config import *
import pandas as pd
import numpy as np
import pickle
from collections import Counter

def load_data(dataset):
    """
    Build the dataframe from the given dataset

    :param dataset: path to csv data file (train, val or test data)
    :return: dataframe corresponding to the dataset
    """

    assert dataset == train_set or dataset == val_set or dataset == test_set

    if dataset == train_set:
        names = ['id', 'title', 'sen1', 'sen2', 'sen3', 'sen4', 'sen5']
    else:
        names = ['id', 'sen1', 'sen2', 'sen3', 'sen4', 'sen5_1', 'sen5_2', 'ans']

    return pd.read_csv(dataset, index_col='id', names=names, skiprows=1)


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
    print("Found", len(words), "words in dataset.")
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


def read_sentences(df):
    """
    Get array of sentences from the given dataframe

    :param df: dataframe containing train / val / test data
    :return: numpy array with sentences
    """

    # filter by sentences columns
    story_df = df.loc[:, df.columns.str.startswith('sen')]

    # get sentences
    sentences = story_df.values

    return sentences


def read_stories(df):
    """
    Get array of stories from the given dataframe

    :param df: dataframe containing train / val / test data
    :return: numpy array with stories
    """

    # read sentences from dataframe
    sentences = read_sentences(df)

    # get number of stories
    n_stories, *_ = sentences.shape

    # gather sentences together for each story
    stories = []
    for i, s in enumerate(sentences):
        story = []
        for ss in s:
            story.extend(ss)
        stories.append(story)

    # convert list of stories to numpy array
    stories = np.asarray(stories)

    return stories


def check_for_unk(sentence, vocabulary):
    """
    Check for unk in sentence given the vocabulary

    :param sentence: raw sentence as list of words
    :param vocabulary: vocabulary generated during training
    :return: sentence where missing words are replaced by unk
    """

    new_sentence = []

    # replace words not in vocabulary with unk
    for word in sentence:
        new_sentence.append(word if word in vocabulary.keys() else unk)

    return new_sentence


def pad_sentences(sentences):
    """
    Pad sentences according to sentence_len

    :param sentences: list of sentences
    :return: array of padded sentences
    """

    padded_sentences = []

    # len_sentences = np.asarray([len(s) for s in sentences])
    # print('max:', np.max(len_sentences))

    for sentence in sentences:

        # trim the sentence if it is too long
        if len(sentence) > sentence_len:
            padded_sentence = sentence[:sentence_len]

        # pad the sentence if it is too short
        else:
            padded_sentence = sentence

            while len(padded_sentence) < sentence_len:
                padded_sentence.append(pad)

        padded_sentences.append(padded_sentence)

    padded_sentences = np.asarray(padded_sentences)

    return padded_sentences


def pad_stories(stories):
    """
    Pad stories according to story_len

    :param stories: list of stories
    :return: array of padded stories
    """

    padded_stories = []

    # len_stories = np.asarray([len(s) for s in stories])
    # print('max:', np.max(len_stories))

    for story in stories:

        # trim the story if it is too long
        if len(story) > story_len:
            padded_story = story[:story_len]

        # pad the story if it is too short
        else:
            padded_story = story

            while len(padded_story) < story_len:
                padded_story.append(pad)

        padded_stories.append(padded_story)

    padded_stories = np.asarray(padded_stories)

    return padded_stories


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
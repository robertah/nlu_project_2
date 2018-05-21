from config import *
import numpy as np
import pandas as pd
import nltk
import os
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem import SnowballStemmer
from nltk import download
from data_utils import wrap_sentence, generate_vocabulary, load_vocabulary
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')


def _load_data(dataset):
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


def _tokenize(dataframe, stop_words=True, lemmatize=False, stem=True):
    """
    Tokenize sentences in the given dataframe

    :param dataframe: dataframe containing train / val / test data
    :param stop_words: True to remove stop words and punctuation
    :param stem: True to stem words
    :return: dataframe with tokenized sentences
    """

    df = dataframe.copy()

    # select columns containing sentences
    sen_cols = [s for s in df if s.startswith('sen')]

    # tokenize sentences and remove punctuation
    tokenizer = RegexpTokenizer(r'\w+')
    for col in sen_cols:
        df[col] = df[col].str.lower()
        df[col] = df[col].apply(tokenizer.tokenize)

    # remove stop words
    if stop_words:
        download('stopwords')
        stop = set(stopwords.words('english'))
        for col in sen_cols:
            df[col] = df[col].apply(lambda x: [y for y in x if y not in stop])

    # lemmatize words  TODO need to add pos tagger to make it work
    if lemmatize:
        download('wordnet')
        lemmatizer = WordNetLemmatizer()
        for col in sen_cols:
            df[col] = df[col].apply(lambda x: [lemmatizer.lemmatize(y) for y in x])

    # stem words
    if stem:
        stemmer = SnowballStemmer('english')
        for col in sen_cols:
            df[col] = df[col].apply(lambda x: [stemmer.stem(y) for y in x])

    return df


def preprocess(dataset, pos_tagging=False):
    """
    Preprocess raw train / val / test data files

    :param dataset: path to dataset
    :return: original dataset and dataset with processed sentences
    """

    assert dataset == train_set or dataset == val_set or dataset == test_set

    print("Preprocessing...")

    # load data from csv
    data_original = _load_data(dataset)

    # tokenize sentences in dataframe
    data_processed = _tokenize(data_original)

    # generate vocabulary if training, otherwise load existing vocabulary
    if dataset == train_set:
        vocabulary = generate_vocabulary(data_processed)
    else:
        vocabulary = load_vocabulary()

    # filter columns by those containing sentences
    sen_cols = [s for s in data_processed if s.startswith('sen')]
    for col in sen_cols:
        for i, row in data_processed.iterrows():
            # process sentences according to built vocabulary
            # row[col] = wrap_sentence(list(row[col]), vocabulary)
            data_processed.set_value(i, col, wrap_sentence(list(row[col]), vocabulary))

    else:
        return data_original, data_processed


def pos_tagging_text(sentence):
    tokens = nltk.word_tokenize(sentence)
    return nltk.pos_tag(tokens)


def pos_tag_dataset(dataset):
    # load data from csv
    data_original = _load_data(dataset)

    pos_begin = pd.DataFrame(columns=['sen1', 'sen2', 'sen3', 'sen4'])
    pos_begin.index.name = 'id'
    pos_end = pd.DataFrame(columns=['sen5'])
    pos_end.index.name = 'id'
    for index, row in data_original.iterrows():
        pos_begin.loc[index] = [pos_tagging_text(row['sen1']), pos_tagging_text(row['sen2']),
                               pos_tagging_text(row['sen3']), pos_tagging_text(row['sen4'])]
        pos_end.loc[index] = [pos_tagging_text(row['sen5'])]

    #saving models in two data files
    cur_dir = os.path.splitext(dataset)[0]
    path_begin = cur_dir + "_pos_begin.csv"
    path_end = cur_dir + "_pos_end.csv"
    pos_begin.to_csv(path_or_buf= path_begin, columns=['sen1', 'sen2', 'sen3', 'sen4'])
    pos_end.to_csv(path_or_buf=path_end, columns=['sen5'])
    print("Model saved to {}".format(path_begin))
    print("Model saved to {}".format(path_end))
    return None


def get_story_matrices(df):
    """
    Get numerical matrices for story beginning and ending given the dataframe.

    :param df: dataframe containing train / val / test data
    :return: story beginning and ending (if val / test data, in the ending the correct sentence is the first one)
    """

    # filter by sentences columns
    story_df = df.loc[:, df.columns.str.startswith('sen')]

    # convert dataframe values into numpy array
    story_values = story_df.values

    # get story array's size
    n_stories, n_sentences = story_values.shape

    # create story matrix
    story = np.empty([n_stories, sentence_len * n_sentences], dtype=int)
    for i, row in enumerate(story_values):
        story[i] = np.concatenate(row)

    story = np.reshape(story, (n_stories, -1, sentence_len))

    # if val or test data, the ending matrix has to contain the correct sentence at first index
    if n_sentences != 5:
        for i, row in df.iterrows():
            # if the answer is the second sentence, swap the two answers
            if row['ans'] == 2:
                pos = df.index.get_loc(i)
                story[pos, n_sentences - 1, :], story[pos, n_sentences - 2, :] = \
                    story[pos, n_sentences - 2, :], story[pos, n_sentences - 1, :]

    # get beginning and ending matrices
    beginning = story[:, :4, :]
    ending = story[:, 4:, :]

    return beginning, ending


# just trying if works
if __name__ == '__main__':
    # data_orig, data_proc = preprocess(train_set)
    # x_begin, x_end = get_story_matrices(data_proc)
    # print(x_begin)
    # print(x_end)
    # n_stories, *_ = x_begin.shape
    # x_begin = np.reshape(x_begin, (n_stories, -1))
    # print(x_begin.shape)

    # data_orig, data_proc, pos_text = preprocess(train_set, pos_tagging=True)
    # print(data_orig)
    # print(data_proc)
    # print(pos_text)
    dataset=train_set
    pos_tag_dataset(dataset)

    # sentences = _load_data(train_set)
    # print(sentences)
    # pos_text = pos_tagging_text(sentences)
    # print(pos_text)
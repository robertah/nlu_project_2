from config import *
import pandas as pd
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk import download
from data_utils import wrap_sentence, generate_vocabulary, load_vocabulary


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


def _tokenize(dataframe, stop_words=True, stem=True):
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

    # tokenize sentences
    tokenizer = RegexpTokenizer(r'\w+')
    for col in sen_cols:
        df[col] = df[col].str.lower()
        df[col] = df[col].apply(tokenizer.tokenize)

    # remove stop words and punctuation
    if stop_words:
        download('stopwords')
        stop = set(stopwords.words('english'))
        for col in sen_cols:
            df[col] = df[col].apply(lambda x: [y for y in x if y not in stop])

    # stem words
    if stem:
        stemmer = SnowballStemmer('english')
        for col in sen_cols:
            df[col] = df[col].apply(lambda x: [stemmer.stem(y) for y in x])

    return df


def preprocess(dataset):
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
            row[col] = wrap_sentence(list(row[col]), vocabulary)

    return data_original, data_processed


# just trying if works
if __name__ == '__main__':
    data_orig, data_proc = preprocess(train_set)
    print(data_orig)
    print(data_proc)

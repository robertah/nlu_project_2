import nltk
import os
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import wordnet
from nltk.stem import SnowballStemmer
from nltk import download
from data_utils import *
# from word_embeddings import word_embedding

def tokenize(dataframe, stop_words=True, lemmatize=False, stem=False):
    """
    Tokenize sentences in the given dataframe

    :param dataframe: dataframe containing train / val / test data
    :param stop_words: True to remove stop words and punctuation
    :param lemmatize: True to lemmatize words
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
        lemmatizer = wordnet.WordNetLemmatizer()
        for col in sen_cols:
            df[col] = df[col].apply(lambda x: [lemmatizer.lemmatize(y) for y in x])

    # stem words
    if stem:
        stemmer = SnowballStemmer('english')
        for col in sen_cols:
            df[col] = df[col].apply(lambda x: [stemmer.stem(y) for y in x])

    return df


def preprocess(dataset, pad='story'):
    """
    Preprocess raw train / val / test data files

    :param dataset: path to dataset
    :return: original dataset with tokenized sentences and processed dataset with padded and indexed sentences
    """

    assert dataset == train_set or dataset == val_set or dataset == test_set

    print("Preprocessing...")

    # load data from csv
    data_original = load_data(dataset)

    # tokenize sentences in dataframe
    data_original = tokenize(data_original)

    # generate vocabulary if training, otherwise load existing vocabulary
    if dataset == train_set:
        vocabulary = generate_vocabulary(data_original)
    else:
        vocabulary = load_vocabulary()

    # filter columns by those containing sentences
    sen_cols = [s for s in data_original if s.startswith('sen')]

    # replace missing words with unk
    for col in sen_cols:
        for i, row in data_original.iterrows():
            data_original.at[i, col] = check_for_unk(list(row[col]), vocabulary)

    # get number of stories and number of sentences per story
    n_stories = len(data_original)
    n_sentences = len(sen_cols)

    # pad stories or sentences or none
    if pad == 'story':
        stories = read_stories(data_original)
        data_processed = pad_stories(stories)
    else:  # pad == 'sentence':
        sentences = read_sentences(data_original)
        sentences = np.reshape(sentences, (n_stories*n_sentences))
        data_processed = pad_sentences(sentences)

    # reshape processed data
    data_processed = np.reshape(data_processed, (n_stories, -1))

    # index words in stories according to the vocabulary
    for i, story in enumerate(data_processed):
        data_processed[i] = get_indexes_from_words(story, vocabulary)

    return data_original, data_processed


def pos_tagging_text(sentence):
    tokens = nltk.word_tokenize(sentence)
    return nltk.pos_tag(tokens)


def pos_tag_dataset(dataset, seperate=False):
    """
    Saves two files containing pos-tagged sentences.
    Arrays of size #stories/rows by 5 (story id, sen1 to sen4) and of size #stories/rows by 2 (story id, sen5).
    Each column (not id) is a tockenized and pos-tagged sentence of the form (ie: [[Kelly, RB], [studied, VBN], [.,.]])

    :param dataset: dataframe containing train / val / test data
    :return: none
    """

    nltk.download('punkt')
    nltk.download('averaged_perceptron_tagger')

    # load data from csv
    data_original = load_data(dataset)

    # Creates dataframes with pos-tagged sentences
    pos_end = pd.DataFrame(columns=['id', 'sen5'])

    story_number = 0
    total_stories = len(pos_end)

    if seperate:
        pos_begin = pd.DataFrame(columns=['id', 'sen1', 'sen2', 'sen3', 'sen4'])
        pos_begin = pd.DataFrame(columns=['id', 'sen'])
        for index, row in data_original.iterrows():
            pos_begin.loc[index] = [index,
                                    np.asarray(pos_tagging_text(row['sen1']), object),
                                    np.asarray(pos_tagging_text(row['sen2']), object),
                                    np.asarray(pos_tagging_text(row['sen3']), object),
                                    np.asarray(pos_tagging_text(row['sen4']), object)
                                    ]
            pos_end.loc[index] = [index, pos_tagging_text(row['sen5'])]
            story_number = story_number + 1

            if story_number % 1000 == 0:
                print("Processed ", story_number, "/", total_stories)

        print("Saving pos tagged corpus..")
        # Saving models in two data files
        cur_dir = os.path.splitext(train_set)[0]
        path_begin = cur_dir + "_pos_begin"
        path_end = cur_dir + "_pos_end"

        np.save(path_begin, pos_begin)
        np.save(path_end, pos_end)
        print("Saved the pos tagged corpus successfully !")

        # To load dataset, then do np.load(train_pos_begin)


    else:
        pos_begin = pd.DataFrame(columns=['id', 'sen'])
        for index, row in data_original.iterrows():
            pos_begin.loc[index] = [index,
                                    np.concatenate((np.asarray(pos_tagging_text(row['sen1']), object),
                                                    np.asarray(pos_tagging_text(row['sen2']), object),
                                                    np.asarray(pos_tagging_text(row['sen3']), object),
                                                    np.asarray(pos_tagging_text(row['sen4']), object)
                                                    ))]
            pos_end.loc[index] = [index, pos_tagging_text(row['sen5'])]
            story_number = story_number + 1

            if story_number % 1000 == 0:
                print("Processed ", story_number, "/", total_stories)

        pos_begin = np.asarray(pos_begin)
        pos_end = np.asarray(pos_end)

        print("Saving pos tagged corpus..")
        # Saving models in two data files
        cur_dir = os.path.splitext(train_set)[0]
        path_begin = cur_dir + "_pos_begin_together"
        path_end = cur_dir + "_pos_end_together"

        np.save(path_begin, pos_begin)
        np.save(path_end, pos_end)
        print("Saved the pos tagged corpus successfully !")


    return pos_begin, pos_end


def combine_matrix_cols(array):

    return combined_matrix


# TODO Remove if not used
def open_csv_asmatrix(datafile):
    print("Loading ", datafile)
    file_csv = pd.read_csv(datafile)
    file = np.asarray(file_csv)
    print("Loaded ",datafile, " successfully!")
    return file


# just trying if works
if __name__ == '__main__':
    # data_orig, data_proc = preprocess(train_set)
    # x_begin, x_end = get_story_matrices(data_proc)
    # print(x_begin)
    # print(x_end)
    # n_stories, *_ = x_begin.shape
    # x_begin = np.reshape(x_begin, (n_stories, -1))
    # print(x_begin.shape)

    # matrix = np.load(train_pos_begin)
    # for item in matrix[0:10,1]:
    #     print(item[:,1])
    # print(matrix[0,1])


    dataset=train_set
    pos_begin, pos_end = pos_tag_dataset(dataset, seperate=False)


    #----------

    # x = [["a", "b"], ["c", ]]
    #
    # result = sum(x, [])
    # print(result)
    #
    # matrix = np.load(train_pos_begin)
    #
    # print(matrix.shape)
    # matrix = matrix[0:5,1:5]
    #
    #
    # print(matrix.shape)
    # # print(matrix[:,0])
    # story = []
    #
    # new_matrix = []
    # array_corpus = []
    # # for i in range(0, matrix.shape[0]):
    # i=0
    # a = matrix[i,0]
    # b = matrix[i,1]
    # c = matrix[i,2]
    # d = matrix[i,3]
    # np.asarray(new_matrix.append(np.concatenate((a, b, c, d))))
    # print(np.asarray([np.concatenate((a,b,c,d))], object).shape)
    # print(new_matrix.append(np.concatenate((a,b,c,d))))
    # print((np.asarray(new_matrix.append(np.concatenate((a,b,c,d))))))
    # np.hstack((array_corpus, new_matrix))
    #
    # print(array_corpus)
    # # print(np.concatenate(a, b), axis=1)
    #
    # new_matrix = [1]
    #
    # for index in range(0,matrix.shape[0]):
    #     print(index)
    #     print(matrix[index, :].tolist().shape)
    #     np.vstack((new_matrix, matrix[index, :].tolist()))
    # print(new_matrix)


    #-------

    # print(np.append(matrix[0:1,1], matrix[0:1,2], matrix[0:1,3], matrix[0:1,4]).shape)

    # sentences = _load_data(train_set)
    # print(sentences)
    # pos_text = pos_tagging_text(sentences)
    # print(pos_text)

    # _, data_proc = preprocess(train_set, pad='sentence')
    # print(data_proc[0])
    # n_stories, *_ = data_proc.shape
    # data_proc = np.reshape(data_proc, (n_stories*sentence_len, -1))
    # word_embedding(data_proc)

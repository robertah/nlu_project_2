import nltk
import os
from data_utils import *


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
        cur_dir = os.path.splitext(dataset)[0]
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
        cur_dir = os.path.splitext(dataset)[0]
        path_begin = cur_dir + "_pos_begin_together"
        path_end = cur_dir + "_pos_end_together"

        np.save(path_begin, pos_begin)
        np.save(path_end, pos_end)
        print("Saved the pos tagged corpus successfully !")

    return pos_begin, pos_end


def preprocess(pos_begin, pos_end, pad='ending', punct=True, stop_words=True, lemm=True, test=False):
    """
    Preprocess pos-tagged data

    :param pos_begin: pos-tagged context sentences in train / val / test data
    :param pos_end: pos-tagged ending sentences in train / val / test data
    :param pad: 'ending' to pad the endings or None
    :param punct: True to remove punctuation from sentences
    :param stop_words: True to remove stop words from sentences
    :param lemm: True to lemmitize words in sentences
    :param test: False if train data, True if val / test data
    :return: processed pos-tagged beginning and ending data
    """

    # remove id from datasets
    pos_begin = pos_begin[:, 1]
    pos_end = pos_end[:, 1]

    # get number of stories
    n_stories = len(pos_begin)

    # clean words in sentences
    begin_processed = word_cleaning(pos_begin, punct, stop_words, lemm)
    end_processed = word_cleaning(pos_end, punct, stop_words, lemm)

    # generate vocabulary if training
    if not test:
        filtered = filter_words(begin_processed) + filter_words(end_processed)
        vocabulary = generate_vocabulary(filtered)
    # load vocabulary if testing
    else:
        vocabulary = load_vocabulary()

    # replace words not in vocab with unk
    begin_processed = check_for_unk(begin_processed, vocabulary)
    end_processed = check_for_unk(end_processed, vocabulary)

    # pad ending if needed
    if pad=='ending':
        begin_processed, end_processed = pad_endings(begin_processed, end_processed)

    # map words to vocabulary indexes
    for i in range(n_stories):
        begin_processed[i] = [get_indexes_from_words(sen, vocabulary) for sen in begin_processed[i]]
        end_processed[i] = [get_indexes_from_words(sen, vocabulary) for sen in end_processed[i]]

    return begin_processed, end_processed



def combine_matrix_cols(array):
    return combined_matrix


# TODO Remove if not used
def open_csv_asmatrix(datafile):
    print("Loading ", datafile)
    file_csv = pd.read_csv(datafile)
    file = np.asarray(file_csv)
    print("Loaded ", datafile, " successfully!")
    return file


# just trying if works
if __name__ == '__main__':
    # context, end,  preprocess(train_set, pad=None)

    # dataset=train_set
    # pos_begin, pos_end = pos_tag_dataset(dataset, seperate=False) OR
    pos_begin = np.load(data_folder + '/train_stories_pos_begin_together.npy')  # (88161, 2)
    pos_end = np.load(data_folder + '/train_stories_pos_end_together.npy')  # (88161, 2)
    pos_begin_processed, pos_end_processed = preprocess(pos_begin, pos_end)
    print(pos_begin_processed)
    print(pos_end_processed)
    beg, end = filter_words(pos_begin_processed), filter_words(pos_end_processed)
    print(beg)
    print(end)




    # ----------

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

    # -------

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

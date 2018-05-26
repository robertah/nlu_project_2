

# def tokenize(dataframe, stop_words, lemmatize, stem):
#     """
#     Tokenize sentences in the given dataframe
#
#     :param dataframe: dataframe containing train / val / test data
#     :param stop_words: True to remove stop words and punctuation
#     :param lemmatize: True to lemmatize words
#     :param stem: True to stem words
#     :return: dataframe with tokenized sentences
#     """
#
#     df = dataframe.copy()
#
#     # select columns containing sentences
#     sen_cols = [s for s in df if s.startswith('sen')]
#
#     # tokenize sentences and remove punctuation
#     tokenizer = RegexpTokenizer(r'\w+')
#     for col in sen_cols:
#         df[col] = df[col].str.lower()
#         df[col] = df[col].apply(tokenizer.tokenize)
#
#     # remove stop words
#     if stop_words:
#         download('stopwords')
#         stop = set(stopwords.words('english'))
#         for col in sen_cols:
#             df[col] = df[col].apply(lambda x: [y for y in x if y not in stop])
#
#     # lemmatize words  TODO need to add pos tagger to make it work
#     if lemmatize:
#         download('wordnet')
#         lemmatizer = wordnet.WordNetLemmatizer()
#         for col in sen_cols:
#             df[col] = df[col].apply(lambda x: [lemmatizer.lemmatize(y) for y in x])
#
#     # stem words
#     if stem:
#         stemmer = SnowballStemmer('english')
#         for col in sen_cols:
#             df[col] = df[col].apply(lambda x: [stemmer.stem(y) for y in x])
#
#     print("Sentences have been tokenized")
#
#     return df



# def preprocess_old(dataset, pad=None, stop_words=True, lemmatize=False, stem=False):
#     """
#     Preprocess raw train / val / test data files
#
#     :param dataset: path to dataset
#     :param pad: pad endings or none ('ending', None)
#     :param id: remove id from the dataset
#     :return: context, ending and answer (for val / test data)
#     """
#
#     assert dataset == train_set or dataset == val_set or dataset == test_set
#
#     print("Preprocessing...")
#
#     # load data from csv
#     data_original = load_data(dataset)
#
#     # tokenize sentences in dataframe
#     data_original = tokenize(data_original, stop_words, lemmatize, stem)
#
#     # generate vocabulary if training, otherwise load existing vocabulary
#     if dataset == train_set:
#         vocabulary = generate_vocabulary(data_original)
#     else:
#         vocabulary = load_vocabulary()
#
#     # filter columns by those containing sentences
#     sen_cols = [s for s in data_original if s.startswith('sen')]
#
#     # replace missing words with unk
#     for col in sen_cols:
#         for i, row in data_original.iterrows():
#             data_original.at[i, col] = check_for_unk(list(row[col]), vocabulary)
#
#     # get number of stories and number of sentences per story
#     n_stories = len(data_original)
#     n_sentences = len(sen_cols)
#
#     # get beginnings (first 4 sentences) and endings (last (two) sentence(s))
#     sentences = read_sentences(data_original)
#     beginnings, endings = get_context_ending(sentences)
#
#     # pad endings or none
#     if pad == 'ending':
#         beginnings, endings = pad_endings(beginnings, endings)
#
#     # index words in stories according to the vocabulary
#     for i in range(n_stories):
#         beginnings[i] = [get_indexes_from_words(sen, vocabulary) for sen in beginnings[i]]
#         endings[i] = [get_indexes_from_words(sen, vocabulary) for sen in endings[i]]
#
#     # return beginnings and endings for train set
#     if dataset == train_set:
#         return beginnings, endings
#
#     # return also answers for val / test set
#     else:
#         return beginnings, endings, get_answers(data_original)

def get_context_ending(data):
    """

    :param data:
    :return:
    """

    n_stories, n_sentences = data.shape

    b, e = data[:, :4], data[:, 4:]

    # squeeze endings matrix if it contains only one sentence
    if n_sentences == 5:
        e = [s[0] for s in e]
        # e = np.asarray(e)
        e = np.reshape(e, n_stories)

    print(e)

    return b, e

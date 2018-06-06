import keras
from keras.preprocessing import sequence
from keras.models import Model, Sequential
from keras.layers import LSTM, Embedding, Input, Merge, Dense
from keras.optimizers import Adadelta,SGD
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.callbacks import TensorBoard
import keras.backend as K
import numpy as np
import sys
sys.path.append("..")
from config import *
from data_utils import *
from preprocessing import *
import os
from numpy import array
from numpy import asarray
from numpy import zeros
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from copy import deepcopy
from negative_endings import *
import pickle



'''
https://stackoverflow.com/questions/46466013/siamese-network-with-lstm-for-sentence-similarity-in-keras-gives-periodically-th
'''

def embedding(docs):
    '''

    :param docs: array containing sentences
    :return:
    '''
    # prepare tokenizer
    t = Tokenizer()
    t.fit_on_texts(docs)
    vocab_size = len(t.word_index) + 1

    # integer encode the documents
    encoded_docs = t.texts_to_sequences(docs)
    print(encoded_docs)

    # pad documents to a max length of 4 words
    max_length = 45
    padded_docs = pad_sequences(encoded_docs, maxlen=max_length, padding='post')
    print(padded_docs)

    # load the whole embedding into memory
    embeddings_index = dict()
    f = open('glove.6B/glove.6B.100d.txt')
    for line in f:
        values = line.split()
        word = values[0]
        coefs = asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()
    print('Loaded %s word vectors.' % len(embeddings_index))
    # create a weight matrix for words in training docs
    embedding_matrix = zeros((vocab_size, 100))
    for word, i in t.word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    return embedding_matrix

#
# def create_base_network(feature_dim,seq_len):
#     '''
#     :param feature_dim:
#     :param seq_len: length of sentences
#     :return:
#     '''
#
#     embeddings_index = dict()
#     f = open('../glove_data/glove.6B/glove.6B.100d.txt')
#     for line in f:
#         values = line.split()
#         word = values[0]
#         coefs = asarray(values[1:], dtype='float32')
#         embeddings_index[word] = coefs
#     f.close()
#     print('Loaded %s word vectors.' % len(embeddings_index))
#     # create a weight matrix for words in training docs
#     embedding_matrix = zeros((vocabulary_size, 100))
#     for word, i in t.word_index.items():
#         embedding_vector = embeddings_index.get(word)
#         if embedding_vector is not None:
#             embedding_matrix[i] = embedding_vector
#
#     # define model
#     model = Sequential()
#     model.add(Embedding(inpu_dim=vocabulary_size, output_dim=embedding_dim, weights=[embedding_matrix],
#                   input_length=story_len, trainable=False))
#     # model.add(Embedding(input_dim = vocabulary_size, output_dim = embedding_dim, input_length = story_len))
#     print(model.summary())
#     model.add(LSTM(100, batch_input_shape=(None, seq_len, feature_dim),return_sequences=True))
#     # model.add(Dense(50, activation='relu'))
#     # model.add(Dense(10, activation='relu'))
#     # print(model.summary())
#     return model


# def siamese(feature_dim, seq_len, n_epoch, train_dataA, train_dataB, train_y, val_dataA, val_dataB, val_y):
def siamese(feature_dim, seq_len, n_epoch, train_data, val_data, embedding_docs):

    # def siamese(feature_dim, seq_len, n_epoch, train_generator, validation_generator):

    # train_data[:,0] =  train_data[:,0].reshape((len(train_data[:,0]), 1))
    # train_data[:,1] =  train_data[:,1].reshape((len(train_data[:,1]), 1))

    print('train_data[:,0].shape: {}'.format(train_data[:,0].shape))
    print('train_data[:,1].shape: {}'.format(train_data[:,1].shape))

    print('len(train_data[:,0]): {}'.format(len(train_data[:,0])))
    print('len(train_data[:,1]): {}'.format(len(train_data[:,1])))

    assert train_data[:,0].shape == train_data[:,1].shape
    assert len(train_data[:,0]) == len(train_data[:,1])



    # If no base_network
    # prepare tokenizer
    embedding_matrix = embedding(embedding_docs)

    # define model
    model = Sequential()
    model.add(Embedding(inpu_dim=vocabulary_size, output_dim=embedding_dim, weights=[embedding_matrix],
                        input_length=story_len, trainable=False))
    # model.add(Embedding(input_dim = vocabulary_size, output_dim = embedding_dim, input_length = story_len))
    model.add(LSTM(100, batch_input_shape=(None, seq_len, feature_dim), return_sequences=True))
    print(model.summary())

    input_a = Input(shape=(seq_len, ), dtype='int32')
    input_b = Input(shape=(seq_len, ), dtype='int32')

    processed_a = model(input_a)
    processed_b = model(input_b)


    # If using base_network
    # base_network = create_base_network(feature_dim, seq_len)

    # model = Sequential()
    # model.add(Embedding(inpu_dim=vocabulary_size, output_dim=embedding_dim, weights=[embedding_matrix],
    #                     input_length=story_len, trainable=False))
    # model.add(Embedding(input_dim = vocabulary_size, output_dim = embedding_dim, input_length = story_len))
    # model.add(LSTM(100, batch_input_shape=(None, seq_len, feature_dim), return_sequences=True))
    # print(model.summary())

    # input_a = Input(shape=(seq_len, ), dtype='int32')
    # input_b = Input(shape=(seq_len, ), dtype='int32')
    # processed_a = base_network(input_a)
    # processed_b = base_network(input_b)


    distance = keras.layers.Lambda(cosine_distance, output_shape=cosine_dist_output_shape)([processed_a, processed_b])

    model = Model([input_a, input_b], distance)

    adam_optimizer = keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

    model.compile(loss='mean_squared_error', optimizer=adam_optimizer, metrics=['accuracy'])
    model.summary()
    print(model.summary())


    # Fitting if not generator
    model.fit(x=[train_data[:,0].reshape((len(train_data[:,0]), 1)), train_data[:,1].reshape((len(train_data[:,0]), 1))], y=train_data[:,2],
              batch_size=1,
              epochs=n_epoch,
              validation_data=([val_data[:,0].reshape((len(val_data[:,0]), 1)), val_data[:,1].reshape((len(val_data[:,1]), 1))], val_data[:,2]))

    # Fitting if generator
    # out_trained_models = '../trained_models'
    #
    # lr_callback = keras.callbacks.ReduceLROnPlateau(monitor="acc",
    #                                                 factor=0.5,
    #                                                 patience=0.5,
    #                                                 verbose=0,
    #                                                 cooldown=0,
    #                                                 min_lr=0)
    # stop_callback = keras.callbacks.EarlyStopping(monitor="acc",
    #                                               min_delta=0.0001,
    #                                               patience=11,
    #                                               verbose=0,
    #                                               mode="auto")
    #
    # tensorboard_callback = keras.callbacks.TensorBoard(log_dir=out_trained_models, histogram_freq=0, batch_size=1,
    #                                                    write_graph=True,
    #                                                    write_grads=False, embeddings_freq=0,
    #                                                    embeddings_layer_names=None, embeddings_metadata=None)
    #
    # checkpoint_callback = keras.callbacks.ModelCheckpoint(
    #     os.path.join(out_trained_models, 'cnn_ngrams/weights.h5'),
    #     monitor='val_loss', verbose=0, save_best_only=True,
    #     save_weights_only=False, mode='auto', period=1)
    #
    # model.fit_generator(train_generator,
    #                     steps_per_epoch=1871,
    #                     verbose=2,
    #                     epochs=n_epoch,
    #                     callbacks=[lr_callback, stop_callback, tensorboard_callback, checkpoint_callback],
    #                     validation_data=validation_generator,
    #                     validation_steps=1871)

    pred = model.predict([train_dataA, train_dataB])
    for i in range(len(pred)):
        print (pred[i], train_y[i])

    return model


def exponent_neg_manhattan_distance(A, B):
    """
    Helper function used to estimate similarity between the LSTM outputs
    :param X:  output of LSTM with input X
    :param Y:  output of LSTM with input Y
    :return: Manhattan distance between input vectors
    """
    return K.exp(-K.sum(K.abs(A - B), axis=1, keepdims=True))


def cosine_distance(vecs):
    #I'm not sure about this function too
    y_true, y_pred = vecs
    y_true = K.l2_normalize(y_true, axis=-1)
    y_pred = K.l2_normalize(y_pred, axis=-1)
    return K.mean(1 - K.sum((y_true * y_pred), axis=-1))


def cosine_dist_output_shape(shapes):
    shape1, shape2 = shapes
    print((shape1[0], 1))
    return (shape1[0], 1)


''' FUNCTIONS TECHNICALLY FROM NEGATIVE_ENDINGS, DATA_UTILS OR TRAINING_UTILITIES'''

def padding(max_len, embedding):
    for i in range(len(embedding)):
        padding = np.zeros(max_len-embedding[i].shape[0])
        embedding[i] = np.concatenate((embedding[i], padding))

    embedding = np.array(embedding)
    return embedding


def initialize_negative_endings(contexts, endings):
    import negative_endings as data_aug #TODO added this for trials, but should remove
    neg_end = data_aug.Negative_endings(contexts = contexts, endings = endings)
    neg_end.load_vocabulary()
    #Preserve here in case the vocabulary change, do not save filters and reload them
    neg_end.filter_corpus_tags()

    return neg_end


def full_stories(contexts, endings, validation=False, list_array=False):

    context_batches = []
    ending_batches = []
    idx_batch_endings = 0

    for context in contexts:
        story_endings = endings[idx_batch_endings]

        context_batch = []
        ending_batch = []
        for ending_word in story_endings:

            if list_array:
                original_context = deepcopy(context[0])
            else:
                original_context = deepcopy(context)

            # if len(list(original_context) + list(ending_word)) != 45:
            #     print("Found wrong len of story: ", len(original_context + ending_word))

            # print(original_context+ending)
            context_batch.append(original_context)
            ending_batch.append(ending_word)

        context_batches.append(context_batch)
        ending_batches.append(ending_batch)
        idx_batch_endings = idx_batch_endings + 1
        if idx_batch_endings % 20000 == 0:
            print("Stories combined together ", idx_batch_endings, "/", len(contexts))

    return np.asarray(context_batches), np.asarray(ending_batches)


def eliminate_tags_corpus(corpus_pos_tagged):
    '''
    Removes pos tagging from corpus
    :param corpus_pos_tagged:
    :return: corpus as a list of words without pos tagging, for each sentence
    '''
    corpus_no_tag = []
    for batch_pos_tagged in corpus_pos_tagged:
        batch_no_tag = []
        for sentence_pos_tagged in batch_pos_tagged:

            batch_no_tag.append([word_tag[0] for word_tag in sentence_pos_tagged])
        #print(batch_endings_no_tag)
        corpus_no_tag.append(batch_no_tag)

    return corpus_no_tag


def load_vocabulary():

    # self.vocabulary = data_utils.load_vocabulary()

    with open(full_vocabulary_pkl, 'rb') as handle:
        vocabulary = pickle.load(handle)
    print("Vocabulary loaded")
    print("Vocabulary saved into negative ending object")


def get_words_from_indexes(indexes, vocabulary, pos_vocabulary=None):
    """
    Get words from indexes in the vocabulary

    :param indexes: list of indexes of words in vocabulary
    :param vocabulary: vocabulary
    :return: words corresponding to given indexes
    """

    # map indexes to words in vocabulary
    vocabulary_reverse = {v: k for k, v in vocabulary.items()}
    if pos_vocabulary is not None:
        pos_vocabulary_reverse = {v: k for k, v in pos_vocabulary.items()}


    # retrieve words corresponding to indexes
    if isinstance(indexes, list) or isinstance(indexes, np.ndarray):
        if isinstance(indexes[0], tuple):
            words = [(vocabulary_reverse[x[0]], pos_vocabulary_reverse[x[1]]) for x in indexes]
        else:
            words = [vocabulary_reverse[x] for x in indexes]
    else:
        if isinstance(indexes, tuple):
            words = (vocabulary_reverse[indexes[0]], pos_vocabulary_reverse[indexes[1]])
        else:
            words = vocabulary_reverse[indexes]
    return words


def get_sentences_from_indices(sentence_vocab_indices):

    with open(full_vocabulary_pkl, 'rb') as handle:
        vocabulary = pickle.load(handle)
    print("Vocabulary loaded")
    print("Vocabulary saved into negative ending object")

    sentence = data_utils.get_words_from_indexes(indexes=sentence_vocab_indices, vocabulary=vocabulary)
    # print(sentence)

    return sentence


def story_into_character_sentences(self, story_vocab_indices):
    story_sentences = []

    for sentence_vocab_indices in story_vocab_indices:
        story_sentences.append(self.get_sentences_from_indices(sentence_vocab_indices=sentence_vocab_indices))

    return story_sentences


def story_into_character_sentences(story_vocab_indices):

    story_sentences = []

    for sentence_vocab_indices in story_vocab_indices:
        story_sentences.append(get_sentences_from_indices(sentence_vocab_indices=sentence_vocab_indices))

    return story_sentences


def dataset_into_character_sentences(dataset):

    all_stories = []
    #story_number = 0
    for story in dataset:
        all_stories.append(story_into_character_sentences(story_vocab_indices=story))
        #story_number = story_number+1
        #print(story_number)

    print("Done -> Dataset into character sentences")
    return all_stories
    #print(all_stories)


def combine_sentences(sentences):
    """
    Combine multiple sentences in one sentence

    :param sentences: array of sentences
    :return: combines sentence
    """

    # get number of stories
    n_stories, *_ = sentences.shape
    print(sentences.shape)
    # combine sentences
    combined = np.empty(n_stories, dtype=list)
    for i in range(n_stories):
        combined[i] = []
        combined[i].extend([sentences[i]])

    print(combined, "\n\n\n\n\n")
    return combined


def batch_iter_val(contexts, endings, neg_end_obj, binary_verifiers, out_tagged_story = False,
                       batch_size = 2, num_epochs = 500, shuffle=True):
    """
    Generates a batch generator for the validation set.
    """
    if not out_tagged_story:
        contexts = eliminate_tags_corpus(corpus_pos_tagged = contexts)
        endings = eliminate_tags_corpus(corpus_pos_tagged = endings)

    while True:
        context_batches, ending_batches = full_stories(contexts = contexts, endings = endings, validation = True)
        total_steps = len(context_batches)

        # for batch_idx in range(0, total_steps):
        #     #batch_size stories -> 1 positive endings + batch_size-1 negative endings ones
        #     context_batch = context_batches[batch_idx]
        #     ending_batch = ending_batches[batch_idx]
        #     binary_batch_verifier = [[int(ver), 1-int(ver)] for ver in binary_verifiers[batch_idx]]
        #      yield (np.asarray(context_batch), np.asarray(ending_batch), np.asarray(binary_batch_verifier))
    return np.asarray(context_batch), np.asarray(ending_batch), np.asarray(binary_batch_verifier)


#For this function the datast needs to be pos tagged
def batches_backwards_neg_endings(neg_end_obj, endings, batch_size, contexts):

    total_stories = len(endings)
    aug_data = []
    ver_aug_data = []
    for story_idx in range(0, total_stories):

        batch_aug_stories, ver_aug_stories = neg_end_obj.backwards_words_substitution_approach(context_story = contexts[story_idx], ending_story = endings[story_idx], batch_size = batch_size)

        if story_idx%20000 ==0:
            print("Negative ending(s) created for : ",story_idx, "/",total_stories)

        aug_data.append(batch_aug_stories)
        ver_aug_data.append(ver_aug_stories)

    neg_end_obj.no_samp = 0
    return aug_data, ver_aug_data


def batches_pos_neg_endings(neg_end_obj, endings, batch_size):
    """INPUT:
             neg_end_obj : Needs the negative endings objects created beforehand
             endings : dataset
             batch_size : batch_size - 1 negative endings will be created
        """
    total_stories = len(endings)
    aug_data = []
    ver_aug_data = []
    for story_idx in range(0, total_stories):

        batch_aug_stories, ver_aug_stories = neg_end_obj.words_substitution_approach(ending_story = endings[story_idx], batch_size = batch_size,
                                                                                         out_tagged_story = False, shuffle_batch = True, debug=False)
        if story_idx%20000 ==0:
            print("Negative ending(s) created for : ",story_idx, "/",total_stories)
        aug_data.append(batch_aug_stories)
        ver_aug_data.append(ver_aug_stories)

    neg_end_obj.no_samp = 0
    return aug_data, ver_aug_data


def batch_iter_backward_train_cnn(contexts, endings, neg_end_obj, out_tagged_story = False,
                                  batch_size = 2, num_epochs = 500, shuffle=True, training_data=True):
    """
    Generates a batch generator for the train set.
    """
    if not out_tagged_story:
        contexts = eliminate_tags_corpus(corpus_pos_tagged = contexts)

    if training_data:
    #for i in range(0,num_epochs):
        print("Augmenting with negative endings for the next epoch -> stochastic approach..")
        batch_endings, ver_batch_end = batches_backwards_neg_endings(neg_end_obj = neg_end_obj, endings = endings,
                                                                     batch_size = batch_size, contexts = contexts)

        context_batches, ending_batches = full_stories(contexts=contexts, endings=batch_endings, validation=True)
        total_steps = len(context_batches)
        print("Train generator for the new epoch ready..")

        for batch_idx in range(0, total_steps):
                # batch_size stories -> 1 positive endings + batch_size-1 negative endings ones
                context_batch = context_batches[batch_idx]
                ending_batch = ending_batches[batch_idx]
                verifier_batch = [[int(ver), 1 - int(ver)] for ver in ver_batch_end[batch_idx]]
                # if batch_idx==(total_steps-1):
                #     print('Last Context batch: {}'.format(context_batch))
                #     print('Last Ending batch: {}'.format(ending_batch))
                #     print('Last Verifier batch: {}'.format(verifier_batch))
                # yield (np.asarray([np.asarray(context_batch), np.asarray(ending_batch)]), np.asarray(verifier_batch))
        print('Shape Context batch: {}'.format(context_batches.shape))
        print('Shape Ending batch: {}'.format(ending_batches.shape))
        print('Shape Verifier batch: {}'.format(np.asarray(ver_batch_end).shape))
        return np.asarray(context_batches), np.asarray(ending_batches), np.asarray(ver_batch_end)

    #     batches_full_stories = full_stories(contexts = contexts, endings = batch_endings)
    #     total_steps = len(batches_full_stories)
    #     print("Train generator for the new epoch ready..")
    #
    #     for batch_idx in range(0, total_steps):
    #         #batch_size stories -> 1 positive endings + batch_size-1 negative endings ones
    #
    #         stories_batch = batches_full_stories[batch_idx]
    #         verifier_batch = [[int(ver), 1-int(ver)] for ver in ver_batch_end[batch_idx]]
    #         yield (np.asarray(stories_batch), np.asarray(verifier_batch))
    # # return (np.asarray(stories_batch), np.asarray(verifier_batch))

if __name__ == '__main__':
    import tensorflow as tf
    import keras

    print(tf.__version__)
    print(keras.__version__) # Make sure version of keras is 2.1.4

    # For Testing, reducing training set to 5 stories
    sample_size = 5

    print("Initializing negative endings...")
    pos_train_begin, pos_train_end, pos_val_begin, pos_val_end = load_train_val_datasets_pos_tagged(together=False)
    pos_train_begin = pos_train_begin[:sample_size]
    pos_train_end = pos_train_end[:sample_size]
    pos_val_begin = pos_val_begin
    pos_val_end = pos_val_end
    pos_train_begin_tog, pos_train_end_tog, pos_val_begin_tog, pos_val_end_tog = load_train_val_datasets_pos_tagged(together=True)
    pos_train_end_tog = pos_train_end_tog[:sample_size]
    pos_train_begin_tog = pos_train_begin_tog[:sample_size]
    pos_val_begin_tog = pos_val_begin_tog
    pos_val_end_tog = pos_val_end_tog

    neg_end = initialize_negative_endings(contexts=pos_train_begin_tog, endings=pos_train_end_tog)

    train_context_notag = np.array(eliminate_tags_corpus(pos_train_begin_tog))

    print(train_context_notag)
    print(train_context_notag.shape)

    val_context_notag = np.array(eliminate_tags_corpus(pos_val_begin_tog))

    print(val_context_notag)
    print(val_context_notag.shape)

    print("pos train begin.shape: {}".format(pos_train_begin.shape))
    print("eliminated pos tag: {}".format(len(eliminate_tags_corpus(pos_val_end))))

    ver_val_set = generate_binary_verifiers(val_set)
    print("neg_ending.shape: {}".format(type(neg_end)))

    n_endings=3

    aug_data, ver_aug_data = batches_pos_neg_endings(neg_end_obj=neg_end,
                                                     endings=pos_train_end,
                                                     batch_size=n_endings)

    # Creating input arrays with 3 columns(context, padded ending and verifier) -> padding of size 43, one hot encoding
    count=0
    train_structured_stories=[]
    for story_ending in aug_data:
        for i in range(0,n_endings):
            print('i: {}'.format(i))
            print(count)
            train_structured_stories.append([train_context_notag[count], story_ending[i], ver_aug_data[count][i]])
        count+=1
    train_structured_stories= np.array(train_structured_stories)
    print(train_structured_stories.shape)

    count=0
    val_structured_stories=[]
    for i in range(0,len(pos_val_begin)):
        print('i: {}'.format(i))
        val_structured_stories.append([val_context_notag[i], pos_val_end[i], ver_val_set[i]])
    val_structured_stories= np.array(val_structured_stories)
    print(val_structured_stories.shape)


    n_epoch = 1
    epochs=n_epoch

    print('Getting words from indices...')
    print(dataset_into_character_sentences(train_structured_stories[:,0]))
    print(np.array(dataset_into_character_sentences(train_structured_stories[:,0])).shape)
    # print('Combined train set: {}'.format(combine_sentences(np.asarray(load_data(train_set)))))
    # print('Combined train set: {}'.format(combine_sentences(np.asarray(load_data(train_set))).shape))


    model = siamese(feature_dim=feature_dim, seq_len=43, n_epoch = n_epoch,
                    train_data=train_structured_stories,
                    val_data = val_structured_stories,
                    embedding_docs = np.array(dataset_into_character_sentences(train_structured_stories[:,0]))
                    )
    # Todo: problem with input array size and embedding doc
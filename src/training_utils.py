import numpy as np
import random


def word_2_vec(model_w2v, batch):
    embedded_sentences = []
    # Embedding is performed per rows (sentences) so we need to transpose

    for sentence in batch:
        embedded_sentences.append(model_w2v[sentence])

    return np.asarray(embedded_sentences)


def create_batch(batch_size, model_w2v, dataset, dataset_size):
    idx_sentences = np.random.choice(dataset_size, batch_size)
    print("Indexes of sentences selected ", idx_sentences)
    batch = []

    for idx in idx_sentences:
        batch.append(dataset[idx])

    # batch=dataset[:,idx_sentences]
    batch = word_2_vec(model_w2v, batch)

    return batch


def create_batches(nb_batches, batch_size, model_w2v, dataset, dataset_size):
    """It returns a 4-dimensional numpy array ready for training.
       The 1-st dimension represents the batch (multiple batches) -> 100 batches
       The 2-nd dimension represents the single word dimension -> 100 dimensions/word
       The 3-rd dimension represents the entire sentence -> 30 word/sentence
       The 4-th dimension represents the number of sentences per batch -> 64 sentences/batch
    """
    batches = []

    print("Creating batches, totally ", nb_batches)

    """Creating a single batch each time could be expensive, 
       whereas ceating multiple ones in one go could be less computationally expensive"""

    if nb_batches == 1:
        batches = (np.transpose(create_batch(batch_size, model_w2v, dataset, dataset_size)))
    else:
        for i in range(0, nb_batches):
            batches.append(np.transpose(create_batch(batch_size, model_w2v, dataset, dataset_size)))

    print("Batches created")
    return np.array(batches)


def words_mapper_to_vocab_indices(x_batch, vocabulary_words_list):
    """Map words in x_batch to the corresponding vocabulary index
       returns x_batch with indices in place of the original words"""

    #print(x_batch)
    nb_sentences = len(x_batch)
    nb_words = len(x_batch[0])
    #print(nb_words)
    print
    for idx_sentence in range(0, nb_sentences):

        for idx_word in range(0, nb_words):
            x_batch[idx_sentence][idx_word] = vocabulary_words_list.index(x_batch[idx_sentence][idx_word])

    return x_batch


def words_mapper_from_vocab_indices(vocab_indices, vocabulary_words_list, is_tuple=False):
    """Map words in x_batch to the corresponding vocabulary index
       returns x_batch with indices in place of the original words"""

    if is_tuple:
        vocab_indices = list(vocab_indices)

    #vocab_indices = list(vocab_indices)

    nb_words_vocabulary_indices = len(vocab_indices)
    # print("NB words ", nb_words_vocabulary_indices)
    # print(list(indices_predictions))
    # nb_words_vocabulary_indices = len(indices_predictions[0])

    # for idx_sentence in range(0,nb_sentences):
    words_of_sentences = []
    vocab_size = len(vocabulary_words_list)
    for idx_word in vocab_indices:
        words_of_sentences.append(vocabulary_words_list[idx_word])

    return words_of_sentences


def batch_iter(data, batch_size, num_epochs, shuffle=False, testing=False):
    """
    Generates a batch generator for a dataset.
    shuffle should stay false because on local pc we get memory error and the PC gets stucked
    """

    batches = []
    # data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data) - 1) / batch_size) + 1

    word_in_sentence=len(data[0])

    for epoch in range(num_epochs):

        # Shuffle the data at each epoch
        if shuffle:
            random.shuffle(data)
            shuffled_data = data
        else:
            shuffled_data = data

        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            if end_index-start_index == batch_size:

                yield zip(shuffled_data[start_index:end_index], np.copy(shuffled_data[start_index:end_index]))
                

# This is used for the training only, because we want to input x and y batches correctly for precitions
def batch_iter_train(data, batch_size, num_epochs, shuffle=False, testing=False):
    """
    Generates a batch generator for a dataset.
    shuffle should stay false because on local pc we get memory error and the PC gets stucked
    """

    batches = []
    data_size = len(data)
    num_batches_per_epoch = int((len(data) - 1) / batch_size) + 1

    words_in_sentence=len(data[0])

    for epoch in range(num_epochs):

        # Shuffle the data at each epoch
        if shuffle:
            random.shuffle(data)
            shuffled_data = data
        else:
            shuffled_data = data

        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)

            if end_index-start_index == batch_size:

                x_batch = shuffled_data[start_index:end_index]
                y_batch = x_batch[:]

                for sentence_idx in range (batch_size):
                    x_batch[sentence_idx] = x_batch[sentence_idx][0:words_in_sentence-1]


                    y_batch[sentence_idx] = y_batch[sentence_idx][1:words_in_sentence]
                

                yield zip(x_batch, y_batch)


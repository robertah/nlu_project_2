import numpy as np
import random


def words_mapper_to_vocab_indices(vocab_indices, vocabulary_words_list):
    """Map words in x_batch to the corresponding vocabulary index
       returns x_batch with indices in place of the original words"""

    nb_words = len(vocab_indices)

    for idx_word in range(0, nb_words):
        vocab_indices[idx_word] = vocabulary_words_list.index(vocab_indices[idx_word])

    vocab_words = vocab_indices
    
    return vocab_words


def words_mapper_from_vocab_indices(vocab_indices, vocabulary_words_list, is_tuple=False):
    """Map words in x_batch to the corresponding vocabulary index
       returns x_batch with indices in place of the original words"""

    if is_tuple:
        vocab_indices = list(vocab_indices)

    nb_words_vocabulary_indices = len(vocab_indices)

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

    return
                

# This is used for the training only, because we want to input x and y batches correctly for precitions
def batch_iter_train(data, batch_size, num_epochs, shuffle=False, testing=False):
    """
    Generates a batch generator for a dataset.
    shuffle should stay false because on local pc we get memory error and the PC gets stucked
    """

    return

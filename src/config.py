""""
File containing configuration variables used across the project
"""
import os

# number of the group for the project
n_group = '20'

# path to data folder and data sets
data_folder = '../data'
out_trained_models = '../trained_models'
#TODO to be specified the dataset path
#train_set = data_folder + '/sentences.train'
#eval_set = data_folder + '/sentences.eval'
#test_set = data_folder + '/sentences_test'

# token used for the language model
bos = '<bos>'  # begin of sentence token
eos = '<eos>'  # end of sentence token
pad = '<pad>'  # padding token
unk = '<unk>'  # unknown token

# TODO to be specified
vocabulary_pkl = 'vocabulary.pkl'
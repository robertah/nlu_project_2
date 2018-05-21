""""
File containing configuration variables used across the project
"""

# number of the group for the project
n_group = '20'

# path to data folder and data sets
data_folder = '../data'
out_trained_models = '../trained_models'
# TODO to be specified the dataset path
train_set = data_folder + '/train_stories.csv'
val_set = data_folder + '/cloze_test_val__spring2016 - cloze_test_ALL_val.csv'
test_set = data_folder + ''  # TODO add test set path when we have it

# token used for the language model
# bos = '<bos>'  # begin of sentence token
# eos = '<eos>'  # end of sentence token
pad = '<pad>'  # padding token
unk = '<unk>'  # unknown token

# TODO to be specified
vocabulary_pkl = data_folder + '/vocabulary.pkl'
# train_pos_begin = data_folder + '/train_pos_begin.pkl'
# train_pos_end = data_folder + '/train_pos_end.pkl'
train_pos_begin = data_folder + '/train_stories_pos_begin.csv'
train_pos_end = data_folder + '/train_stories_pos_end.csv'

vocabulary_size = 20000  # None for not limited vocabulary size
sentence_len = 15  # None for not limited sentence length

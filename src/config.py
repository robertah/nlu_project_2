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
test_set = data_folder + '/test_set.csv'  # TODO add test set path when we have it


#Ready to be used for preprocessing -> do not touch
train_pos_context_tog = data_folder + '/train_stories_pos_begin_together.npy'
train_pos_end_tog = data_folder + '/train_stories_pos_end_together.npy'
val_pos_context_tog = data_folder + '/cloze_test_val__spring2016 - cloze_test_ALL_val_pos_begin_together.npy'
val_pos_end_tog = data_folder +  '/cloze_test_val__spring2016 - cloze_test_ALL_val_pos_end_together.npy'
test_pos_begin_tog = data_folder + '/test_set_pos_begin_together.npy'
test_pos_end_tog = data_folder + '/test_set_pos_end_together.npy'
# token used for the language model
# bos = '<bos>'  # begin of sentence token
# eos = '<eos>'  # end of sentence token
pad = '<pad>'  # padding token
unk = '<unk>'  # unknown token

# TODO to be specified
vocabulary_pkl = data_folder + '/vocabulary.pkl'
pos_vocabulary_pkl = data_folder + '/pos_vocabulary.pkl'
full_vocabulary_pkl = data_folder + '/full_vocabulary.pkl'
sentiment_pkl = data_folder + '/sentiment.pkl'
train_pos_begin = data_folder + '/train_stories_pos_begin.npy'
train_pos_end = data_folder + '/train_stories_pos_end.npy'
val_pos_begin = data_folder + '/cloze_test_val__spring2016 - cloze_test_ALL_val_pos_begin.npy'
val_pos_end = data_folder + '/cloze_test_val__spring2016 - cloze_test_ALL_val_pos_end.npy'

test_pos_begin = data_folder + '/test_set_pos_begin.npy'
test_pos_end = data_folder + '/test_set_pos_end.npy'

word_embedding = data_folder + '/word_embedding'


vocabulary_size = 25000  # None for not limited vocabulary size
sentence_len = 10  # None for not limited sentence length     max len 12 in train, 11 in val
# story_len = 80  # in the train set the max story len is 74
story_len = 70      #### 43

embedding_dim = 150
num_steps = 10 #used for the lstm embedding layer (# of steps/words in each sample)
hidden_size = 100

#Add other tags if you want negative endings to probabilistically sample from them
#Remeber to add the corresponding sampling probability in the correct order
tags_to_sample_from = ["VB", "VBD", "VBG", "VBN", "VBP", "VBZ", #verbs
                       "NN", "NNP", "NNPS", "NNS", #nouns
                       "PRP", #pronouns
                       "RB", "RBR", "RBS", # adverbs
                       "JJ", "JJR", "JJS"] # adjectives
probs_tags_to_sample_from = [0.8, 0.8, 0.8, 0.8, 0.8, 0.8, #verbs
                             0.8, 0.8, 0.8, 0.8, #nouns
                             0.8, #pronouns
                             0.8, 0.8, 0.8, # adverbs
                             0.8, 0.8, 0.8] # adjectives

#For Siamese LSTM
n_hidden = 50
gradient_clipping_norm = 1.25
batch_size = 64
n_epoch = 25
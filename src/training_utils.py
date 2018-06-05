import numpy as np
from data_utils import *
import negative_endings as neg_end
from copy import deepcopy
from sentiment import *
from preprocessing import *
from config import *


"""THIS PART UNTIL THE BATCH ITERS SHOULD E MOVED SOMEWHERE ELSE"""

def aggregate_contexts(contexts):
    contexts_aggregated = []
    for context in contexts:
        context_aggregated = []

        for sentence in context:
            context_aggregated = context_aggregated + sentence

        #print("CHECK PERFORM context_aggregated is a list ",context_aggregated)
        contexts_aggregated.append(context_aggregated)

    return contexts_aggregated


def full_stories_together(contexts, endings, contexts_aggregated = True, validation = False, list_array = False):

    if not contexts_aggregated:
        contexts = aggregate_contexts(contexts)

    full_stories_batches = []
    #print("ENDING")
    #print(len(endings))
    #print(len(endings[0]))
    #print("CONTEXT")
    #print(len(contexts[0]))


    idx_batch_endings = 0

    for context in contexts:
        story_endings = endings[idx_batch_endings]

        full_story_batch = []
        for ending in story_endings:
            #print("\n Ending is\n ",ending)
            if list_array:
                original_context = deepcopy(context[0])
            else:
                original_context = deepcopy(context)
            #print("\n Original_context is\n ", original_context)
            #print(len(ending))
            full_story = list(original_context) + list(ending)
            #print("UNIQUE is ", full_story)
            lenght = len(full_story)
            if lenght > story_len:
                print("Found wrong len of story: ", lenght)
                full_story = full_story[0:story_len]
            else:
                full_story = full_story


            #print(original_context+ending)
            full_story_batch.append(full_story)
        
        full_stories_batches.append(full_story_batch)
        idx_batch_endings = idx_batch_endings + 1
        if idx_batch_endings%20000 == 0:
            print("Stories combined together ",idx_batch_endings,"/",len(contexts))

    return full_stories_batches



#For this function the datast needs to be pos tagged
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

def eliminate_tags_in_contexts(contexts_pos_tagged):


    contexts_no_tag = []
    """print("CHECK PERFORM: len contexts")
    print(len(contexts_pos_tagged))
    print(len(contexts_pos_tagged[0]))
    print(len(contexts_pos_tagged[0][0]))"""

    for context in contexts_pos_tagged:
        context_no_tag = []
        for sentence in context:
            context_no_tag.append([word_tag[0] for word_tag in sentence])
        contexts_no_tag.append(context_no_tag)
    return contexts_no_tag

def eliminate_tags_in_val_endings(endings_pos_tagged):



    endings_no_tag = []
    for endings_batch_pos_tagged in endings_pos_tagged:
        batch_endings_no_tag = []
        for ending_pos_tagged in endings_batch_pos_tagged:

            batch_endings_no_tag.append([word_tag[0] for word_tag in ending_pos_tagged])
        #print(batch_endings_no_tag)
        endings_no_tag.append(batch_endings_no_tag)

    return endings_no_tag

"""UNTIL HERE, MAYBE MOVE IN PREPROCESSING IMPORTING * such that nothing has to be modified"""


"""********************************** CNN TRAIN - VALIDATION SETS GENERATOR *****************************"""

def batch_iter_val_cnn(contexts, endings, neg_end_obj, binary_verifiers, out_tagged_story = False,
                       batch_size = 2, num_epochs = 500, shuffle=True):
    """
    Generates a batch generator for the validation set.
    """
    if not out_tagged_story:
        contexts = eliminate_tags_in_contexts(contexts_pos_tagged = contexts)
        endings = eliminate_tags_in_val_endings(endings_pos_tagged = endings)

    while True:

        batches_full_stories = full_stories_together(contexts = contexts, endings = endings, list_array = True)

        #batches_full_stories = full_stories_together(contexts = contexts, endings = endings, validation = True)

        total_steps = len(batches_full_stories)

        for batch_idx in range(0, total_steps):
            #batch_size stories -> 1 positive endings + batch_size-1 negative endings ones
            stories_batch = batches_full_stories[batch_idx]
            binary_batch_verifier = [[int(ver), 1-int(ver)] for ver in binary_verifiers[batch_idx]]
            yield (np.asarray(stories_batch), np.asarray(binary_batch_verifier))

def batch_iter_train_cnn(contexts, endings, neg_end_obj, out_tagged_story = False,
                         batch_size = 2, num_epochs = 500, shuffle=True, test = False):
    """
    Generates a batch generator for the train set.
    """
    if not out_tagged_story:
        contexts = eliminate_tags_in_contexts(contexts_pos_tagged= contexts)
    while True:
    #for i in range(0,num_epochs):
        print("Augmenting with negative endings for the next epoch -> stochastic approach..")
        batch_endings, ver_batch_end= batches_pos_neg_endings(neg_end_obj = neg_end_obj, endings = endings,
                                                              batch_size = batch_size)
        if not test:
            batches_full_stories = full_stories_together(contexts = contexts, endings = batch_endings)
        else:
            batches_full_stories = full_stories_together(contexts = contexts, endings = batch_endings)

        total_steps = len(batches_full_stories)
        print("Train generator for the new epoch ready..")

        for batch_idx in range(0, total_steps):
            #batch_size stories -> 1 positive endings + batch_size-1 negative endings ones

            stories_batch = batches_full_stories[batch_idx]
            verifier_batch = [[int(ver), 1-int(ver)] for ver in ver_batch_end[batch_idx]]
            yield (np.asarray(stories_batch), np.asarray(verifier_batch))

# --------- For Siamese LSTM --------

def batch_iter_train_SiameseLSTM(contexts, endings, neg_end_obj, out_tagged_story = False,
                         batch_size = 2, num_epochs = 500, shuffle=True):
    '''
    Generates a batch generator for the train set.
    Same idea as batch_iter_train_cnn function except that returns context, ending and verifier separately
    :param contexts: array of all context (separate or together...?) #TODO check which one applies
    :param endings: array of all endings
    :param neg_end_obj: array with all negative endings
    :param out_tagged_story:
    :param batch_size:
    :param num_epochs:
    :param shuffle:
    :return:
    '''

    if not out_tagged_story:
        contexts = eliminate_tags_in_contexts(contexts_pos_tagged= contexts)
    while True:
    #for i in range(0,num_epochs):
        print("Augmenting with negative endings for the next epoch -> stochastic approach..")
        batch_endings, ver_batch_end= batches_pos_neg_endings(neg_end_obj = neg_end_obj, data = endings,
                                                              batch_size = batch_size)
        batches_full_stories = np.array([contexts, batch_endings])  #full_stories_together(contexts = contexts, endings = batch_endings)
        print("Shape of batches_full_stories: {}".format(batches_full_stories).shape)
        print("Shape should be {} times {}".format(len(context), 2))

        print("Length context: {}".format(len(contexts)))
        total_steps = len(contexts)
        print("Train generator for the new epoch ready..")

        for batch_idx in range(0, total_steps):
            #batch_size stories -> 1 positive endings + batch_size-1 negative endings ones

            stories_batch = batches_full_stories[batch_idx] # TODO Change batches_full_stories
            verifier_batch = [[int(ver), 1-int(ver)] for ver in ver_batch_end[batch_idx]]
            yield (np.asarray(stories_batch), np.asarray(verifier_batch))


def batch_iter_val_SiameseLSTM(contexts, endings, neg_end_obj, binary_verifiers, out_tagged_story = False,
                       batch_size = 2, num_epochs = 500, shuffle=True):
    """
    Generates a batch generator for the validation set.
    """
    if not out_tagged_story:
        contexts = eliminate_tags_in_contexts(contexts_pos_tagged = contexts)
        endings = eliminate_tags_in_val_endings(endings_pos_tagged = endings)

    while True:

        batches_full_stories = np.array([contexts, batch_endings])#full_stories_together(contexts = contexts, endings = endings, validation = True)

        total_steps = len(batches_full_stories)

        for batch_idx in range(0, total_steps):
            #batch_size stories -> 1 positive endings + batch_size-1 negative endings ones
            stories_batch = batches_full_stories[batch_idx]
            binary_batch_verifier = [[int(ver), 1-int(ver)] for ver in binary_verifiers[batch_idx]]
            yield (np.asarray(stories_batch), np.asarray(binary_batch_verifier))

def batch_iter_backward_train_cnn(contexts, endings, neg_end_obj, out_tagged_story = False,
                                  batch_size = 2, num_epochs = 500, shuffle=True):
    """
    Generates a batch generator for the train set.
    """
    if not out_tagged_story:
        contexts_no_tag = eliminate_tags_in_contexts(contexts_pos_tagged = contexts)

    while True:
    #for i in range(0,num_epochs):
        print("Augmenting with negative endings for the next epoch -> stochastic approach..")
        batch_endings, ver_batch_end = batches_backwards_neg_endings(neg_end_obj = neg_end_obj, endings = endings,
                                                                     batch_size = batch_size, contexts = contexts)
        batches_full_stories = full_stories_together(contexts = contexts_no_tag, endings = batch_endings, contexts_aggregated = False)
        total_steps = len(batches_full_stories)
        print("Train generator for the new epoch ready..")

        for batch_idx in range(0, total_steps):
            #batch_size stories -> 1 positive endings + batch_size-1 negative endings ones

            stories_batch = batches_full_stories[batch_idx]
            verifier_batch = [[int(ver), 1-int(ver)] for ver in ver_batch_end[batch_idx]]
            yield (np.asarray(stories_batch), np.asarray(verifier_batch))




"""***************************CNN LSTM sentiment********************"""


def batch_iter_val_cnn_sentiment(contexts, endings, binary_verifiers):
    """
    Generates a batch generator for the validation set.
    """
    contexts = eliminate_tags_in_contexts(contexts_pos_tagged = contexts)
    endings = no_tags_in_val_endings(endings_pos_tagged = endings)
    print("LEN ENDINGS ",len(endings))
    #print("\n\nCONTEXT & ENDINGS\n\n")
    #print(binary_verifiers)
    #print(contexts[0])
    #print(endings[0])
    context_sentiments = sentences_to_sentiments(contexts = contexts)
    endings_sentiments = endings_to_sentiments(endings = endings)

    while True:

        batches_full_stories = full_stories_together(contexts = context_sentiments, endings = endings_sentiments)#, list_array = True)

        total_steps = len(batches_full_stories)

        for batch_idx in range(0, total_steps):
            #batch_size stories -> 1 positive endings + batch_size-1 negative endings ones
            stories_batch = batches_full_stories[batch_idx]
            binary_batch_verifier = [[int(ver), 1-int(ver)] for ver in binary_verifiers[batch_idx]]
            # print(np.asarray(stories_batch), np.asarray(binary_batch_verifier))
            yield (np.asarray(stories_batch), np.asarray(binary_batch_verifier))


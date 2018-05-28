import numpy as np
import random
import data_utils
import negative_endings as neg_end
from copy import deepcopy


def batch_iter(data, batch_size, num_epochs, shuffle=False):
    """
    Generates a batch generator for the dataset.
    """

    return
                


def full_stories_together(contexts, endings):

    full_stories_batches = []

    idx_batch_endings = 0
    for context in contexts:
        #endings = len()
        #print(endings[0])
        print("")
        print("NEW CONTEXT GOT")
        #print(endings[0][0])
        #print(endings[0][0][0])
        #print(endings[0][0][0][0])
        #print("END")
        story_endings = endings[idx_batch_endings]
        #for batch_endings in endings[idx_batch_endings]:
        idx_ending = 0
        full_story_batch = []
        for ending in story_endings:
            ending = ending[0]
            #print("Context")
            #print(context)
            #print("Ending")
            #print(ending)
            print("CONTEXT LEN")
            print(len(context[0]))
            print("ENDING LEN")
            #print(ending)
            print(len(ending))
            print("ALL STORY TOGETHER LEN ")
            #print(context[0]+ending)

            #print(len(ending[0]))
            #print(len(ending[0][0]))
            #print("END analysis")
            original_context = deepcopy(context[0])
            print(len(original_context))
            #l = original_context[0] + ending
            #print("CONTEXT")
            #print(original_context)
            print("IDX ending is: ", idx_ending)
            #print(l)
            #print("len(l) ", len(l))
            #print("l[0] : ",l[0])
            #print("l[1] : ",l[1])
            #print("len(l[1]) : ",len(l[1]))
            idx_ending = idx_ending + 1 
            #print(len(original_context + ending))
            #if l != :
            #    print()
            full_story_batch.append(original_context + ending)
        
        full_stories_batches.append(full_story_batch)
        idx_batch_endings = idx_batch_endings + 1
    #print("Full stories in input: ", full_stories_batch)
    return full_stories_batch

#For this function the datast needs to be pos tagged
def batches_pos_neg_endings(neg_end_obj, data, batch_size):
    """INPUT:
             neg_aug_obj : Needs the negative endings objects created beforehand
             data : dataset
             batch_size : batch_size - 1 negative endings will be created 
             total_stories : total stories in the dataset
             is_w2v : if the input is already in a w2v form
        """
    total_stories = len(data)
    aug_data = []
    ver_aug_data = []
    for story_idx in range(0, total_stories):
        #print("Before words sub")
        #print(data[story_idx])
        batch_aug_stories, ver_aug_stories = neg_end_obj.words_substitution_approach(ending_story = data[story_idx],
                                                                                     batch_size = batch_size,
                                                                                     out_tagged_story = True,
                                                                                     shuffle_batch = True)
        #print("After words sub")
        if story_idx%20000 ==0:
            print("Negative ending(s) created for : ",story_idx, "/",total_stories)
            #print(aug_data)

        aug_data.append(batch_aug_stories)
        ver_aug_data.append(ver_aug_stories)
    print("OUT")
    print("Same endings batches : ",neg_end_obj.no_samp)
    return aug_data, ver_aug_data


"""********************************** CNN TRAIN - VALIDATION SETS GENERATOR *****************************"""

def batch_iter_val_cnn(contexts, endings, neg_end_obj, binary_verifiers,
                       batch_size = 2, num_epochs = 5000, shuffle=True):
    """
    Generates a batch generator for the validation set.
    """

    for i in range(0,num_epochs):

        batches_full_stories = full_stories_together(contexts = contexts, endings = endings)
        
        print("LEN data ", len(batches_full_stories))

        total_steps = len(batches_full_stories)
        print("Validation generator for the new epoch ready..")

        for batch_idx in range(0, total_steps):
            #batch_size stories -> 1 positive endings + batch_size-1 negative endings ones
            yield zip(batches_full_stories[batch_idx], binary_verifiers[batch_idx])

def batch_iter_train_cnn(contexts, endings, neg_end_obj, 
                         batch_size = 2, num_epochs = 5000, shuffle=True):
    """
    Generates a batch generator for the train set.
    """

    for i in range(0,num_epochs):
        print("Augmenting with negative endings for the next epoch -> stochastic approach..")
        batch_endings, ver_batch_end= batches_pos_neg_endings(neg_end_obj = neg_end_obj, data = endings,
                                                              batch_size = batch_size)

        batches_full_stories = full_stories_together(contexts = contexts, endings = batch_endings)
        
        print("LEN data ", len(batches_full_stories))
        #total_steps = len(aug_data)/2
        
        total_steps = len(batches_full_stories)
        print("Train generator for the new epoch ready..")

        for batch_idx in range(0, total_steps):
            #batch_size stories -> 1 positive endings + batch_size-1 negative endings ones
            yield zip(batches_full_stories[batch_idx], ver_batch_end[batch_idx])

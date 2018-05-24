import numpy as np
import random
import data_utils
import negative_endings as neg_end


def batch_iter(data, batch_size, num_epochs, shuffle=False):
    """
    Generates a batch generator for the dataset.
    """

    return
                



#For this function the datast needs to be pos tagged
def batches_pos_neg_endings(neg_aug_obj, data, total_stories, is_w2v, batch_size = 2):
    """INPUT:
             neg_aug_obj : Needs the negative endings objects created beforehand
             data : dataset
             batch_size : batch_size - 1 negative endings will be created 
             total_stories : total stories in the dataset
             is_w2v : if the input is already in a w2v form
        """
    aug_data = []
    ver_aug_data = []
    for story_idx in range(0, total_stories):
        
        batch_aug_stories, ver_aug_stories = neg_aug_obj.words_substitution_approach(ending_story = data[story_idx], #Can be full story or just id & final sentence [[id,final_sentence],[id2,....]..] 
                                                                                     is_w2v = is_w2v, #If the story is w2v already and tags are in a numerical form as well
                                                                                     batch_size = batch_size,
                                                                                     shuffle_batch = True)
        if story_idx%20000 ==0:
            print("Negative ending(s) created for : ",story_idx, "/",total_stories)

        aug_data.append(batch_aug_stories)
        ver_aug_data.append(ver_aug_stories)

    return aug_data, ver_aug_data


def batch_iter_train_cnn(data, neg_aug_obj, is_w2v, 
                         batch_size = 2, num_epochs = 5000, shuffle=True):
    """
    Generates a batch generator for the dataset.
    """
    total_stories = len(data) #total stories

    for i in range(0,num_epochs):
        print("Augmenting the data for the next epoch -> stochastic approach..")
        aug_data, ver_aug_data= batches_pos_neg_endings(neg_aug_obj = neg_aug_obj, data = data,
                                                        is_w2v = is_w2v, batch_size = batch_size, total_stories = total_stories)
        total_steps = len(aug_data)/2

        print("Generator for the new epoch ready..")

        for batch_stories_idx in range(0, total_steps):
            #two stories -> positive endings + negative endings ones
            yield zip(aug_data[batch_stories_idx], ver_aug_data[batch_stories_idx])
            #yield zip(data[2*j : 2*j+1],data[2*j : 2*j+2])

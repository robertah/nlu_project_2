import preprocessing as prep
import training_utils as train_utils
from config import *
import random
import pickle
import nltk
import numpy as np
import os

class data_augmentation:
    
    def __init__(self, thr_new_noun, thr_new_verb, thr_new_adj, thr_new_adv):

        self.name = "data"
        self.set_sample_probabilities(thr_sample_new_noun = thr_new_noun, thr_sample_new_verb = thr_new_verb, 
                                 thr_sample_new_adj = thr_new_adj, thr_sample_new_adv = thr_new_adv)



    def set_sample_probabilities(self, thr_sample_new_noun, thr_sample_new_verb, thr_sample_new_adj, thr_sample_new_adv):
        """
            Sample probabilites thresholds for different logical part of the sentence
            NB: keep the thr_sample_new_noun high (0.9) -> nouns are very important in semantics
        """
        self.thr_sample_new_verb = thr_sample_new_verb
        self.thr_sample_new_adj = thr_sample_new_adj
        self.thr_sample_new_noun = thr_sample_new_noun
        self.thr_sample_new_adv = thr_sample_new_adv
    
    def augment_data_batch(self, dataset, batch_size, training_story):
    
        """
        Input:
        training_sentence: single full story (5 sentences)
        batch_size: desired stories augmentation (future input to the model)
                    e.g batch_size = 10 -> original training story + 9 constructed stories

        Output:
       
        1) 3d array batch_aug_stories -> [batch_size, len(training_story), 2]
        2) 1d array ver_aug_stories -> [batch_size]

        The only modified story sentence is the fifth one!

        batch_aug_stories: Augmented training_stories with 
                                original training_story + (batch_size-1) augmented_training_stories. 
                                augmented training stories differ AT LEAST with some grammatical component.

        ver_aug_stories: contains 
                                    1 for the SINGLE correct story 
                                    all 0s for the incorrect stories
       
        After the augmentation the order of the stories in the batch is shuffled (with the corresponding verifier).

        Nothing is stored in the objective fileds !
        A stories batch is just returned
        """


        return batch_aug_stories, ver_aug_stories

    def change_sentence(self, sentence):

        """Check tags. If there is a noun, verb, adverb, adjective:
        1) Sample a random number from [0,1]
        2) Compare with threshold
        3) Substitute the word if sampled number > threshold with a random dataset word belonging to the same tag
        Note that each sentence has words of the type ("I", "Subj") so to substitute just the word it should be taken position 0 of the index
        """

        index=0
    
        for word_tag in sentence:

            if word_tag[1] == "" : #Verb
            
                p = random.uniform(0, 1)

                if p > self.thr_sample_new_verb:
                    sentence[index][0] = sample_from_verbs()  

            if word_tag[1] == "" : #Noun
            
                p = random.uniform(0, 1)

                if p > self.thr_sample_new_noun :
                    sentence[index][0] = sample_from_nouns()

            if word_tag[1] == "" : #Adj
            
                p = random.uniform(0, 1)

                if p > self.thr_sample_new_adj:
                    sentence[index][0] = sample_from_adjectives()  

            if word_tag[1] == "" : #Adverb
            
                p = random.uniform(0, 1)

                if p > self.thr_sample_new_adv:
                    sentence[index][0] = sample_from_adverbs()  



    def sample_from_nouns(self):
        """
            Output: sample a noun from the all the nouns
            of the dataset
        """

        return noun


    def sample_from_verbs(self):
        """
            Output: sample a verb from the all the verbs
            of the dataset
        """

        return verb


    def sample_from_adverbs(self):
        """
            Output: sample an adverb from the all the adverbs
            of the dataset
        """

        return adv

    def sample_from_adjectives(self):
        """
            Output: sample an adjective from the all the adjectives
            of the dataset
        """

        return adj



    def divide_by_tags(self):
        """Input:
           dataset
       
           Output:
           Save in different object fields (1d arrays):
           1) All the distinct verbs of the dataset
           2) All the distinct adjectives of the dataset
           3) All the distinct nouns of the dataset
           4) All the distinct adverbs of the dataset
 
           This because these arrays will be used to create this online augmentation during training
        """
        all_stories_pos_tagged = []
        story_number = 0
        for story in self.all_stories:
            #TODO to be contunued
            all_stories_pos_tagged.append(self.pos_tagger_story(story = story))
            story_number = story_number + 1
            print(story_number)
        print(all_stories_pos_tagged)
        print("Done -> Dataset into pos tagged dataset")
        self.all_stories_pos_tagged = all_stories_pos_tagged


        return


    """******************FROM STORIES TO POS TAGGED STORIES & SAVE TO FILE*****************"""
    
    def save_pos_tagged_story(self, tagged_story):
        #TODO to be completed
        for tagged_sent in tagged_story:
            text = " ".join(w+"/"+t for w,t in tagged_sent)



    def save_pos_tagged_corpus():
        #TODO to be completed
        outfile = open(os.path.join(data_folder,"tagged_corpus.txt"))
        for tagged_story in self.all_stories_pos_tagged:
            outfile.write(self.save_pos_tagged_story(tagged_story = tagged_story)+"\n")

    
    def pos_tagger_sentence(self, sentence):

        sentence_pos_tagged = nltk.pos_tag(sentence)

        return sentence_pos_tagged

    def pos_tagger_story(self, story):

        story_pos_tagged = []

        for sentence in story:

            story_pos_tagged.append(self.pos_tagger_sentence(sentence = sentence))

        return story_pos_tagged



    def pos_tagger_dataset(self):

        all_stories_pos_tagged = []
        story_number = 0
        for story in self.all_stories:

            all_stories_pos_tagged.append(self.pos_tagger_story(story = story))
            story_number = story_number + 1
            print(story_number)
        print(all_stories_pos_tagged)
        print("Done -> Dataset into pos tagged dataset")
        self.all_stories_pos_tagged = all_stories_pos_tagged




    """******************FROM VOCABULARY INDICES DATASET TO CHARACTER DATASET*****************"""


    def load_vocabulary(self):
        
        with open(vocabulary_pkl, 'rb') as v:
            self.vocabulary = pickle.load(v)
        #print(self.vocabulary)
        self.vocabulary_list = list(self.vocabulary)


    def get_sentences_from_indices(self, sentence_vocab_indices):


        sentence = train_utils.words_mapper_from_vocab_indices(sentence_vocab_indices, self.vocabulary_list)
        #print(sentence)
        
        return sentence

    def story_into_character_sentences(self, story_vocab_indices):

        story_sentences = []

        for sentence_vocab_indices in story_vocab_indices:
            story_sentences.append(self.get_sentences_from_indices(sentence_vocab_indices=sentence_vocab_indices))

        return story_sentences

    def dataset_into_character_sentences(self, dataset):

        all_stories = []
        story_number = 0
        for story in dataset:
            all_stories.append(self.story_into_character_sentences(story_vocab_indices=story))
            story_number = story_number+1
            #print(story_number)

        print("Done -> Dataset into character sentences")
        self.all_stories = all_stories
        #print(all_stories)



def main():

    d_orig, d_prep = prep.preprocess(train_set)
    x_begin, x_end = prep.get_story_matrices(d_prep)
    
    x_begin = x_begin.tolist()
    x_end = x_end.tolist()

    data_aug = data_augmentation(0.5,0.5,0.5,0.5)
    data_aug.load_vocabulary()

    data_aug.dataset_into_character_sentences(x_begin)
    data_aug.pos_tagger_dataset()

main()
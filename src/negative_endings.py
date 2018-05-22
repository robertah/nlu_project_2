import preprocessing as prep
import training_utils as train_utils
from config import *
import random
import pickle
from collections import Counter
import nltk
import numpy as np
import os
from random import randint
from random import shuffle

class negative_endings:

    """For reference on basic augmentation in negative endings and get inspiration from
       please see the paper An RNN-based Binary Classifier for the Story Cloze Test
       """

    def __init__(self, thr_new_noun, thr_new_pronoun, thr_new_verb, thr_new_adj, thr_new_adv):

        self.name = "data"
        self.set_sample_probabilities(thr_sample_new_noun = thr_new_noun, thr_sample_new_pronoun = thr_new_pronoun,
                                      thr_sample_new_verb = thr_new_verb, 
                                 thr_sample_new_adj = thr_new_adj, thr_sample_new_adv = thr_new_adv)


    
    def set_sample_probabilities(self, thr_sample_new_noun, thr_sample_new_pronoun, thr_sample_new_verb, thr_sample_new_adj, thr_sample_new_adv):
        """
            Sample probabilities thresholds for different logical part of the sentence
            NB: keep the thr_sample_new_noun high (0.9) -> nouns are very important in semantics
        """
        self.thr_sample_new_verb = thr_sample_new_verb
        self.thr_sample_new_adj = thr_sample_new_adj
        self.thr_sample_new_noun = thr_sample_new_noun
        self.thr_sample_new_pronoun = thr_sample_new_pronoun
        self.thr_sample_new_adv = thr_sample_new_adv




    """******************USER FUNCTIONS: THESE FUNCTIONS ARE THE ONE TO USE FOR TRAINING*****************"""


    def augment_data_batch(self, full_training_story, is_tagged_story = False,
                           out_tagged_story = False, #Output a pos_tagged story if True
                           words_substitution_approach = True, #Replace probbilstically tagged words in the 5th sentence with same tagged words from the entire corpus
                           Random_approach = False, #Replace 5th sentence with a random one from the corpus endings
                           Backward_approach = False, #Replace 5th sentence with one random of the context
                           batch_size = 2,
                           shuffle_batch = True):
    
        """
        Input:
        training_sentence: single full story (5 sentences)
        batch_size: desired story endings augmentation (future input to the model)
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


        #NB: Only words_substitution_approach is under implementation now!

        if not is_tagged_story:
            pos_tagged_story = self.pos_tagger_story(story = full_training_story)
        else:
            pos_tagged_story = full_training_story


        batch_aug_stories = []


        batch_aug_stories.append(self.join_story_from_sentences(pos_tagged_story))


        if words_substitution_approach:

            ver_aug_stories = np.zeros(batch_size)
            ver_aug_stories[0] = 1


            if batch_size == 2:

                changed_story_ending = self.change_sentence(pos_tagged_story[-1])
                new_story = list(pos_tagged_story)
                new_story[-1] = changed_story_ending
                
                batch_aug_stories.append(self.join_story_from_sentences(new_story))
                

            else:
                for i in range(batch_size-1):
                    changed_story_ending = self.change_sentence(pos_tagged_story[-1])
                    new_story = list(pos_tagged_story)
                    new_story[-1] = changed_story_ending
                    
                    batch_aug_stories.append(self.join_story_from_sentences(new_story))
        
        if shuffle_batch:
            #Shuffle data preserving order stories - verifier
            shuffled_idx = shuffle(np.arange(batch_size))
            batch_aug_stories = batch_aug_stories[shuffled_idx]
            ver_aug_stories = ver_aug_stories[shuffled_idx]

        return batch_aug_stories, ver_aug_stories




    """******************************END USER FUNCTIONS**************************"""

    def join_story_from_sentences(self, story_sentences):

        """Join together the different sentences of the story into a unique array"""
        
        #NB not the best and efficient way to do that -> please change if u have more efficient algorithm
        joined_story = []
        for sentence in story_sentences:
            joined_story.extend(sentence)

        return joined_story


    def change_sentence(self, sentence):

        """Check tags. If there is a noun, verb, adverb, adjective:
        1) Sample a random number from [0,1]
        2) Compare with threshold
        3) Substitute the word if sampled number > threshold with a random dataset word belonging to the same tag

        nltk.help.upenn_tagset() -> displays the different tags and meaning
        VB, VBD, VBG, VBN, VBP, VBZ grouped together as Verbs
        NN, NNP, NNPS, NNS grouped together as nouns
        PRP grouped together as pronouns
        RB, RBR, RBS grouped together as adverbs
        JJ, JJR, JJS grouped together as adjectives

        Note that each sentence has words of the type ("I", "Subj") so to substitute just the word it should be taken position 0 of the index
        
        """

        index=0
    
        for word_tag in sentence:

            if "VB" in tagged_word[1] and tagged_word[0]!=pad: #Verbs
            
                p = random.uniform(0, 1)

                if p > self.thr_sample_new_verb:
                    sentence[index][0] = sample_from_verbs()  

            if "NN" in tagged_word[1] and tagged_word[0]!=pad: #Nouns
            
                p = random.uniform(0, 1)

                if p > self.thr_sample_new_noun :
                    sentence[index][0] = sample_from_nouns()

            if "PRP" in tagged_word[1] and tagged_word[0]!=pad: #Pronouns
            
                p = random.uniform(0, 1)

                if p > self.thr_sample_new_pronoun :
                    sentence[index][0] = sample_from_pronouns()

            if "JJ" in tagged_word[1] and tagged_word[0]!=pad: #Adjs
            
                p = random.uniform(0, 1)

                if p > self.thr_sample_new_adj:
                    sentence[index][0] = sample_from_adjectives()  

            if "RB" in tagged_word[1] and tagged_word[0]!=pad: #Advs
            
                p = random.uniform(0, 1)

                if p > self.thr_sample_new_adv:
                    sentence[index][0] = sample_from_adverbs()  



    def sample_from_nouns(self):
        """
            Output: sample a noun from the all the nouns
            of the dataset
        """

        return self.dict_corpus_nouns[randint(0,self.total_corpus_nouns)]
    
    def sample_from_pronouns(self):
        """
            Output: sample a pronoun from the all the pronouns
            of the dataset
        """

        return self.dict_corpus_pronouns[randint(0,self.total_corpus_pronouns)]


    def sample_from_verbs(self):
        """
            Output: sample a verb from the all the verbs
            of the dataset
        """

        return self.dict_corpus_verbs[randint(0,self.total_corpus_verbs)]


    def sample_from_adverbs(self):
        """
            Output: sample an adverb from the all the adverbs
            of the dataset
        """

        return self.dict_corpus_advs[randint(0,self.total_corpus_advs)]

    def sample_from_adjectives(self):
        """
            Output: sample an adjective from the all the adjectives
            of the dataset
        """

        return self.dict_corpus_adjs[randint(0,self.total_corpus_adjs)]



    """******************GROUPING TAGS PER TYPE TO FORM SETS TO SAMPLE FROM*****************"""

    
    def define_vocab_tags(self, all_corpus_nouns, all_corpus_pronouns, all_corpus_verbs,
                               all_corpus_advs, all_corpus_adjs):
        
        self.dict_corpus_nouns = list(Counter(all_corpus_nouns))
        self.total_corpus_nouns = len(self.dict_corpus_nouns)
        print(self.dict_corpus_nouns)

        self.dict_corpus_pronouns = list(Counter(all_corpus_pronouns))
        self.total_corpus_pronouns = len(self.dict_corpus_pronouns)
        print(self.dict_corpus_pronouns)

        self.dict_corpus_verbs = list(Counter(all_corpus_verbs))
        self.total_corpus_verbs = len(self.dict_corpus_verbs)
        print(self.dict_corpus_verbs)

        self.dict_corpus_advs = list(Counter(all_corpus_advs))
        self.total_corpus_advs = len(self.dict_corpus_advs) 
        
        self.dict_corpus_adjs = list(Counter(all_corpus_adjs))
        self.total_corpus_adjs = len(self.dict_corpus_adjs)



    def filter_story_tags(self, tagged_story):

        """Find different tags at :https://www.nltk.org/_modules/nltk/tag/mapping.html
           nltk.help.upenn_tagset() -> displays the different tags and meaning
           VB, VBD, VBG, VBN, VBP, VBZ grouped together as Verbs
           NN, NNP, NNPS, NNS grouped together as nouns
           PRP grouped together as pronouns
           RB, RBR, RBS grouped together as adverbs
           JJ, JJR, JJS grouped together as adjectives
           """
        
        all_nouns = []
        all_pronouns = []
        all_verbs = []
        all_advs = []
        all_adjs = []

        for tagged_sent in tagged_story:

            for tagged_word in tagged_sent:

                print("tagged word ", tagged_word)
                if "NN" in tagged_word and tagged_word!=pad: #Nouns
                    all_nouns.extend(tagged_word)
                if "PRP" in tagged_word and tagged_word!=pad: #Nouns
                    all_pronouns.extend(tagged_word)
                if "VB" in tagged_word and tagged_word!=pad: #Verbs
                    all_verbs.extend(tagged_word)
                if "RB" in tagged_word and tagged_word!=pad: #Advs
                    all_advs.extend(tagged_word)
                if "JJ" in tagged_word and tagged_word!=pad: #Adjs
                    all_adjs.extend(tagged_word)

        return all_nouns, all_pronouns, all_verbs, all_advs, all_adjs



    def filter_corpus_tags(self):
        """Input:
           dataset
       
           Output:
           Save in different object fields (1d arrays):
           1) All the distinct verbs of the dataset
           2) All the distinct adjectives of the dataset
           3) All the distinct nouns of the dataset
           4) All the distinct pronouns of the dataset
           5) All the distinct adverbs of the dataset
 
           This because these arrays will be used to create this online augmentation during training
        """
        all_corpus_nouns = []
        all_corpus_pronouns = []
        all_corpus_verbs = []
        all_corpus_advs = []
        all_corpus_adjs = []


        story_number = 0

        for tagged_story in self.all_stories_pos_tagged:
            all_story_nouns, all_story_pronouns, all_story_verbs, all_story_advs, all_story_adjs = self.filter_story_tags(tagged_story = tagged_story)
            
            all_corpus_nouns.extend(all_story_nouns)
            all_corpus_pronouns.extend(all_story_pronouns)
            all_corpus_verbs.extend(all_story_verbs)
            all_corpus_advs.extend(all_story_advs)
            all_corpus_adjs.extend(all_story_adjs)

            story_number = story_number + 1
            if story_number % 10000 == 0:
                print("Filtering: ",story_number)

        print("All corpus nouns")
        print(all_story_nouns)
        print(all_corpus_pronouns)
        self.define_vocab_tags(all_corpus_nouns = all_corpus_nouns, 
                               all_corpus_pronouns = all_corpus_pronouns,
                               all_corpus_verbs = all_corpus_verbs,
                               all_corpus_advs = all_corpus_advs, 
                               all_corpus_adjs = all_corpus_adjs)

        print("Done -> filtered corpus by tags")


        return


    """******************FROM CORPUS TO POS TAGGED CORPUS & SAVE TO FILE*****************"""


    
    def load_pos_tagged_corpus(self):
        #TODO to be completed Nini doing that?

        return

    def save_pos_tagged_story(self, tagged_story):

        #TODO to be completed Nini doing that?
        for tagged_sent in tagged_story:
            text = " ".join(w+"/"+t for w,t in tagged_sent)



    def save_pos_tagged_corpus(self):

        #TODO to be completed Nini doing that ?
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
            if story_number % 1000 == 0:
                print(story_number)
        #print(all_stories_pos_tagged)
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
        #story_number = 0
        for story in dataset:
            all_stories.append(self.story_into_character_sentences(story_vocab_indices=story))
            #story_number = story_number+1
            #print(story_number)

        print("Done -> Dataset into character sentences")
        self.all_stories = all_stories
        #print(all_stories)



def main():

    d_orig, d_prep = prep.preprocess(train_set)
    x_begin, x_end = prep.get_story_matrices(d_prep)
    
    x_begin = x_begin.tolist()
    x_end = x_end.tolist()

    neg_end = negative_endings(0.5, 0.5, 0.5,0.5,0.5)
    neg_end.load_vocabulary()

    neg_end.dataset_into_character_sentences(x_begin)
    neg_end.pos_tagger_dataset()
    neg_end.filter_corpus_tags()

main()
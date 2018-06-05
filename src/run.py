#!/usr/bin/env python3 -W ignore::DeprecationWarning

import argparse
import logging
import os
import sys
import datetime
import time
import training_utils as train_utils
import negative_endings as data_aug
import numpy as np
import keras

from models import cnn_ngrams
from models import cnn_lstm_sent

from config import *
from preprocessing import *
from data_utils import *
from training_utils import *

# Remove tensorflow CPU instruction information.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def _setup_argparser():
    """Sets up the argument parser and returns the arguments.

    Returns:
        argparse.Namespace: The command line arguments.
    """
    parser = argparse.ArgumentParser(description="Control program to launch all actions related to this project.")

    parser.add_argument("-m", "--model", action="store",
                        choices=["cnn_ngrams", "cnn_lstm", "put_your_model_name_here3", "put_your_model_name_here4"],
                        default="cnn_ngrams",
                        type=str,
                        help="the model to be used, defaults to cnn_ngrams")
    parser.add_argument("-t", "--train",
                        help="train the given model",
                        action="store_true")
    parser.add_argument("-p", "--predict",
                        help="predict on a test set given the model",
                        action="store_true")


    args, unknown = parser.parse_known_args()

    return args


def get_latest_model():
    """Returns the latest directory of the model specified in the arguments.

    Returns:
        (path) a path to the directory.
    """
    print("Retrieving trained model from {}".format(os.path.join(os.path.dirname(os.path.abspath(__file__)), out_trained_models)))
    if not os.path.exists(os.path.join(out_trained_models, args.model)):
        print("No trained model {} exists.".format(args.model))
        sys.exit(1)
    #Path to the model
    res = os.path.join(os.path.abspath("run.py"), "../trained_models", args.model)
    all_runs = [os.path.join(res, o) for o in os.listdir(res) if os.path.isdir(os.path.join(res, o))]
    res = max(all_runs, key=os.path.getmtime)

    return res


def get_submission_filename():
    """
    Returns:
        (path to directory) + filename of the submission file.
    """
    ts = int(time.time())
    submission_filename = "submission_" + str(args.model) + "_" + str(ts) + ".csv"
    submission_path_filename = os.path.join(get_latest_model(), submission_filename)

    return submission_path_filename

def initialize_negative_endings(contexts, endings):
    neg_end = data_aug.Negative_endings(contexts = contexts, endings = endings)
    neg_end.load_vocabulary()
    #Preserve here in case the vocabulary change, do not save filters and reload them
    neg_end.filter_corpus_tags()

    return neg_end

"""********************************************** USER ACTIONS from parser ************************************************************"""

if __name__ == "__main__":

    __file__ = "run.py"
    file_path = os.path.dirname(os.path.abspath(__file__))

    args = _setup_argparser()
    out_trained_models = ""
    #Once we get ready we can decomment. This avoids creating files when things have still to be debugged
    if args.train:
        out_trained_models = os.path.normpath(os.path.join(file_path,
                                              "../trained_models/",
                                              args.model,
                                              datetime.datetime.now().strftime(r"%Y-%m-%d[%Hh%M]")))
        try:
            os.makedirs(out_trained_models)
        except OSError:
            pass
    else:
        out_trained_models = os.path.normpath("..")
    
    print("Trained model will be saved in ", out_trained_models)
    if args.train:
        """Create a field with your model (see the default one to be customized) and put the procedure to follow to train it"""
        if args.model == "cnn_ngrams":
            
            #indices = [3749,47,424,196,65, 52,731]
            #vocab = load_vocabulary()
            #print("SENTECe is ",get_words_from_indexes(indexes = indices,vocabulary=vocab))
            

            #TOGETHER THE DATASET
            print("CNN grams training invoked")
            print("Loading dataset..")
            pos_train_begin_tog, pos_train_end_tog, pos_val_begin_tog, pos_val_end_tog = load_train_val_datasets_pos_tagged()
            ver_val_set = generate_binary_verifiers(val_set)
            print("Initializing negative endings..")
            neg_end = initialize_negative_endings(contexts = pos_train_begin_tog, endings = pos_train_end_tog)
            
            print("Loading validation set together..")
            pos_test_begin_tog, pos_test_end_tog = preprocess(pos_begin = np.load(test_pos_begin_tog), pos_end = np.load(test_pos_end_tog), test=True, pad='ending', punct=True,
                                                              stop_words=True, lemm=False)
            
            print("LENS")
            print(pos_test_begin_tog[0][0])
            print(len(pos_test_begin_tog[0][0]))
            print(len(pos_test_end_tog[0][0]))
            
            #Construct data generators
            ver_test_set = generate_binary_verifiers(test_set)

            #train_generator = train_utils.batch_iter_train_cnn(contexts = pos_train_begin_tog, endings = pos_train_end_tog, neg_end_obj = neg_end,
            #                                                   batch_size = 2, num_epochs = 500, shuffle=True)
            validation_generator = train_utils.batch_iter_val_cnn(contexts = pos_val_begin_tog, endings = pos_val_end_tog, binary_verifiers = ver_val_set, 
                                                                  neg_end_obj = neg_end, batch_size = 2, num_epochs = 500, shuffle=True)
            test_generator = train_utils.batch_iter_val_cnn(contexts = pos_test_begin_tog, endings = pos_test_end_tog, binary_verifiers = ver_test_set, 
                                                                  neg_end_obj = neg_end, batch_size = 2, num_epochs = 500, shuffle=True)
            #Initialize model
            #model = cnn_ngrams.CNN_ngrams(train_generator = validation_generator, validation_generator = test_generator)
            model = full_nn.CNN_ngrams(train_generator = validation_generator, validation_generator = test_generator)
            model.train(save_path = out_trained_models)
            """
            #SEPARATE SENTENCES DATASET
            print("Loading dataset..")
            pos_train_begin, pos_train_end, pos_val_begin, pos_val_end = load_train_val_datasets_pos_tagged(together = False)
            print("Initializing negative endings..")
            #Needed for negative endings this data load together
            pos_train_begin_tog, pos_train_end_tog, pos_val_begin_tog, pos_val_end_tog = load_train_val_datasets_pos_tagged()
            ver_val_set = generate_binary_verifiers()

            neg_end = initialize_negative_endings(contexts = pos_train_begin_tog, endings = pos_train_end_tog)
            
            #Construct data generators
            #print(pos_train_begin)
            #print(pos_train_end[0])
            #print(pos_val_begin[0])
            #print(pos_val_end[0])

            train_generator = train_utils.batch_iter_backward_train_cnn(contexts = pos_train_begin, endings = pos_train_end, neg_end_obj = neg_end,
                                                                        batch_size = 2, num_epochs = 500, shuffle=True)
            validation_generator = train_utils.batch_iter_val_cnn(contexts = pos_val_begin_tog, endings = pos_val_end_tog, binary_verifiers = ver_val_set, 
                                                                  neg_end_obj = neg_end, batch_size = 2, num_epochs = 500, shuffle=True)
            """
            #Initialize model
            #model = cnn_ngrams.CNN_ngrams(train_generator = validation_generator, validation_generator = validation_generator)
            #model.train()

            #print("TRAINING STORIES")
            #for batch in train_generator:
                #stories_train, verif_train = zip(*batch)
                #print(len(stories_train))
                #print(len(stories_train[0]))
                #print(verif_train)
            #print("EVALUATION STORIES")
            #for batch in validation_generator:
                #stories_train, verif_train = zip(*batch)
                #print(len(stories_train))
                #print(len(stories_train[0]))
                #print(verif_train)

        elif args.model == "cnn_lstm":

            print("Please put your procedure in here before running & remember to add the name of the model into the options of the parser!")
            contexts_val = np.load(val_pos_begin)
            endings_val = np.load(val_pos_end)

            contexts_val = eliminate_id(dataset = contexts_val)
            endings_val = eliminate_id(dataset = endings_val)

            contexts_test = np.load(test_pos_begin)
            endings_test = np.load(test_pos_end)

            contexts_test = eliminate_id(dataset = contexts_test)
            endings_test = eliminate_id(dataset = endings_test)

            binary_verifiers_val = generate_binary_verifiers(dataset = val_set)
            binary_verifiers_test = [[1,0]]*len(contexts_test)


            gen_val = batch_iter_val_cnn_sentiment(contexts = contexts_val, endings = endings_val, binary_verifiers = binary_verifiers_val)
            gen_test = batch_iter_val_cnn_sentiment(contexts = contexts_test, endings = endings_test, binary_verifiers = binary_verifiers_test)
            
            model = cnn_lstm_sent.Cnn_lstm_sentiment(train_generator = gen_val, validation_generator = gen_test)
            model.train(save_path = out_trained_models)
        
        elif args.model == "put_your_model_name_here3":
            
            print("Please put your procedure in here before running & remember to add the name of the model into the options of the parser!")
        

    if args.predict:

        """
           Path to the model to restore for predictions
        """

        """Submission file"""
        submission_path_filename = get_submission_filename()

        if args.model == "cnn_ngrams":
            
            #path_model_to_restore = os.path.join(get_latest_model(), "weights.h5")
            #print("Loading the last checkpoint of the model ", args.model, " from: ", path_model_to_restore)
            print("cnn grams prediction invoked")


        elif args.model == "put_your_model_name_here2":

            print("Put your code here before calling predict")

        elif args.model == "put_your_model_name_here3":

            print("Put your code here before calling predict")

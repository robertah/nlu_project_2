#!/usr/bin/env python3 -W ignore::DeprecationWarning

import argparse
import logging
import os
import sys
import datetime
import time

from config import *


# Remove tensorflow CPU instruction information.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def _setup_argparser():
    """Sets up the argument parser and returns the arguments.

    Returns:
        argparse.Namespace: The command line arguments.
    """
    parser = argparse.ArgumentParser(description="Control program to launch all actions related to this project.")

    parser.add_argument("-m", "--model", action="store",
                        choices=["cnn_ngrams", "put_your_model_name_here2", "put_your_model_name_here3"],
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

def initialize_negative_endings():
    neg_end = data_aug.Negative_endings()
    neg_end.load_vocabulary()
    context_pos_tagged, endings_pos_tagged = neg_end.load_corpus_no_ids()    
    #Preserve here in case the vocabulary change, do not save filters and reload them
    neg_end.filter_corpus_tags()

    return neg_end, context_pos_tagged, endings_pos_tagged

"""********************************************** USER ACTIONS from parser ************************************************************"""

if __name__ == "__main__":

    __file__ = "run.py"
    file_path = os.path.dirname(os.path.abspath(__file__))

    args = _setup_argparser()
    """
    #Once we get ready we can decomment. This avoids creating files when things have still to be debugged
    if args.train:
        out_trained_models = os.path.normpath(os.path.join(properties["SRC_DIR"],
                                              "../trained_models/",
                                              args.model,
                                              datetime.datetime.now().strftime(r"%Y-%m-%d[%Hh%M]")))
        try:
            os.makedirs(out_trained_models)
        except OSError:
            pass
    else:
        out_trained_models = os.path.normpath("..")
    """

    import training_utils as train_utils
    import negative_endings as data_aug
    from models import cnn_ngrams
    import numpy as np
    import keras


    if args.train:
        """Create a field with your model (see the default one to be customized) and put the procedure to follow to train it"""
        if args.model == "cnn_ngrams":

            print("CNN grams training invoked")
            
            print("Initializing negative endings..")
            neg_end, context_pos_tagged, endings_pos_tagged = initialize_negative_endings()
            
            train_data_generator = train_utils.batch_iter_train_cnn(data = context_pos_tagged, neg_aug_obj = neg_end,
                                             is_w2v = False, batch_size = 2, num_epochs = 5000, 
                                             shuffle=True)
            for batch in train_data_generator:
                stories_train, verif_train = zip*(batch)
                print(stories_train)
            #The things below are just a trial !

            

            #train_generator = train_utils() # TODO
            #validation_generator = train_utils() # TODO



        elif args.model == "put_your_model_name_here2":

            print("Please put your procedure in here before running & remember to add the name of the model into the options of the parser!")

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

#!/usr/bin/env python3 -W ignore::DeprecationWarning

import argparse
import sys
import datetime
import time
import negative_endings as data_aug

from models import cnn_ngrams, cnn_lstm_sent, SiameseLSTM, ffnn

from training_utils import *


from sentiment import *
from negative_endings import *
from models.skip_thoughts import skipthoughts
# Remove tensorflow CPU instruction information.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from keras.models import load_model
import csv

def _setup_argparser():
    """Sets up the argument parser and returns the arguments.

    Returns:
        argparse.Namespace: The command line arguments.
    """
    parser = argparse.ArgumentParser(description="Control program to launch all actions related to this project.")

    parser.add_argument("-m", "--model", action="store",
                        choices=["cnn_ngrams", "SiameseLSTM", "cnn_lstm", "ffnn", "ffnn_val", "ffnn_val_test"],
                        default="cnn_ngrams",
                        type=str,
                        help="the model to be used, defaults to cnn_ngrams")
    parser.add_argument("-t", "--train",
                        help="train the given model",
                        action="store_true")
    parser.add_argument("-p", "--predict",
                        help="predict on a test set given the model",
                        action="store_true")
    parser.add_argument("-e", "--evaluate",
                        help="evaluate on a test set given the model",
                        action="store_true")

    args, unknown = parser.parse_known_args()

    return args


def get_latest_model():
    """Returns the latest directory of the model specified in the arguments.

    Returns:
        (path) a path to the directory.
    """
    __file__ = "run.py"
    file_path = os.path.dirname(os.path.abspath(__file__))

    if not os.path.exists(os.path.join(file_path, "..","trained_models", args.model)):
        print("No trained model {} exists.".format(args.model))
        sys.exit(1)
    #Path to the model
    res = os.path.join(file_path, "..","trained_models", args.model)
    all_runs = [os.path.join(res, o) for o in os.listdir(res) if os.path.isdir(os.path.join(res, o))]
    res = max(all_runs, key=os.path.getmtime)
    print("Retrieving trained model from {}".format(res))
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

def get_verifiers_difference(Y_predict):
    total_labels = len(Y_predict)
    index = 0
    diff_verifiers =[]
    while index < total_labels:
        diff_verifiers.append(Y_predict[index] - Y_predict[index+1])
        index = index + 2
    return diff_verifiers

def get_predicted_labels(probabilities, submission_filename):
    labels = [1 if prob[0]>prob[1] else 2 for prob in probabilities]
    labels = np.asarray(labels, dtype=int)
    with open(submission_path_filename, "w+") as f:
        np.savetxt(submission_path_filename, labels.astype(int), fmt='%i', delimiter=',')

    print("Predicted endings saved in", submission_filename)
    return labels

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
        out_trained_models = os.path.join(os.path.abspath("run.py"), "..", "trained_models", args.model)

    print("Trained model will be saved in ", out_trained_models)


    if args.train:
        """Create a field with your model (see the default one to be customized) and put the procedure to follow to train it"""
        if args.model == "cnn_ngrams":

            print("CNN grams training invoked")
            print("Loading dataset..")
            pos_train_begin_tog, pos_train_end_tog, pos_val_begin_tog, pos_val_end_tog = load_train_val_datasets_pos_tagged(stop_words=False, lemm=True)

            print("Initializing negative endings..")
            neg_end = initialize_negative_endings(contexts = pos_train_begin_tog, endings = pos_train_end_tog)
            
            ver_val_set = generate_binary_verifiers(val_set)

            train_generator = train_utils.batch_iter_backward_train_cnn(contexts = pos_train_begin_tog, endings = pos_train_end_tog, neg_end_obj = neg_end,
                                                               batch_size = 2, num_epochs = 500, shuffle=True)
            validation_generator = train_utils.batch_iter_val_cnn(contexts = pos_val_begin_tog, endings = pos_val_end_tog, binary_verifiers = ver_val_set, 
                                                                  neg_end_obj = neg_end, batch_size = 2, num_epochs = 500, shuffle=True)
            #test_generator = train_utils.batch_iter_val_cnn(contexts = pos_test_begin_tog, endings = pos_test_end_tog, binary_verifiers = ver_test_set,
            #                                                      neg_end_obj = neg_end, batch_size = 2, num_epochs = 500, shuffle=True)
            #Initialize model
            #model = cnn_ngrams.CNN_ngrams(train_generator = validation_generator, validation_generator = test_generator)
            model = cnn_ngrams.CNN_ngrams(train_generator = train_generator, validation_generator = validation_generator)
            model.train(save_path = out_trained_models)

        elif args.model == "cnn_lstm":

            print("Please put your procedure in here before running & remember to add the name of the model into the options of the parser!")
            contexts_val = np.load(val_pos_begin)
            endings_val = np.load(val_pos_end)

            contexts_val = eliminate_id(dataset = contexts_val)
            endings_val = eliminate_id(dataset = endings_val)

            contexts_test = np.load(test_cloze_pos_begin)
            endings_test = np.load(test_cloze_pos_end)


            contexts_test = eliminate_id(dataset = contexts_test)
            endings_test = eliminate_id(dataset = endings_test)

            binary_verifiers_val = generate_binary_verifiers(dataset = val_set)
            binary_verifiers_test = [[1,0]]*len(contexts_test)


            gen_val = batch_iter_val_cnn_sentiment(contexts = contexts_val, endings = endings_val, binary_verifiers = binary_verifiers_val)
            gen_test = batch_iter_val_cnn_sentiment(contexts = contexts_test, endings = endings_test, binary_verifiers = binary_verifiers_test)

            model = cnn_lstm_sent.Cnn_lstm_sentiment(train_generator = gen_val, validation_generator = gen_test)
            model.train(save_path = out_trained_models)

        elif args.model == "SiameseLSTM":

            print("Please put your procedure in here before running & remember to add the name of the model into the options of the parser!")
            print("You chose the Siamese LSTM model; Good for you!")

            # Loading datasets (training and validation)
            print("Loading dataset..")
            pos_train_begin_tog, pos_train_end_tog, pos_val_begin_tog, pos_val_end_tog = load_train_val_datasets_pos_tagged()
            ver_val_set = generate_binary_verifiers()
            print("Initializing negative endings..")
            neg_end = initialize_negative_endings(contexts=pos_train_begin_tog, endings=pos_train_end_tog)

            # Construct data generators
            train_generator = train_utils.batch_iter_train_SiameseLSTM(contexts = pos_train_begin_tog,
                                                               endings = pos_train_end_tog,
                                                               neg_end_obj = neg_end,
                                                               batch_size = 3,
                                                               num_epochs = 500,
                                                               shuffle=True)
            validation_generator = train_utils.batch_iter_val_SiameseLSTM(contexts = pos_val_begin_tog,
                                                                  endings = pos_val_end_tog,
                                                                  binary_verifiers = ver_val_set,
                                                                  neg_end_obj = neg_end,
                                                                  batch_size = 2,
                                                                  num_epochs = 500,
                                                                  shuffle=True)

            #Creating model
            model = SiameseLSTM.SiameseLSTM(train_generator=train_generator, validation_generator = validation_generator)
            model.train()

        elif args.model == "ffnn":

            print("Loading dataset...")
            # get train data
            train_data = load_data(train_set)
            sens = [col for col in train_data if col.startswith('sen')]
            train_data = train_data[sens].values

            print("Initializing negative endings...")
            _, pos_train_end, _, _ = load_train_val_datasets_pos_tagged(together=False, stop_words=False, lemm=True)
            pos_train_begin_tog, pos_train_end_tog, _, _ = load_train_val_datasets_pos_tagged(stop_words=False, lemm=True)

            neg_end = initialize_negative_endings(contexts=pos_train_begin_tog, endings=pos_train_end_tog)

            print("Sentiment analysis...")
            sentiment_train = sentiment_analysis(train_set).values

            print("Loading skip-thoughts_model for embedding...")
            skipthoughts_model = skipthoughts.load_model()
            encoder = skipthoughts.Encoder(skipthoughts_model)

            print("Defining batch data generators... ")
            # train_generator = ffnn.batch_iter(train_data, pos_train_end, neg_end, sentiment_train, encoder, 128)
            # validation_generator = ffnn.batch_iter_val(val_data, sentiment_val, encoder, ver_val_set, 128)
            train_generator = ffnn.batch_iter(train_data, pos_train_end, neg_end, sentiment_train, encoder, 64)
            # validation_generator = ffnn.batch_iter_val(val_data, sentiment_val, encoder, ver_val_set, 64)
            validation_generator = ffnn.batch_iter_val(val_set, encoder, batch_size=64)

            train_size, val_size = len(train_data), 1871

            print("Initializing feed-forward neural network...")
            model = ffnn.FFNN(train_generator=train_generator, validation_generator=validation_generator)
            model.train(train_size, val_size, out_trained_models)

        elif args.model == "ffnn_val":

            print("Loading skip-thoughts_model for embedding...")
            skipthoughts_model = skipthoughts.load_model()
            encoder = skipthoughts.Encoder(skipthoughts_model)

            X = ffnn.transform(val_set, encoder)
            Y = generate_binary_verifiers(val_set)

            n_stories = len(X)
            train_indexes = np.random.choice(n_stories, int(n_stories*0.9), replace=False)

            X_train = np.take(X, train_indexes, axis=0)
            Y_train = np.take(Y, train_indexes, axis=0)

            X_val = np.delete(X, train_indexes, axis=0)
            Y_val = np.delete(Y, train_indexes, axis=0)

            print("Defining batch data generators... ")
            train_generator = ffnn.batch_iter_val(X_train, Y_train, batch_size=64)
            validation_generator = ffnn.batch_iter_val(X_val, Y_val, batch_size=64)

            print("Initializing feed-forward neural network...")
            model = ffnn.FFNN(train_generator=train_generator, validation_generator=validation_generator)
            model.train(len(X_train), len(X_val), out_trained_models)


        elif args.model == "ffnn_val_test":

            print("Loading skip-thoughts_model for embedding...")
            skipthoughts_model = skipthoughts.load_model()
            encoder = skipthoughts.Encoder(skipthoughts_model)

            X_train = ffnn.transform(val_set, encoder)
            Y_train = generate_binary_verifiers(val_set)

            X_val = ffnn.transform(test_set_cloze, encoder)
            Y_val = generate_binary_verifiers(test_set_cloze)

            print("Defining batch data generators... ")
            train_generator = ffnn.batch_iter_val(X_train, Y_train, batch_size=64)
            validation_generator = ffnn.batch_iter_val(X_val, Y_val, batch_size=64)

            print("Initializing feed-forward neural network...")
            model = ffnn.FFNN(train_generator=train_generator, validation_generator=validation_generator)
            model.train(len(X_train), len(X_val), out_trained_models)



    if args.predict:

        """Path to the model to restore for predictions -> be sure you save the model as model.h5
           In reality, what is saved is not just the weights but the entire model structure"""
        model_path = os.path.join(get_latest_model(), "model.h5")
        """Submission file -> It will be in the same folder of the model restored to predict
           e.g trained_model/27_05_2012.../submission_modelname...."""
        submission_path_filename = get_submission_filename()

        if args.model == "cnn_ngrams":
            
            print("This prediction branch has not been implemented")


        elif args.model == "cnn_lstm":

            print("Predicting with CNN_LSTM sentiment based..")
            contexts_test = np.load(test_cloze_pos_begin)
            endings_test = np.load(test_cloze_pos_end)

            contexts_test = eliminate_id(dataset = contexts_test)
            endings_test = eliminate_id(dataset = endings_test)

            test_generator = batch_iter_val_cnn_sentiment(contexts = contexts_test, endings = endings_test, binary_verifiers = [], test = True)
            model_class = cnn_lstm_sent.Cnn_lstm_sentiment(train_generator = [], path=model_path)
            model = model_class.model
            Y_predict = model.predict_generator(test_generator, steps=2343)
            verifiers_differences = get_verifiers_difference(Y_predict = Y_predict)
            Y_labels = get_predicted_labels(verifiers_differences, submission_path_filename)

            print(Y_labels)
            print(Y_labels.shape)

        elif args.model == "ffnn":

            print("bla bla")

        elif args.model == "ffnn_val" or args.model == "ffnn_val_test":

            model = load_model(model_path)

            print("Loading skip-thoughts_model for embedding...")

            skipthoughts_model = skipthoughts.load_model()
            encoder = skipthoughts.Encoder(skipthoughts_model)

            X_test = ffnn.transform(test_set, encoder)
            Y_predict = model.predict(X_test)
            Y_labels = get_predicted_labels(Y_predict, submission_path_filename)

    if args.evaluate:

        """Path to the model to restore for predictions -> be sure you save the model as model.h5
           In reality, what is saved is not just the weights but the entire model structure""" #TODO
        model_path = os.path.join(get_latest_model(), "model.h5")
        """Submission file -> It will be in the same folder of the model restored to predict
           e.g trained_model/27_05_2012.../submission_modelname...."""
        submission_path_filename = get_submission_filename()

        if args.model == "ffnn_val" or args.model == "ffnn_val_test":

            model = load_model(model_path)

            print("Loading skip-thoughts_model for embedding...")

            skipthoughts_model = skipthoughts.load_model()
            encoder = skipthoughts.Encoder(skipthoughts_model)

            X_test = ffnn.transform(test_set_cloze, encoder)
            Y_test = generate_binary_verifiers(test_set_cloze)
            Y_test = np.asarray(Y_test)
            (loss, accuracy) = model.evaluate(X_test, Y_test, batch_size=64, verbose=1)
            print("[INFO] loss={:.4f}, accuracy: {:.4f}%".format(loss, accuracy * 100))


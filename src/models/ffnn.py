from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
from keras.regularizers import l2
from run import *
import os


class FFNN():
    """
    Feed-forward neural network:
        - input: 9604 dim array, with embeddings with last sentences and two endings, and sentiment analysis
        - 3 layers: dimensions 3200, 1600, 800
        - activation: softmax
    """

    def __init__(self, train_generator, validation_generator=[], batch_size=128, path=None):
        """
        Initialize the feed-forwards neural network

        :param train_generator: generator for training batches
        :param validation_generator: generator for validation batches
        :param path: path to trained model if it exists
        """

        # initialize model parameters
        self.train_generator = train_generator
        self.validation_generator = validation_generator
        self.batch_size = batch_size

        # load trained model if the path is given
        if path:
            print("Loading existing model from {}".format(path))
            self.load(path)
            print("Finished loading model")
            return

        # create feed-forward neural network layers
        self.model = Sequential()
        self.model.add(Dense(3200, input_dim=9604, kernel_initializer="uniform", activation="relu"))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(1600, kernel_initializer="uniform", activation="relu"))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(800, kernel_initializer="uniform", activation="relu"))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(2, kernel_regularizer=l2(1e-3), activation="softmax"))

    def train(self, train_size, val_size, save_path):
        """
        Train function for the feed forward neural networks

        :param train_size: number of samples in the train data
        :param val_size: number of samples in the validation data
        :param save_path: path to model for saving
        :return:
        """
        print("Compiling model...")

        # configure the model for training
        self.model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

        print(self.model.summary())

        # reduce learning rate when a metric has stopped improving
        lr_callback = ReduceLROnPlateau(monitor="acc", factor=0.5, patience=0.5, verbose=0, cooldown=0, min_lr=0)

        # stop training when validation loss has stopped improving
        stop_callback = EarlyStopping(monitor="val_loss", min_delta=0, patience=3)

        tensorboard_callback = TensorBoard(log_dir=out_trained_models, histogram_freq=0, batch_size=1,
                                           write_graph=True,
                                           write_grads=False, embeddings_freq=0,
                                           embeddings_layer_names=None, embeddings_metadata=None)

        # save the model after every epoch
        checkpoint_callback = ModelCheckpoint(os.path.join(save_path, 'model.h5'), monitor='val_acc', verbose=0, save_best_only=True,
                                              save_weights_only=False, mode='auto', period=1)

        # train the model on data generated batch-by-batch by customized generator
        n_batches = np.ceil(train_size/batch_size)
        n_batches_val = np.ceil(val_size/batch_size)
        self.model.fit_generator(generator=self.train_generator, steps_per_epoch=n_batches,
                                 verbose=1, epochs=50, shuffle=True,
                                 callbacks=[lr_callback, stop_callback, tensorboard_callback,
                                            checkpoint_callback],
                                 validation_data=self.validation_generator,
                                 validation_steps=n_batches_val)
        return self


# BATCH GENERATORS AND ADDITIONAL PREPRPCESSING #############################################################

def batch_iter(sentences, endings, neg_end_obj, sentiment, encoder, batch_size, aug_batch_size=2):
    """
    Create batch generator for training data

    :param sentences: sentences in the train data
    :param endings: endings in the train data
    :param neg_end_obj: object for negative endings
    :param sentiment: sentiment analysis for training
    :param encoder: encoder for skip-thought embedding
    :param batch_size: batch size for training
    :param aug_batch_size: number of negative endings + right ending per story
    :return: batches
    """

    # get last sentences
    last_sentences = get_context_sentence(sentences, 4)

    # get number of stories
    n_stories = len(last_sentences)

    # load vocabulary
    vocabulary = load_vocabulary()

    # generate negative endings
    print("Data augmentation for negative endings for the next epoch -> stochastic approach..")
    batch_endings, ver_batch_end = batches_pos_neg_endings(neg_end_obj, endings, aug_batch_size)

    # map negative endings to words
    batch_endings_words = np.empty((n_stories, aug_batch_size), dtype=np.ndarray)
    print("Mapping generated negative endings to words...")
    for i in range(n_stories):
        for j in range(aug_batch_size):
            batch_endings_words[i, j] = get_words_from_indexes(batch_endings[i][j], vocabulary)
            batch_endings_words[i, j] = ' '.join([word for word in batch_endings_words[i, j] if word != pad])
        if i % 5000 == 0:
            print(i, "/", n_stories)

    # create embeddings
    print("Generating skip-thoughts embeddings for last sentences (it might take a while)...")
    last_sentences_encoded = encoder.encode(last_sentences, verbose=False)
    print("Generating skip-thoughts embeddings for endings (it might take a while)...")
    sentences_encoded = np.empty((n_stories, aug_batch_size, 4800))
    for i in range(endings.shape[1]):
        sentences_encoded[:, i] = encoder.encode(batch_endings_words[:, i], verbose=False)
    sentences_encoded = np.reshape(sentences_encoded, (n_stories, -1))

    # sum last sentences and endings
    for i in range(n_stories):
        sentences_encoded[i] = np.tile(last_sentences_encoded[i], aug_batch_size) + sentences_encoded[i]

    # concatenate features
    features = np.concatenate((sentences_encoded, sentiment), axis=1)

    # get binary verifiers for labels
    ver_batch_end = np.reshape(ver_batch_end, (n_stories, 2))

    print("Train generator for the new epoch ready..")

    while True:
        for i in range(n_stories-batch_size):
            index = np.random.choice(np.arange(0, n_stories-batch_size, 2), 1)[0]
            X_train = features[index:index+batch_size]
            Y_train = ver_batch_end[index:index+batch_size]
            yield X_train, Y_train


def transform(dataset, encoder):
    """
    Transform validation and test data for batch generators

    :param dataset: validation or test dataset
    :param encoder: encoder for skip-thought embeddings
    :return: features vector with embeddings and sentiment analysis
    """

    # get sentences from data set
    sentences = load_data(dataset)
    sens = [col for col in sentences if col.startswith('sen')]
    sentences = sentences[sens].values

    # get last sentences
    last_sentences = sentences[:, 4]

    # get endings
    endings = sentences[:, 4:]

    # get number of stories
    n_stories = len(last_sentences)

    # generate skip-thought embeddings
    last_sentences_encoded = encoder.encode(last_sentences, verbose=False)
    sentences_encoded = np.empty((n_stories, 2, 4800))
    for i in range(endings.shape[1]):
        sentences_encoded[:, i] = encoder.encode(endings[:, i], verbose=False)
    sentences_encoded = np.reshape(sentences_encoded, (n_stories, -1))

    # sum last sentences and endings
    for i in range(n_stories):
        sentences_encoded[i] = np.tile(last_sentences_encoded[i], 2) + sentences_encoded[i]

    # create sentiment array for dataset
    sentiment = sentiment_analysis(dataset)

    # concatenate features
    features = np.concatenate((sentences_encoded, sentiment), axis=1)

    return features


def batch_iter_val(features, labels, batch_size):
    """
    Create batch generator for validation and test data

    :param features: features array generated with transform()
    :param labels: labels for the endings
    :param batch_size: batch size for validation
    :return: batches
    """

    # get number of stories
    n_stories = len(features)

    # reshape labels
    labels = np.reshape(labels, (n_stories, 2))

    while True:
        for i in range(n_stories-batch_size):
            index = np.random.choice(n_stories-batch_size, 1)[0]
            X = features[index:index+batch_size]
            Y = labels[index:index+batch_size]
            yield X, Y

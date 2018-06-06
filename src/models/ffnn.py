from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
from keras.regularizers import l2
from run import *
import os


class FFNN():
    """
    Feed-forward neural network:
        - input: sentiment analysis of the context, last sententece, ending, answer label
        - embedding: skip thoughts embeddings
        - 3 layers: dimensions 2400, 1200, 600
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
        # self.model.add(Dense(4800, input_dim=9604, kernel_initializer="uniform", activation="relu"))
        # self.model.add(Dense(2400, kernel_initializer="uniform", activation="relu"))
        self.model.add(Dense(2400, input_dim=4804, kernel_initializer="uniform", activation="relu"))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(1200, kernel_initializer="uniform", activation="relu"))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(600, kernel_initializer="uniform", activation="relu"))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(2, kernel_regularizer=l2(1e-3), activation="softmax"))



    def train(self):
        out_trained_models = '../trained_models'

        print("Compiling model...")

        # configure the model for training
        self.model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

        print(self.model.summary())

        # reduce learning rate when a metric has stopped improving TODO check params
        lr_callback = ReduceLROnPlateau(monitor="acc", factor=0.5, patience=0.5, verbose=0, cooldown=0, min_lr=0)

        # stop training when a monitored quantity has stopped improving
        stop_callback = EarlyStopping(monitor="acc", min_delta=0.0001, patience=5)

        # TODO change log path??

        tensorboard_callback = TensorBoard(log_dir=out_trained_models, histogram_freq=0, batch_size=1,
                                           write_graph=True,
                                           write_grads=False, embeddings_freq=0,
                                           embeddings_layer_names=None, embeddings_metadata=None)

        checkpoint_path = os.path.join(out_trained_models, 'ffnn/model.h5')

        # save the model after every epoch
        checkpoint_callback = ModelCheckpoint(checkpoint_path, monitor='val_loss', verbose=0, save_best_only=True,
                                              save_weights_only=False, mode='auto', period=1)

        # train the model on data generated batch-by-batch by customized generator
        n_batches = np.ceil(88161/batch_size)  # batches of samples
        n_batches_val = np.ceil(1871/batch_size)

        self.model.fit_generator(generator=self.train_generator, steps_per_epoch=n_batches,
                                 verbose=1, epochs=100, shuffle=True,
                                 callbacks=[lr_callback, stop_callback, tensorboard_callback,
                                            checkpoint_callback],
                                 validation_data=self.validation_generator,
                                 validation_steps=n_batches_val)

        return self

    def evaluate(self, X_test, Y_test):
        print("Evaluating on testing set...")
        (loss, accuracy) = self.model.evaluate(X_test, Y_test,
                                               batch_size=128, verbose=1)
        print("[INFO] loss={:.4f}, accuracy: {:.4f}%".format(loss, accuracy * 100))

    def save(self, path):
        """
        Save the trained model
        :param path: path to trained model
        :return:
        """

        self.model.save(path)
        print("Model saved to {}".format(path))



def batch_iter(sentences, endings, neg_end_obj, sentiment, encoder, batch_size, aug_batch_size=2):
    """
    Generates a batch generator for the train set.
    """

    last_sentences = get_context_sentence(sentences, 4)

    print("Data augmentation for negative endings for the next epoch -> stochastic approach..")
    batch_endings, ver_batch_end = batches_pos_neg_endings(neg_end_obj, endings, aug_batch_size)

    # batch_endings, ver_batch_end = batches_backwards_neg_endings(neg_end_obj, endings, aug_batch_size)

    n_stories = len(batch_endings)

    vocabulary = load_vocabulary()
    batch_endings_words = np.empty((n_stories * aug_batch_size), dtype=np.ndarray)

    print("Mapping generated negative endings to words...")
    for i in range(n_stories):
        for j in range(aug_batch_size):
            batch_endings_words[i * aug_batch_size + j] = get_words_from_indexes(batch_endings[i][j], vocabulary)
            batch_endings_words[i * aug_batch_size + j] = ' '.join(
                [word for word in batch_endings_words[i * aug_batch_size + j] if word != pad])
        if i % 5000 == 0:
            print(i, "/", n_stories)

    # create embeddings
    print("Generating skip-thoughts embeddings for last sentences (it might take a while)...")
    last_sentences_encoded = encoder.encode(last_sentences, verbose=False)
    print("Generating skip-thoughts embeddings for endings (it might take a while)...")
    batch_endings_encoded = encoder.encode(batch_endings_words, verbose=False)

    # create features array
    print("Creating features array...")
    sentiment_repeat = np.repeat(sentiment, aug_batch_size, axis=0)
    last_sentences_repeat = np.repeat(last_sentences_encoded, aug_batch_size, axis=0)
    last_sentences_endings = last_sentences_repeat + batch_endings_encoded
    features = np.concatenate((last_sentences_endings, sentiment_repeat), axis=1)

    # create labels array
    print("Creating labels array...")
    labels = np.empty((n_stories * aug_batch_size, 2), dtype=int)
    for i in range(n_stories):
        for j in range(aug_batch_size):
            labels[i * aug_batch_size + j] = [ver_batch_end[i][j], 1 - ver_batch_end[i][j]]

    print("Train generator for the new epoch ready..")

    total_steps = len(features)

    while True:
        for i in range(total_steps-batch_size):
            index = np.random.choice(np.arange(0, total_steps-batch_size, 2), 1)[0]
            X_train = features[index:index+batch_size]
            Y_train = labels[index:index+batch_size]
            yield X_train, Y_train


def batch_iter_val(sentences, sentiment, encoder, labels, batch_size):

    last_sentences = sentences[:, 4]
    endings = sentences[:, 4:]

    n_stories = len(last_sentences)

    new_labels = np.empty((len(labels)*2, 2), dtype=int)
    for i in range(len(labels)):
        new_labels[i*2] = labels[i]
        new_labels[i*2+1][0], new_labels[i*2+1][1] = labels[i][1], labels[i][0]

    sentences_encoded = np.empty((n_stories, 2, 4800))

    # print("##### Starting encoding......")

    last_sentences_encoded = encoder.encode(last_sentences, verbose=False)
    # print(last_sentences_encoded)
    # print(last_sentences_encoded.shape)

    for i in range(endings.shape[1]):
        sentences_encoded[:, i] = encoder.encode(endings[:, i], verbose=False)

    # print("RESHAPING SENTENCES")
    # print(sentences_encoded[:4])
    sentences_encoded = np.reshape(sentences_encoded, (-1, 4800))

    for i in range(n_stories):
        sentences_encoded[i*2] += last_sentences_encoded[i]
        sentences_encoded[i*2+1] += last_sentences_encoded[i]

    # print("ADDED LAST SENTENCE")
    # print(sentences_encoded[:4])

    features = np.concatenate((sentences_encoded, np.repeat(sentiment, 2, axis=0)), axis=1)

    # print("SENTIMENT ADDED")
    # print(features)
    # print(features.shape)

    n_stories = len(features)

    while True:
        for i in range(n_stories-batch_size):
            index = np.random.choice(n_stories-batch_size, 1)[0]
            X = features[index:index+batch_size]
            Y = new_labels[index:index+batch_size]
            yield X, Y


# def batch_iter_val_data(sentences, sentiment, encoder, labels, batch_size):
#
#     last_sentences = sentences[:, 4]
#     endings = sentences[:, 4:]
#
#     n_stories = len(last_sentences)
#
#     sentences_encoded = np.empty((n_stories, 2, 4800))
#
#     print("##### Starting encoding......")
#
#     last_sentences_encoded = encoder.encode(last_sentences, verbose=False)
#
#     for i in range(endings.shape[1]):
#         sentences_encoded[:, i] = encoder.encode(endings[:, i], verbose=False)
#
#     sentences_encoded = np.reshape(sentences_encoded, (n_stories, -1))
#
#     for i in range(n_stories):
#         sentences_encoded[i] = np.tile(last_sentences_encoded[i], 2) + sentences_encoded[i]
#
#     features = np.concatenate((sentences_encoded, sentiment), axis=1)
#
#     n_stories = len(features)
#
#     while True:
#         for i in range(n_stories-batch_size):
#             index = np.random.choice(n_stories-batch_size, 1)[0]
#             X = features[index:index+batch_size]
#             Y = labels[index:index+batch_size]
#             yield X, Y

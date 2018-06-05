from keras.models import Sequential
from keras.layers import Dense
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
        self.model.add(Dense(4800, input_dim=9604, kernel_initializer="uniform", activation="relu"))
        self.model.add(Dense(2400, kernel_initializer="uniform", activation="relu"))
        self.model.add(Dense(1200, kernel_initializer="uniform", activation="relu"))
        self.model.add(Dense(600, kernel_initializer="uniform", activation="relu"))
        self.model.add(Dense(2, kernel_regularizer=l2(1e-6), activation="softmax"))



    def train(self):
        out_trained_models = '../trained_models'

        print("[INFO] compiling model...")

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

        checkpoint_path = os.path.join(out_trained_models, 'ffnn/weights.h5')

        # save the model after every epoch
        checkpoint_callback = ModelCheckpoint(checkpoint_path, monitor='val_loss', verbose=0, save_best_only=True,
                                              save_weights_only=False, mode='auto', period=1)

        # train the model on data generated batch-by-batch by customized generator
        n_batches = int(88161 / batch_size)  # batches of samples

        self.model.fit_generator(generator=self.train_generator, steps_per_epoch=n_batches,
                                 verbose=2, epochs=100, shuffle=True,
                                 callbacks=[lr_callback, stop_callback, tensorboard_callback,
                                            checkpoint_callback],
                                 validation_data=self.validation_generator,
                                 validation_steps=n_batches)

        return self

    def evaluate(self, X_test, Y_test):
        print("[INFO] evaluating on testing set...")
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



def batch_iter(sentences, endings, neg_end_obj, sentiment, encoder, batch_size=2):
    """
    Generates a batch generator for the train set.
    """

    vocabulary = load_vocabulary()

    sentiment = sentiment[:20]

    last_sentences = get_context_sentence(sentences, 4)[:20]  # TODO remove size limit

    aug_batch_size = 2

    print("Augmenting with negative endings for the next epoch -> stochastic approach..")
    batch_endings, ver_batch_end = batches_pos_neg_endings(neg_end_obj=neg_end_obj, endings=endings,
                                                           batch_size=batch_size)

    batch_endings = batch_endings[:20]  # TODO remove size limit
    ver_batch_end = ver_batch_end[:20]

    n_stories = len(batch_endings)

    batch_endings_words = np.empty((n_stories * aug_batch_size), dtype=np.ndarray)

    for i in range(n_stories):
        for j in range(aug_batch_size):
            batch_endings_words[i * aug_batch_size + j] = get_words_from_indexes(batch_endings[i][j], vocabulary)
            batch_endings_words[i * aug_batch_size + j] = ' '.join(
                [word for word in batch_endings_words[i * aug_batch_size + j] if word != pad])

    # create embeddings
    last_sentences_encoded = encoder.encode(last_sentences, batch_size=1)
    batch_endings_encoded = encoder.encode(batch_endings_words, batch_size=1)

    # create features array
    sentiment_repeat = np.repeat(sentiment, aug_batch_size, axis=0)
    last_sentences_repeat = np.repeat(last_sentences_encoded, aug_batch_size, axis=0)
    last_sentences_endings = last_sentences_repeat + batch_endings_encoded
    features = np.concatenate((sentiment_repeat, last_sentences_endings), axis=1)

    # create labels array
    labels = np.empty((n_stories * aug_batch_size), dtype=np.ndarray)
    labels_n = np.empty((n_stories * aug_batch_size), dtype=int)
    for i in range(n_stories):
        for j in range(aug_batch_size):
            labels[i * aug_batch_size + j] = [ver_batch_end[i][j], 1 - ver_batch_end[i][j]]
            labels_n[i * aug_batch_size + j] = ver_batch_end[i][j] + 1


    print(labels_n)
    print(len(labels_n))

    print("Train generator for the new epoch ready..")

    total_steps = len(features)

    # X_train = np.empty((batch_size, features.shape[1]))
    # Y_train = np.empty((batch_size, 2))

    while True:
        for i in range(0, total_steps):
            index = np.random.choice(np.arange(0, total_steps - batch_size, 2), 1)[0]
            X_train = np.asarray(features[index:index+batch_size])
            Y_train = labels_n[index:index+batch_size]
            print(X_train, Y_train)
            yield X_train, Y_train


def batch_iter_val(sentences, sentiment, encoder):
    last_sentences = get_context_sentence(sentences, 4)[:20]  # TODO remove size limit
    endings = get_context_sentence(sentences, 5)[:20]

    n_stories = len(last_sentences)

    # create embeddings
    last_sentences_encoded = encoder.encode(last_sentences, batch_size=1)
    batch_endings_encoded = encoder.encode(endings, batch_size=1)

    # create features array
    sentiment = sentiment[:20]
    last_sentences_endings = last_sentences_encoded + batch_endings_encoded
    features = np.concatenate((sentiment, last_sentences_endings), axis=1)

    print(features)

    # create labels array
    ver_batch_end = generate_binary_verifiers()[:20]

    labels = np.empty((n_stories), dtype=np.ndarray)
    labels_n = np.empty(n_stories, dtype=int)
    for i in range(n_stories):
        labels[i] = ver_batch_end[i]

    while True:
        for i in range(batch_size):
            index = np.random.choice(features.shape[0], 1)
            X_val = features[index]
            Y_val = labels_n[index][0]
            yield np.asarray(X_val), np.asarray(Y_val)


def batch_iter_val_train(sentences, sentiment, encoder, labels):

    last_sentences = sentences[:, 4]
    endings = sentences[:, 4:]

    n_stories = len(last_sentences)

    sentences_encoded = np.empty((n_stories, 2, 4800))

    print("##### Starting encoding......")

    last_sentences_encoded = encoder.encode(last_sentences)

    for i in range(endings.shape[1]):
        sentences_encoded[:, i] = encoder.encode(endings[:, i])

    sentences_encoded = np.reshape(sentences_encoded, (n_stories, -1))
    print(sentences_encoded)
    print(last_sentences_encoded)

    for i in range(n_stories):
        sentences_encoded[i] = np.tile(last_sentences_encoded[i], 2) + sentences_encoded[i]

    print("\n\n\n#######################")
    print(sentences_encoded[:10])

    features = np.concatenate((sentences_encoded, sentiment), axis=1)
    print(features)

    n_stories = len(features)

    while True:
        for i in range(n_stories):
            X_train = features[i]
            Y_train = labels[i]
            print(X_train.shape, Y_train.shape)
            yield X_train, Y_train


def batch_iter_val_val(sentences, sentiment, encoder, labels):
    last_sentences = sentences[:, 4]
    endings = sentences[:, 4:]

    sentences_encoded = np.empty(endings.shape)

    for i, end in np.ndenumerate(endings):
        sentences_encoded = encoder.encode(last_sentences[i[0]]) + encoder.encode(endings[i])

    print("\n\n\n#######################")
    print(sentences_encoded[:10])

    features = np.concatenate((sentences_encoded, sentiment), axis=1)
    print(features)

    n_stories = len(features)

    while True:
        for i in range(n_stories):
            X_val = features[i]
            Y_val = labels[i]
            yield X_val, Y_val


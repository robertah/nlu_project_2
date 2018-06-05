from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Activation

from .skip_thoughts import skipthoughts


class FFNN():

    def __init__(self):

        # load skip thoughts model and encoder for embeddings
        self.skipthoughts_model = skipthoughts.load_model()
        self.encoder = skipthoughts.encode(self.skipthoughts_model)

        # create feed-forward neural network layers
        self.model = Sequential()
        self.model.add(Dense(2400, input_dim=3072, init="uniform", activation="relu"))
        self.model.add(Dense(1200, init="uniform", activation="relu"))
        self.model.add(Dense(600))
        self.model.add(Activation("softmax"))


    def train(self, X_train, Y_train, nb_epoch, batch_size, X_val, Y_val):
        print("[INFO] compiling model...")
        self.model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
        self.model.fit(X_train, Y_train, nb_epoch=nb_epoch, batch_size=batch_size, validation_data=(X_val, Y_val))
        return self

    def evaluate(self, X_test, Y_test):
        print("[INFO] evaluating on testing set...")
        (loss, accuracy) = self.model.evaluate(X_test, Y_test,
                                               batch_size=128, verbose=1)
        print("[INFO] loss={:.4f}, accuracy: {:.4f}%".format(loss, accuracy * 100))

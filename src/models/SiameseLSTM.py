import keras
from keras.preprocessing import sequence
from keras.models import Model, Sequential
from keras.layers import LSTM, Embedding, Input, Merge, Dense
from keras.optimizers import Adadelta,SGD
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.callbacks import TensorBoard
import keras.backend as K
import numpy as np
import sys
sys.path.append("..")
from config import *
from data_utils import *
from preprocessing import *
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from copy import deepcopy
from negative_endings import *
import pickle
import training_utils as train_utils


'''
https://stackoverflow.com/questions/46466013/siamese-network-with-lstm-for-sentence-similarity-in-keras-gives-periodically-th
'''

class SiameseLSTM():
    '''
    Model for a Siamese LSTM - > will need two inputs for LSTM's of equal weights
    '''

    def __init__(self, seq_len, n_epoch, train_dataA, train_dataB, train_y, val_dataA, val_dataB, val_y,  embedding_docs):


        self.seq_len = seq_len
        self.m_epoch = n_epoch
        self.embedding_docs = embedding_docs

        self.train_dataA = train_dataA
        self.train_dataB = train_dataB
        self.train_y = train_y
        self.val_dataA = val_dataA
        self.val_dataB = val_dataB
        self.val_y = val_y

        # From config file
        self.embedding_dim = embedding_dim

        # If no base_network
        # prepare embedding
        self.embeddings = train_utils.embedding(embedding_docs)
        print(self.embeddings.shape)

        # define model
        self.model = Sequential()
        print(self.model.summary())
        print("Embedding input shape: {}".format(self.seq_len))
        self.model.add(Embedding(input_dim=len(self.embeddings), output_dim=self.embedding_dim,
                            weights=[self.embeddings], input_shape=(self.seq_len,), trainable=False))
        # model.add(Embedding(input_dim = vocabulary_size, output_dim = embedding_dim, input_length = seq_len))
        print("LSTM batch input shape: {}".format([self.seq_len, self.embedding_dim]))
        self.model.add(LSTM(128, batch_input_shape=(None, seq_len, self.embedding_dim), return_sequences=False))
        # model.add(Dense(50, activation='relu'))
        # model.add(Dense(10, activation='relu'))
        # model.add(Dense(1, activation='relu'))
        print(self.model.summary())

        self.input_a = Input(shape=(self.seq_len, ), dtype='int32')
        self.input_b = Input(shape=(self.seq_len, ), dtype='int32')

        self.processed_a = self.model(self.input_a)
        self.processed_b = self.model(self.input_b)


        # If using base_network
        # base_network = create_base_network(feature_dim, seq_len)

        # model = Sequential()
        # model.add(Embedding(inpu_dim=vocabulary_size, output_dim=embedding_dim, weights=[embedding_matrix],
        #                     input_length=story_len, trainable=False))
        # model.add(Embedding(input_dim = vocabulary_size, output_dim = embedding_dim, input_length = story_len))
        # model.add(LSTM(100, batch_input_shape=(None, seq_len, feature_dim), return_sequences=True))
        # print(model.summary())

        # input_a = Input(shape=(seq_len, ), dtype='int32')
        # input_b = Input(shape=(seq_len, ), dtype='int32')
        # processed_a = base_network(input_a)
        # processed_b = base_network(input_b)

        self.distance = keras.layers.Lambda(self.cosine_distance, output_shape=self.cosine_dist_output_shape)([self.processed_a, self.processed_b])
        self.model = Model([self.input_a, self.input_b], self.distance)
        self.adam_optimizer = keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

        self.model.compile(loss='mean_squared_error', optimizer=self.adam_optimizer, metrics=['accuracy'])
        self.model.summary()
        print(self.model.summary())

        # Fitting if not generator
        # model.fit(x=[train_data[:,0], train_data[:,1]], y=[train_data[:,2]],
        #           batch_size=2,
        #           epochs=n_epoch,
        #           validation_data=([val_data[:,0], val_data[:,1]], [val_data[:,2]]))


        # Fitting if generator
        # out_trained_models = '../trained_models'
        #
        # lr_callback = keras.callbacks.ReduceLROnPlateau(monitor="acc",
        #                                                 factor=0.5,
        #                                                 patience=0.5,
        #                                                 verbose=0,
        #                                                 cooldown=0,
        #                                                 min_lr=0)
        # stop_callback = keras.callbacks.EarlyStopping(monitor="acc",
        #                                               min_delta=0.0001,
        #                                               patience=11,
        #                                               verbose=0,
        #                                               mode="auto")
        #
        # tensorboard_callback = keras.callbacks.TensorBoard(log_dir=out_trained_models, histogram_freq=0, batch_size=1,
        #                                                    write_graph=True,
        #                                                    write_grads=False, embeddings_freq=0,
        #                                                    embeddings_layer_names=None, embeddings_metadata=None)
        #
        # checkpoint_callback = keras.callbacks.ModelCheckpoint(
        #     os.path.join(out_trained_models, 'cnn_ngrams/model.h5'),
        #     monitor='val_loss', verbose=0, save_best_only=True,
        #     save_weights_only=False, mode='auto', period=1)
        #
        # model.fit_generator(train_generator,
        #                     steps_per_epoch=1871,
        #                     verbose=2,
        #                     epochs=n_epoch,
        #                     callbacks=[lr_callback, stop_callback, tensorboard_callback, checkpoint_callback],
        #                     validation_data=validation_generator,
        #                     validation_steps=1871)
        return None


    def train(self, save_path):
        """Train the model.
        Args:
            epochs (int): default: 100 - epochs to train.
            steps (int): default: len(dataset) - batches per epoch to train.
        """

        out_trained_models = '../trained_models'

        lr_callback = keras.callbacks.ReduceLROnPlateau(monitor="acc",
                                                        factor=0.5,
                                                        patience=0.5,
                                                        verbose=0,
                                                        cooldown=0,
                                                        min_lr=0)
        stop_callback = keras.callbacks.EarlyStopping(monitor="acc",
                                                      min_delta=0.0001,
                                                      patience=11,
                                                      verbose=0,
                                                      mode="auto")

        tensorboard_callback = keras.callbacks.TensorBoard(log_dir=out_trained_models, histogram_freq=0, batch_size=1,
                                                           write_graph=True,
                                                           write_grads=False, embeddings_freq=0,
                                                           embeddings_layer_names=None, embeddings_metadata=None)

        checkpoint_callback = keras.callbacks.ModelCheckpoint(
            os.path.join(save_path, 'model.h5'),
            monitor='val_loss', verbose=0, save_best_only=True,
            save_weights_only=False, mode='auto', period=1)

        self.model.fit(x=[self.train_dataA, self.train_dataB], y=self.train_y,
                       epochs=n_epoch,
                       validation_data=([self.val_dataA, self.val_dataB], self.val_y),
                       steps_per_epoch=1871,
                       verbose=2,
                       shuffle=True,
                       callbacks=[lr_callback, stop_callback, tensorboard_callback, checkpoint_callback],
                       validation_steps=140)

    def save(self, path):
        """Save the model of the trained model.

        Args:
            path (path): path for the model file.
        """
        self.model.save(path)
        print("Model saved to {}".format(path))

    def exponent_neg_manhattan_distance(self, A, B):
        """
        Helper function used to estimate similarity between the LSTM outputs
        :param X:  output of LSTM with input X
        :param Y:  output of LSTM with input Y
        :return: Manhattan distance between input vectors
        """
        return K.exp(-K.sum(K.abs(A - B), axis=1, keepdims=True))


    def cosine_distance(self, vecs):
        #I'm not sure about this function too
        y_true, y_pred = vecs
        y_true = K.l2_normalize(y_true, axis=-1)
        y_pred = K.l2_normalize(y_pred, axis=-1)
        return K.mean(1 - K.sum((y_true * y_pred), axis=-1))


    def cosine_dist_output_shape(self, shapes):
        shape1, shape2 = shapes
        print((shape1[0], 1))
        return (shape1[0], 1)


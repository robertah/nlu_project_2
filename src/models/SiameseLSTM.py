import keras
from keras.preprocessing import sequence
from keras.models import Model,Sequential
from keras.layers import LSTM, Embedding, Input, Merge
from keras.optimizers import Adadelta,SGD
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.callbacks import TensorBoard
import keras.backend as K

"""
Resources used: 
https://github.com/rupak-118/Quora-Question-Pairs/blob/master/MaLSTM_train.py
LeCun's paper on Siamese networks: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf

Still to read (could be interesting):
http://www.mit.edu/~jonasm/info/MuellerThyagarajan_AAAI16.pdf
https://ac.els-cdn.com/S1877050917320847/1-s2.0-S1877050917320847-main.pdf?_tid=2a77f1f1-60fe-4992-8a23-041c11da5a30&acdnat=1527584460_92f00d6539077486b1bc415347a7be48

For me (Keras implementation of LSTM):
http://adventuresinmachinelearning.com/keras-lstm-tutorial/
"""

"""
IDEA: Use siamese LSTM network with Manhattan distance as similarity measure
Input 1 = sen1-sen4 concatenated 
Input 2 = sen5 or sen 6
Each word in input is converted to a word index, used with word-2-vec -> 300-dimensional embedding layer
Embedding layer -> LSTM layer -> Manhattan similarity measure
Objective: Run Siamese LSTM for both endings and choose ending sentence with highest Manhattan similarity measure 
"""


class SiameseLSTM():

    def __init__(self, train_generator, validation_generator = [], path=None):
        '''

        :param train_generator: Input array of the form([context, ending], [verifier]) for training set
        :param validation_generator: Input array of the form([context, ending], [verifier]) for validation set
        :param path: Path to the model to restore for predictions
        '''
        # Model variables
        self.train_generator = train_generator
        self.validation_generator = validation_generator
        self.n_hidden = n_hidden
        self.gradient_clipping_norm = gradient_clipping_norm
        self.batch_size = batch_size
        self.n_epoch = n_epoch
        self.story_len = 45
        self.embedding_dim = embedding_dim

        n_gram_size = 5
        vocabulary_size_tags = 45
        self.vocabulary_size = vocabulary_size

        """Loading trained model for predicting"""
        if path:
            print("Loading existing model from {}".format(path))
            self.load(path)
            print("Finished loading model")
            return


        # TODO Make sure inputs are: lemmatized, no stop words, padded and embedded

        # TODO Make sure that  input is of the form ([context, ending], [verifier])
        print("Train_generator[1]: {}".format(train_generator[1]))
        print("Train_generator[2]: {}".format(train_generator[2]))
        self.X = self.train_generator[1]
        self.Y = self.train_generator[2]

        # Check that shapes and sizes match
        assert self.X.shape == self.Y.shape
        assert len(self.X) == len(self.Y)

        # Visible layer of input
        self.X_input = Input(shape=(story_len,), dtype='int32', name='X_Context')
        self.Y_input = Input(shape=(story_len,), dtype='int32', name = 'Y_Ending')

        # Add embedding layer
        embedding_layer = Embedding(len(embeddings),self.embedding_dim, weights=[embeddings],
                                    input_length=story_len, trainable=False, name='embedding_layers')

        # TODO Get embedded version of inputs
        self.X_encoded = embedding_layer(self.X)
        self.Y_encoded = embedding_layer(self.Y)

        # Since we have a siamese network, both sides share the same LSTM
        self.shared_lstm = LSTM(n_hidden, activation='relu', name='LSTM_1_2')

        # Possible parameters for LSTM
        # keras.layers.LSTM(units, activation='tanh', recurrent_activation='hard_sigmoid',
        #                   use_bias=True, kernel_initializer='glorot_uniform',
        #                   recurrent_initializer='orthogonal', bias_initializer='zeros',
        #                   unit_forget_bias=True, kernel_regularizer=None,
        #                   recurrent_regularizer=None, bias_regularizer=None,
        #                   activity_regularizer=None, kernel_constraint=None, recurrent_constraint=None,
        #                   bias_constraint=None, dropout=0.0, recurrent_dropout=0.0, implementation=1,
        #                   return_sequences=False, return_state=False, go_backwards=False, stateful=False, unroll=False)


        self.X_output = shared_lstm(self.X_encoded)
        self.Y_output = shared_lstm(self.Y_encoded)

        # Calculates the distance as defined by the MaLSTM model
        manhattan_distance = Merge(mode=lambda x: exponent_neg_manhattan_distance(x[0], x[1]),
                                output_shape=lambda x: (x[0][0], 1))([self.X_output, self.Y_output])

        # Making the model
        model = Model(inputs=[self.X_input, self.Y_input], outputs=[manhattan_distance])

        # TODO Check which optimizer works best
        # Adadelta optimizer, with gradient clipping by norm
        # optimizer = Adadelta(clipnorm=gradient_clipping_norm)
        optimizer = tf.keras.optimizers.Adam()

        model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['accuracy'])
        model.summary()
        shared_model.summary()
        print(model_summary)


    # Taken From CNN_ngrams
    def train(self):
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
            os.path.join(out_trained_models, 'siameseLSTM/weights.h5'),
            monitor='val_loss', verbose=0, save_best_only=True,
            save_weights_only=False, mode='auto', period=1)

        # TODO train generator + validation generator to be implemented once preprocessing is ready

        # TODO Need self.train_generator as [X_train, Y_train], Ver_train
        print("Length train_generator: {}".format(len(self.train_generator)))
        print("Length val_generator: {}".format(len(self.validation_generator)))

        training_start_time = time()
        self.model.fit_generator(self.train_generator,
                                 steps_per_epoch=88161,
                                 verbose=2,
                                 epochs=n_epoch,
                                 callbacks=[lr_callback, stop_callback, tensorboard_callback,
                                            checkpoint_callback],
                                 validation_data=self.validation_generator,
                                 validation_steps=1871)
                                # TODO add batch size? (batch_size=batch_size)

        training_end_time = time()
        print("Training time finished.\n%d epochs in %12.2f" % (n_epoch,
                                                                training_end_time - training_start_time))

        model.save('./data/SiameseLSTM.h5')


    def exponent_neg_manhattan_distance(self, X, Y):
        """
        Helper function used to estimate similarity between the LSTM outputs
        :param X:  output of LSTM with input X
        :param Y:  output of LSTM with input Y
        :return: Manhattan distance between input vectors
        """
        return K.exp(-K.sum(K.abs(X - Y), axis=1, keepdims=True))


    # TODO Verify (copied from cnn_ngrams)

    def training(self):
        '''

        :return:
        '''

        for i in range(0, 3):
            model.fit([X_train, Y_train], sol_train,
                      batch_size=batch_size, epochs=n_epoch,
                      validation_data=([X_validation, Y_validation], sol_validation))
            malstm_trained = model.fit([X_train['left'], X_train['right']], y_train, batch_size=batch_size,
                                       epochs=n_epoch,
                                       validation_data=([X_val['left'], X_val['right']], y_val))
            model.save_weights("model30_relu_epoch_{}.h5".format(i + 1))


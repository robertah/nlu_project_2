import os
import keras
from keras.models import load_model
#from ... import config

class CNN_ngrams():
    """CNN model implementing a classifier using leaky ReLU and dropouts.
       INPUT -> sentence + pos tags as follows [(I,Sbj),(am,Verb),(bored,Adj)]
       """

    def __init__(self, train_generator, validation_generator = [], path=None):
        """Initialise the model.

        If `path` is given, the model is loaded from memory instead of compiled.
        """
        self.train_generator = train_generator
        self.validation_generator = validation_generator
        self.n_gram_size = 10
        self.stride_size = 1

        """Loading trained model for predicting"""
        if path:
            print("Loading existing model from {}".format(path))
            self.load(path)
            print("Finished loading model")
            return

        #TODO train generator + validation generator to be implemented once preprocessing is ready

        """Model creation"""
        self.model = keras.models.Sequential()
        
        """Blocks of layers
           First block"""
        self.model.add(keras.layers.Convolution2D(filters=100,
                                                  strides=stride_size,
                                                  kernel_size=(n_gram_size, 2),
                                                  padding="same",
                                                  input_shape=train_generator.input_dim()))
        self.model.add(keras.layers.MaxPooling2D(pool_size=(n_gram_size, n_gram_size),
                                                 strides=stride_size,
                                                 padding="same"))
        self.model.add(keras.layers.LeakyReLU(alpha=0.1))
        self.model.add(keras.layers.Dropout(rate=0.25))

        """Second block"""
        self.model.add(keras.layers.Convolution2D(filters=100,
                                                  strides=stride_size,
                                                  kernel_size=(n_gram_size, 2),
                                                  padding="same"))
        self.model.add(keras.layers.LeakyReLU(alpha=0.1))
        self.model.add(keras.layers.MaxPooling2D(pool_size=(n_gram_size, n_gram_size),
                                                 strides=stride_size,
                                                 padding="same"))
        self.model.add(keras.layers.Dropout(rate=0.25))

        """Third block"""
        self.model.add(keras.layers.Convolution2D(filters=100,
                                                  strides=stride_size,
                                                  kernel_size=(n_gram_size, n_gram_size),
                                                  padding="same"))
        self.model.add(keras.layers.LeakyReLU(alpha=0.1))
        self.model.add(keras.layers.MaxPooling2D(pool_size=(n_gram_size, n_gram_size),
                                                 strides=stride_size,
                                                 padding="same"))
        self.model.add(keras.layers.Dropout(rate=0.25))

        """Fourth block"""
        self.model.add(keras.layers.Convolution2D(filters=100,
                                                  strides=stride_size,
                                                  kernel_size=(n_gram_size, n_gram_size),
                                                  padding="same"))
        self.model.add(keras.layers.LeakyReLU(alpha=0.1))
        self.model.add(keras.layers.MaxPooling2D(pool_size=(n_gram_size, n_gram_size),
                                                 strides=stride_size,
                                                 padding="same"))
        self.model.add(keras.layers.Dropout(rate=0.25))

        """Fifth block -> dense layers + out layer"""
        self.model.add(keras.layers.Flatten())
        self.model.add(keras.layers.Dense(units=150,
                                          kernel_regularizer=keras.regularizers.l2(1e-6)))
        self.model.add(keras.layers.LeakyReLU(alpha=0.1))
        self.model.add(keras.layers.Dropout(rate=0.5))

        self.model.add(keras.layers.Dense(units=2,
                                          kernel_regularizer=keras.regularizers.l2(1e-6),
                                          activation="softmax"))


        optimiser = keras.optimizers.Adam()
        self.model.compile(loss=keras.losses.categorical_crossentropy,
                           optimizer=optimiser,
                           metrics=["accuracy"])
        print(self.model.summary())



    def train(self, epochs=100):
        """Train the model.

        Args:
            epochs (int): default: 100 - epochs to train.
            steps (int): default: len(dataset) - batches per epoch to train.
        """


        lr_callback = keras.callbacks.ReduceLROnPlateau(monitor="acc",
                                                        factor=0.5,
                                                        patience=0.5,
                                                        verbose=0,
                                                        min_delta=0.0001,
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
            os.path.join(out_trained_models, 'cnn_ngrams/weights.h5'),
            monitor='val_loss', verbose=0, save_best_only=True,
            save_weights_only=False, mode='auto', period=1)
        
        steps = len(self.train_generator.dataset())
        #TODO train generator + validation generator to be implemented once preprocessing is ready
        self.model.fit_generator(self.train_generator.generate_patch(),
                                 steps_per_epoch=steps,
                                 verbose=True,
                                 epochs=epochs,
                                 callbacks=[lr_callback, stop_callback, tensorboard_callback,
                                            checkpoint_callback],
                                 validation_data=self.validation_generator.generate_patch(),
                                 validation_steps=100)

    def save(self, path):
        """Save the model of the trained model.

        Args:
            path (path): path for the model file.
        """
        self.model.save(path)
        print("Model saved to {}".format(path))

import keras_tuner
from keras import Sequential
from keras.src.layers import Dropout, Conv1D, BatchNormalization, Dense, Flatten, AveragePooling1D
from keras.src.regularizers import l2
from keras_core.src.initializers import RandomNormal
import tensorflow as tf


class CustomEarlyStopping(tf.keras.callbacks.Callback):
    def __init__(self, accuracy_threshold, patience=0):
        super(CustomEarlyStopping, self).__init__()
        self.accuracy_threshold = accuracy_threshold
        self.patience = patience
        self.wait = 0  # Counter for patience
        self.threshold_count = 0  # Counter for patience

    def on_epoch_end(self, epoch, logs=None):
        current_accuracy = logs.get('accuracy')  # Use 'accuracy' or 'acc' based on your metric name
        if current_accuracy is None:
            raise ValueError(
                "Accuracy is not found in logs. Make sure the model is compiled with accuracy as a metric.")

        if current_accuracy >= self.accuracy_threshold:
            self.threshold_count += 1
            if self.threshold_count >= 2:
                self.model.stop_training = True
                print(f"\nReached accuracy threshold ({self.accuracy_threshold}). Training stopped.")
        else:
            self.threshold_count = 0
            self.wait += 1
            if self.wait >= self.patience:
                self.model.stop_training = True
                print(f"\nAccuracy did not reach threshold in {self.patience} epochs. Training stopped.")


# THIS IS OUR CNN MODEL
def train_best_model(train_x, train_y, test_x, test_y):
    model = Sequential()

    # start temporal convolution
    model.add(Conv1D(filters=64,
                     data_format="channels_first",
                     kernel_size=3,
                     activation="relu",
                     kernel_regularizer=l2(0.1),
                     kernel_initializer=RandomNormal(),
                     input_shape=(train_x.shape[1], train_x.shape[2])))
    model.add(AveragePooling1D(
        pool_size=2,
        data_format="channels_first"))
    model.add(BatchNormalization())

    model.add(Conv1D(
        filters=32,
        data_format="channels_first",
        activation='relu',
        kernel_size=3))
    model.add(AveragePooling1D(
        pool_size=2,
        data_format="channels_first"))
    model.add(BatchNormalization())

    model.add(Conv1D(
        filters=1,
        data_format="channels_first",
        activation='relu',
        kernel_size=3))

    # start spacial convolution
    model.add(Conv1D(
        filters=64,
        kernel_regularizer=l2(0.1),
        data_format="channels_last",
        activation='relu',
        kernel_size=3,
        strides=1,
        padding="same"))

    model.add(Conv1D(
        filters=32,
        data_format="channels_last",
        activation='relu',
        kernel_size=3,
        strides=1,
        padding="same"))

    # start classification
    model.add(Flatten())
    model.add(Dense(32, activation="relu"))
    model.add(Dropout(0.1))
    model.add(Dense(16, activation="relu"))
    model.add(Dense(4, activation="softmax"))

    # print(model.summary())

    accuracy_threshold = 1.0
    custom_early_stopping = CustomEarlyStopping(accuracy_threshold=accuracy_threshold, patience=50)

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    model.fit(train_x, train_y, epochs=50, batch_size=16, verbose=0, callbacks=[custom_early_stopping], validation_data=(test_x, test_y))

    # This line is used to clear GPU memory.
    model.save("temp")

    return model


def tune_hyperparameters(train_x, train_y, test_x, test_y):
    def build_model(hp):
        model = Sequential()

        # start temporal convolution
        model.add(Conv1D(filters=hp.Choice('l1_filters', [32, 64, 128]),
                         data_format="channels_first",
                         kernel_size=3,
                         activation="relu",
                         kernel_regularizer=l2(0.1),
                         kernel_initializer=RandomNormal(),
                         input_shape=(train_x.shape[1], train_x.shape[2])))
        model.add(AveragePooling1D(
            pool_size=2,
            data_format="channels_first"))
        model.add(BatchNormalization())

        model.add(Conv1D(
            filters=hp.Choice('l2_filters', [16, 32, 64]),
            data_format="channels_first",
            activation='relu',
            kernel_size=3))
        model.add(AveragePooling1D(
            pool_size=2,
            data_format="channels_first"))
        model.add(BatchNormalization())

        model.add(Conv1D(
            filters=1,
            data_format="channels_first",
            activation='relu',
            kernel_size=3))

        # start spacial convolution
        model.add(Conv1D(
            filters=hp.Choice('l3_filters', [16, 32, 64]),
            kernel_regularizer=l2(0.1),
            data_format="channels_last",
            activation='relu',
            kernel_size=3,
            strides=1,
            padding="same"))

        model.add(Conv1D(
            filters=hp.Choice('l4_filters', [8, 16, 32]),
            data_format="channels_last",
            activation='relu',
            kernel_size=3,
            strides=1,
            padding="same"))

        # start classification
        model.add(Flatten())
        model.add(Dense(hp.Choice('l5_units', [16, 32]), activation="relu"))
        model.add(Dropout(0.1))
        model.add(Dense(hp.Choice('l7_units', [8, 16]), activation="relu"))
        model.add(Dense(4, activation="softmax"))

        model.compile(optimizer='adam',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
        return model

    tuner = keras_tuner.RandomSearch(
        build_model,
        max_trials=600,
        objective='accuracy',
        metrics=['accuracy'])

    tuner.search(train_x, train_y, epochs=10, batch_size=32)

    best_hps = tuner.get_best_hyperparameters(5)
    # Build the model with the best hp.
    model = build_model(best_hps[0])
    model.fit(train_x, train_y, epochs=1)

    return model

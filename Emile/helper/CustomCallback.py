import numpy as np
from keras.src.callbacks import Callback
import tensorflow as tf
from sklearn.metrics import accuracy_score


class CustomEarlyStopping(Callback):
    def __init__(self, test_acc, x, y):
        super(CustomEarlyStopping, self).__init__()
        self.test_acc = test_acc
        self.x = x
        self.y = y

    def on_epoch_end(self, epoch, logs=None):
        # Generate predictions on the test set
        predictions = self.model.predict(self.x, batch_size=32, verbose=0)

        # Print report
        predicted_classes = np.argmax(predictions, axis=1)  # Get the index of class with highest prob
        true_classes = np.argmax(self.y, axis=1)  # Convert one-hot to index
        accuracy_score(true_classes, predicted_classes)

        self.test_acc.append(accuracy_score(true_classes, predicted_classes))
        tf.keras.backend.clear_session()

import os

import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix


def create_classification_report(model, test_x, test_y):
    # Generate predictions on the test set
    with tf.device('/GPU:0'):
        predictions = model.predict(test_x, batch_size=64, verbose=0)

    # Print report
    predicted_classes = np.argmax(predictions, axis=1) # Get the index of class with highest prob
    true_classes = np.argmax(test_y, axis=1) # Convert one-hot to index

    accuracy = accuracy_score(true_classes, predicted_classes)

    # return (None, accuracy)
    return (confusion_matrix(true_classes, predicted_classes), accuracy)

def analyze_train_history(model_name, history):
    # Plotting the training accuracy
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Test Accuracy')

    # Adding labels and title
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy per Epoch')
    plt.legend()

    plt.ylim(0, 1.1)
    plt.yticks(np.arange(0, 1.2, 0.1))

    # Create the folder if it doesn't exist
    folder_path = f"Emile/results/{model_name}"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    plt.savefig(os.path.join(folder_path, "epoch_analysis.png"))

    plt.clf()
    return

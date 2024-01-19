import os
import shutil

# disable all tensorflow warnings/info and use better memory allocator
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
# os.environ['CUDA_VISIBLE_DEVICES'] = ''

from Emile.helper.analyze_run import create_classification_report
from Emile.models.CNN import train_best_model, tune_hyperparameters

from Emile.models.LSTM import train_lstm
from data.data_preparation import get_intra_subject_data, get_cross_subject_data
import numpy as np
from data.pre_processing import preprocess_data
train_hyperparameters = False

def train_models(train_x, train_y, test_x, test_y):
    if train_hyperparameters:
        best_tuned_model = tune_hyperparameters(train_x, train_y, test_x, test_y)
        cnn_report = create_classification_report(best_tuned_model, test_x, test_y)
        print(cnn_report)
    else:
        training_runs = 20
        accuracies_per_category = np.zeros((training_runs, 4))

        for run in range(training_runs):
            print(f'Run: {run+1}')
            model = train_best_model(train_x, train_y, test_x, test_y)
            # model = train_lstm(train_x, train_y, test_x, test_y)
            (cnn_report, accuracy) = create_classification_report(model, test_x, test_y)

            accuracy_per_category = cnn_report.diagonal() / cnn_report.sum(axis=1)
            print(accuracy)
            print(accuracy_per_category)
            accuracies_per_category[run,:] = accuracy_per_category

        print(f'Completed {training_runs} runs:')

        mean_accuracy_category = np.mean(accuracies_per_category, axis=0)
        std_deviation_category = np.std(accuracies_per_category, axis=0)

        print(f'Mean accuracy: {np.mean(mean_accuracy_category):.2f}')
        print(f'Standard Deviation: {np.mean(std_deviation_category):.2f}')
        mean_cat = ['{:.2f}'.format(f) for f in mean_accuracy_category]
        std_cat = ['{:.2f}'.format(f) for f in std_deviation_category]
        print(f'Mean accuracy per category: {mean_cat}')
        print(f'Standard Deviation per category: {std_cat}')


if __name__ == '__main__':
    # uncomment following to redo the data preprocessing
    preprocess_data()

    if os.path.exists("untitled_project"):
        shutil.rmtree("untitled_project")

    # train_x, train_y, test_x, test_y = get_intra_subject_data()
    train_x, train_y, test_x, test_y = get_cross_subject_data()

    train_models(train_x, train_y, test_x, test_y)
import os
import h5py
import numpy as np
from Emile.data.pre_processing import get_dataset_name


def get_intra_subject_data():
    labels = {'rest': 0, 'task_story_math': 1, 'task_working_memory': 2, 'task_motor': 3}

    # Arrays to store data and labels
    train_x = []
    train_y = []
    test_x = []
    test_y = []

    # Get training data and labels
    dir_path = 'Preprocessed data/Intra/train'
    for file in os.listdir(f'{dir_path}'):
        file_path = f'{dir_path}/{file}'
        with h5py.File(file_path, 'r') as f:
            name = get_dataset_name(file_path)

            # Get the label
            for key in labels:
                if key in name:
                    one_hot_label = np.zeros(4)
                    one_hot_label[labels[key]] = 1
                    train_y.append(one_hot_label)

            # Get the matrix
            matrix = f.get(name)[()]
            train_x.append(matrix)

    # Get test data and labels
    dir_path = 'Preprocessed data/Intra/test'
    for file in os.listdir(f'{dir_path}'):
        file_path = f'{dir_path}/{file}'
        with h5py.File(file_path, 'r') as f:
            name = get_dataset_name(file_path)

            # Get the label
            for key in labels:
                if key in name:
                    one_hot_label = np.zeros(4)
                    one_hot_label[labels[key]] = 1
                    test_y.append(one_hot_label)

            # Get the matrix
            matrix = f.get(name)[()]
            test_x.append(matrix)

    # Convert to np arrays and reshape for compatibility with model
    train_x = np.array(train_x).transpose(0, 2, 1)
    train_y = np.array(train_y)
    test_x = np.array(test_x).transpose(0, 2, 1)
    test_y = np.array(test_y)

    return train_x, train_y, test_x, test_y


def get_cross_subject_data():
    labels = {'rest': 0, 'task_story_math': 1, 'task_working_memory': 2, 'task_motor': 3}

    # Arrays to store data and labels
    train_x = []
    train_y = []
    test_x = []
    test_y = []

    # Get training data and labels
    dir_path = 'Preprocessed data/Cross/train'
    for file in os.listdir(f'{dir_path}'):
        file_path = f'{dir_path}/{file}'
        with h5py.File(file_path, 'r') as f:
            name = get_dataset_name(file_path)

            # Get the label
            for key in labels:
                if key in name:
                    one_hot_label = np.zeros(4)
                    one_hot_label[labels[key]] = 1
                    train_y.append(one_hot_label)

            # Get the matrix
            matrix = f.get(name)[()]
            train_x.append(matrix)

    # Get test data and labels
    for i in range(1, 4):
        dir_path = f'Preprocessed data/Cross/test{i}'
        for file in os.listdir(f'{dir_path}'):
            file_path = f'{dir_path}/{file}'
            with h5py.File(file_path, 'r') as f:
                name = get_dataset_name(file_path)

                # Get the label
                for key in labels:
                    if key in name:
                        one_hot_label = np.zeros(4)
                        one_hot_label[labels[key]] = 1
                        test_y.append(one_hot_label)

                # Get the matrix
                matrix = f.get(name)[()]
                test_x.append(matrix)

    # Convert to np arrays and reshape for compatibility with model
    train_x = np.array(train_x).transpose(0, 2, 1)
    train_y = np.array(train_y)
    test_x = np.array(test_x).transpose(0, 2, 1)
    test_y = np.array(test_y)

    return train_x, train_y, test_x, test_y
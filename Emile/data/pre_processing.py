import os
import h5py
from sklearn.preprocessing import StandardScaler


def get_dataset_name(file_name_with_dir):
    filename_without_dir = file_name_with_dir.split('/')[-1]
    temp = filename_without_dir.split('_')[:-1]
    dataset_name = "_". join(temp)
    return dataset_name


def preprocess_data():
    # Scaler for Z-normalization per sequence
    scaler = StandardScaler()

    # Downsampling factor
    downsampling_factor = 4

    # Loop through files in the folder
    for dir in ['Cross', 'Intra']:
        if dir == 'Cross':
            print("Preprocessing cross-subject data...")
            for subdir in ['test1', 'test2', 'test3', 'train']:
                for file in os.listdir(f'Final Project data/{dir}/{subdir}'):
                    file_path = f'Final Project data/{dir}/{subdir}/{file}'
                    with h5py.File(file_path, 'r') as f:
                        dataset_name = get_dataset_name(file_path)
                        matrix = f.get(dataset_name)[()]

                        # Normalize
                        normalized_matrix = scaler.fit_transform(matrix)

                        # Downsample
                        downsampled_matrix = normalized_matrix[:, ::downsampling_factor]

                        # Save
                        save_dir = f'Preprocessed data/{dir}/{subdir}'
                        if not os.path.exists(save_dir):
                            os.makedirs(save_dir)

                        with h5py.File(f'{save_dir}/{file}', 'w') as hf:
                            hf.create_dataset(get_dataset_name(f'{dir}/{subdir}/{file}'), data=downsampled_matrix)

        if dir == 'Intra':
            print("Preprocessing intra-subject data...")
            for subdir in ['test', 'train']:
                for file in os.listdir(f'Final Project data/{dir}/{subdir}'):
                    file_path = f'Final Project data/{dir}/{subdir}/{file}'
                    with h5py.File(file_path, 'r') as f:
                        dataset_name = get_dataset_name(file_path)
                        matrix = f.get(dataset_name)[()]

                        # Normalize
                        normalized_matrix = scaler.fit_transform(matrix)

                        # Downsample
                        downsampled_matrix = normalized_matrix[:, ::downsampling_factor]

                        # Save
                        save_dir = f'Preprocessed data/{dir}/{subdir}'
                        if not os.path.exists(save_dir):
                            os.makedirs(save_dir)

                        with h5py.File(f'{save_dir}/{file}]', 'w') as hf:
                            hf.create_dataset(get_dataset_name(f'{dir}/{subdir}/{file}'), data=downsampled_matrix)

        print("Done")
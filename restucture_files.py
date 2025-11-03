import os
import shutil
from tqdm import tqdm
import random

def get_split_indices(splits, length):
    indices = [i for i in range(length)]
    random.shuffle(indices)
    train_size = 56604
    val_size = 7076
    return [indices[0:train_size], indices[train_size:train_size+val_size], indices[train_size+val_size::]]

def split(main_folder, splits, length):
    indices_splits = get_split_indices(splits, length)
    types = ['train', 'val', 'test']
    modalities = ['brt', 'dsm', 'ortho']
    for type, indices in zip(types, indices_splits):
        for modality in modalities:
            if not os.path.exists(os.path.join(main_folder, type, modality)):
                os.makedirs(os.path.join(main_folder, type, modality))
            for index in tqdm(indices, desc=f'Filling {type} set for modality: {modality}'):
                input_file = os.path.join(main_folder, modality, f'{index}.tif')
                output_file = os.path.join(main_folder, type, modality, f'{index}.tif')
                shutil.copy(input_file, output_file)

def group_data(global_directory):
    modalities = ['brt', 'dsm', 'ortho']
    sub_directories = [os.path.join(global_directory, item) for item in os.listdir(global_directory)
                       if os.path.isdir(os.path.join(global_directory, item))]
    assert len(sub_directories) == len(modalities)
    for sub_directory in sub_directories:
        for modality in modalities:
            modality_dir = os.path.join(sub_directory, modality)
            target_dir = os.path.join(global_directory, modality)
            if not os.path.exists(target_dir):
                os.makedirs(target_dir)
            files = os.listdir(modality_dir)
            for file in tqdm(files, f'Processing {modality} of directory {sub_directory}'):
                input_file = os.path.join(modality_dir, file)
                suffix = os.path.splitext(file)[-1]
                index = len(os.listdir(target_dir))
                output_file = os.path.join(target_dir, f'{index}{suffix}')
                shutil.copy(input_file, output_file)
    return len(os.listdir(os.path.join(global_directory, modalities[0])))

if __name__ == "__main__":
    global_directory = "C:/MTP-Data/dataset_diverse_2022_512"
    # data_size = group_data(global_directory)
    split(global_directory, [0.8, 0.1, 0.1], 70756)




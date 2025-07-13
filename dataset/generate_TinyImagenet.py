import numpy as np
import os
import sys
import random
import torch
import torchvision
import torchvision.transforms as transforms
from utils.dataset_utils import check, separate_data, split_data, save_file
from torchvision.datasets import ImageFolder, DatasetFolder
from PIL import Image
import argparse

random.seed(1)
np.random.seed(1)
num_clients = 20
dir_path = "TinyImagenet/"

# https://github.com/QinbinLi/MOON/blob/6c7a4ed1b1a8c0724fa2976292a667a828e3ff5d/datasets.py#L148
class ImageFolder_custom(DatasetFolder):
    def __init__(self, root, dataidxs=None, train=True, transform=None, target_transform=None):
        self.root = root
        self.dataidxs = dataidxs
        self.train = train
        self.transform = transform
        self.target_transform = target_transform

        imagefolder_obj = ImageFolder(self.root, self.transform, self.target_transform)
        self.loader = imagefolder_obj.loader
        if self.dataidxs is not None:
            self.samples = np.array(imagefolder_obj.samples)[self.dataidxs]
        else:
            self.samples = np.array(imagefolder_obj.samples)

    def __getitem__(self, index):
        path = self.samples[index][0]
        target = self.samples[index][1]
        target = int(target)
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __len__(self):
        if self.dataidxs is None:
            return len(self.samples)
        else:
            return len(self.dataidxs)


# Allocate data to users
def generate_dataset(dir_path, num_clients, niid, balance, partition, task_type=None, classes_per_task=None): # Added task_type, classes_per_task
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        
    # Setup directory for train/test data
    config_path = dir_path + "config.json"
    train_path = dir_path + "train/"
    test_path = dir_path + "test/"

    # Check if dataset already generated, including task_type and classes_per_task
    if check(config_path, train_path, test_path, num_clients, niid, balance, partition, task_type, classes_per_task):
        return

    # Get data
    if not os.path.exists(f'{dir_path}/rawdata/'):
        print(f'Downloading Tiny-ImageNet to {dir_path}/rawdata/')
        # Use subprocess.run for better control and error handling
        import subprocess
        try:
            subprocess.run(['wget', '--directory-prefix', f'{dir_path}/rawdata/', 'http://cs231n.stanford.edu/tiny-imagenet-200.zip'], check=True)
            subprocess.run(['unzip', f'{dir_path}/rawdata/tiny-imagenet-200.zip', '-d', f'{dir_path}/rawdata/'], check=True)
            # Tiny ImageNet requires a validation data preprocessing step
            # Create train-like structure for val data for ImageFolder
            val_dir = f'{dir_path}/rawdata/tiny-imagenet-200/val/'
            with open(os.path.join(val_dir, 'val_annotations.txt'), 'r') as f:
                for line in f.readlines():
                    parts = line.strip().split('\t')
                    img_name, class_id = parts[0], parts[1]
                    class_dir = os.path.join(val_dir, class_id)
                    os.makedirs(class_dir, exist_ok=True)
                    os.rename(os.path.join(val_dir, 'images', img_name), os.path.join(class_dir, img_name))
            os.rmdir(os.path.join(val_dir, 'images')) # Remove empty images folder
        except subprocess.CalledProcessError as e:
            print(f"Error during TinyImageNet download or unzip: {e}")
            sys.exit(1) # Exit if download/unzip fails
    else:
        print('rawdata already exists.\n')

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # TinyImageNet has train and val folders
    trainset = ImageFolder_custom(root=dir_path+'rawdata/tiny-imagenet-200/train/', transform=transform)
    # The original MOON code uses ImageFolder for train and a custom DatasetFolder for val/test if dataidxs are passed.
    # For initial data generation, we combine train and val as the full dataset.
    valset = ImageFolder_custom(root=dir_path+'rawdata/tiny-imagenet-200/val/', transform=transform) # Use val as test data source
    
    # TinyImageNet needs custom handling to get data and targets from ImageFolder
    # It stores data as samples = [(path, class_idx), ...]
    # We need to load images and convert to numpy array and targets to numpy array
    
    dataset_image = []
    dataset_label = []

    # Process trainset
    for img_path, label in trainset.samples:
        img = Image.open(img_path).convert('RGB')
        dataset_image.append(np.array(img))
        dataset_label.append(label)
    
    # Process valset (treating as part of overall data for partitioning)
    for img_path, label in valset.samples:
        img = Image.open(img_path).convert('RGB')
        dataset_image.append(np.array(img))
        dataset_label.append(label)

    dataset_image = np.array(dataset_image)
    dataset_label = np.array(dataset_label)


    num_classes = len(set(dataset_label))
    print(f'Number of classes: {num_classes}')

    # Call separate_data with new parameters and retrieve new return values
    # For TinyImageNet, class_per_client=20 might be suitable for Non-IID
    X, y, statistic, client_task_data, num_tasks, task_classes_map = separate_data(
        (dataset_image, dataset_label), 
        num_clients, 
        num_classes, 
        niid, 
        balance, 
        partition, 
        class_per_client=20, # For I.2/I.3, adjust as needed. For I.1, this is ignored.
        task_type=task_type, 
        classes_per_task=classes_per_task # For II.3
    )
    
    train_data, test_data = split_data(X, y)
    
    # Save the new task-related information
    save_file(config_path, train_path, test_path, train_data, test_data, num_clients, num_classes, 
        statistic, niid, balance, partition, task_type, classes_per_task, num_tasks, task_classes_map, client_task_data)


if __name__ == "__main__":
    # Use argparse for robust parsing instead of sys.argv directly
    parser = argparse.ArgumentParser()
    parser.add_argument('niid_str', type=str, help="'noniid' or 'iid'")
    parser.add_argument('balance_str', type=str, help="'True' or 'False'")
    parser.add_argument('partition_str', type=str, help="Partition type, e.g., 'dir', 'pat', 'no_overlap'")
    parser.add_argument('task_type_str', type=str, help="Task type, e.g., 'class_incremental', 'iid_tasks'")
    parser.add_argument('classes_per_task_val', type=str, nargs='?', default=None,
                        help="Number of classes per task for incremental learning (optional)")
    
    args_gen = parser.parse_args() # Use a different variable name to avoid conflict with main.py args

    niid = True if args_gen.niid_str == "noniid" else False
    balance = True if args_gen.balance_str == "True" else False
    partition = args_gen.partition_str if args_gen.partition_str != "-" else None
    
    task_type = args_gen.task_type_str if args_gen.task_type_str != "-" else 'iid_tasks'
    classes_per_task = int(args_gen.classes_per_task_val) if args_gen.classes_per_task_val else None

    # Original call
    generate_dataset(dir_path, num_clients, niid, balance, partition, task_type, classes_per_task)
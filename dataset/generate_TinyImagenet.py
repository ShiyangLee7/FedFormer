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
import requests
import zipfile
from tqdm import tqdm

random.seed(2026)
np.random.seed(1)
num_clients = 10
dir_path = "TinyImagenet/"

def download_file(url, save_path):
    """Download file from url with progress bar"""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024  # 1 KB
    progress_bar = tqdm(total=total_size, unit='iB', unit_scale=True)
    
    with open(save_path, 'wb') as f:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            f.write(data)
    progress_bar.close()

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
    rawdata_path = dir_path + "rawdata/"
    
    # Check if dataset already generated, including task_type and classes_per_task
    if check(config_path, train_path, test_path, num_clients, niid, balance, partition, task_type, classes_per_task):
        return
        
    # Download Tiny-ImageNet if not exists
    zip_file_path = os.path.join(rawdata_path, 'tiny-imagenet-200.zip')
    if not os.path.exists(zip_file_path):
        print("Downloading Tiny-ImageNet...")
        os.makedirs(rawdata_path, exist_ok=True)
        try:
            download_file('http://cs231n.stanford.edu/tiny-imagenet-200.zip', zip_file_path)
        except Exception as e:
            print(f"Error downloading file: {e}")
            print("Please download tiny-imagenet-200.zip manually from http://cs231n.stanford.edu/tiny-imagenet-200.zip")
            print(f"and place it in {rawdata_path}")
            return
    
    # Extract the zip file if not already extracted
    extract_path = os.path.join(rawdata_path, 'tiny-imagenet-200')
    if not os.path.exists(extract_path):
        print("Extracting Tiny-ImageNet...")
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            zip_ref.extractall(rawdata_path)

    # Tiny ImageNet requires a validation data preprocessing step
    # Create train-like structure for val data for ImageFolder
    val_dir = os.path.join(extract_path, 'val')
    val_images_dir = os.path.join(val_dir, 'images')
    val_annotations_file = os.path.join(val_dir, 'val_annotations.txt')

    # Only process validation data if it's in the original structure
    if os.path.exists(val_images_dir) and os.path.exists(val_annotations_file):
        print("Processing validation data...")
        try:
            with open(val_annotations_file, 'r') as f:
                for line in f.readlines():
                    parts = line.strip().split('\t')
                    img_name, class_id = parts[0], parts[1]
                    class_dir = os.path.join(val_dir, class_id)
                    os.makedirs(class_dir, exist_ok=True)
                    
                    # Check if the image is still in the original location
                    src_path = os.path.join(val_images_dir, img_name)
                    dst_path = os.path.join(class_dir, img_name)
                    
                    if os.path.exists(src_path):
                        os.rename(src_path, dst_path)
            
            # Only try to remove the images directory if it exists and is empty
            if os.path.exists(val_images_dir) and not os.listdir(val_images_dir):
                os.rmdir(val_images_dir)
                print("Validation data processing completed.")
            
        except Exception as e:
            print(f"Error processing validation data: {e}")
            print("This might not be an issue if the data was already processed.")
    else:
        print("Validation data appears to be already processed.")

    # Set up transforms with resize for TinyImageNet's 64x64 images
    transform = transforms.Compose([
        transforms.Resize((64, 64)),  # Ensure consistent size
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Load datasets with error handling
    try:
        trainset = ImageFolder_custom(root=os.path.join(extract_path, 'train'), transform=transform)
        valset = ImageFolder_custom(root=os.path.join(extract_path, 'val'), transform=transform)
    except Exception as e:
        print(f"Error loading datasets: {e}")
        return
    
    print(f"Loaded {len(trainset.samples)} training samples and {len(valset.samples)} validation samples")
    
    dataset_image = []
    dataset_label = []

    # Process datasets with progress bars and error handling
    print("Processing training data...")
    for img_path, label in tqdm(trainset.samples, desc="Training data"):
        try:
            img = Image.open(img_path).convert('RGB')
            img = img.resize((64, 64))  # Ensure consistent size
            dataset_image.append(np.array(img))
            dataset_label.append(label)
        except Exception as e:
            print(f"Error processing image {img_path}: {e}")
            continue
    
    print("Processing validation data...")
    for img_path, label in tqdm(valset.samples, desc="Validation data"):
        try:
            img = Image.open(img_path).convert('RGB')
            img = img.resize((64, 64))  # Ensure consistent size
            dataset_image.append(np.array(img))
            dataset_label.append(label)
        except Exception as e:
            print(f"Error processing image {img_path}: {e}")
            continue

    if not dataset_image:
        print("Error: No images were successfully processed")
        return

    dataset_image = np.array(dataset_image)
    dataset_label = np.array(dataset_label)

    num_classes = len(set(dataset_label))
    print(f'Number of classes: {num_classes}')
    print(f'Total number of images: {len(dataset_image)}')

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
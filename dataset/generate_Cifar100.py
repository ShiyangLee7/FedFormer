import numpy as np
import os
import sys
import random
import torch
import torchvision
import torchvision.transforms as transforms
from utils.dataset_utils import check, separate_data, split_data, save_file
import argparse


random.seed(2026)
np.random.seed(1)
num_clients = 10  # Initial active clients
reserve_clients = 5  # Additional clients that can join later
total_clients = num_clients + reserve_clients  # Total number of clients to generate data for
dir_path = "Cifar100/"


# Allocate data to users
def generate_dataset(dir_path, num_clients, niid, balance, partition, task_type=None, classes_per_task=None): # Added task_type, classes_per_task
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        
    # Setup directory for train/test data
    config_path = dir_path + "config.json"
    train_path = dir_path + "train/"
    test_path = dir_path + "test/"

    # Use total_clients instead of num_clients for data generation
    total_clients_for_data = num_clients + reserve_clients

    # Check if dataset already generated, including task_type and classes_per_task
    if check(config_path, train_path, test_path, total_clients_for_data, niid, balance, partition, task_type, classes_per_task):
        return
        
    # Get Cifar100 data
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR100(
        root=dir_path+"rawdata", train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR100(
        root=dir_path+"rawdata", train=False, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=len(trainset.data), shuffle=False)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=len(testset.data), shuffle=False)

    for _, train_data in enumerate(trainloader, 0):
        trainset.data, trainset.targets = train_data
    for _, test_data in enumerate(testloader, 0):
        testset.data, testset.targets = test_data

    dataset_image = []
    dataset_label = []

    dataset_image.extend(trainset.data.cpu().detach().numpy())
    dataset_image.extend(testset.data.cpu().detach().numpy())
    dataset_label.extend(trainset.targets.cpu().detach().numpy())
    dataset_label.extend(testset.targets.cpu().detach().numpy())
    dataset_image = np.array(dataset_image)
    dataset_label = np.array(dataset_label)

    num_classes = len(set(dataset_label))
    print(f'Number of classes: {num_classes}')

    # Call separate_data with total_clients and new parameters
    X, y, statistic, client_task_data, num_tasks, task_classes_map = separate_data(
        (dataset_image, dataset_label), 
        total_clients_for_data,  # Use total_clients instead of num_clients
        num_classes, 
        niid, 
        balance, 
        partition, 
        class_per_client=10,
        task_type=task_type, 
        classes_per_task=classes_per_task
    )
    
    train_data, test_data = split_data(X, y)
    
    # Save with total_clients
    save_file(config_path, train_path, test_path, train_data, test_data, total_clients_for_data, num_classes, 
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
import os
import numpy as np
import ujson
import torch
from torch.utils.data import Dataset
import logging

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Add a function to get the project root directory
def get_project_root():
    """Get the project root directory."""
    # When running from system/main.py, we need to go up one level
    if os.path.basename(os.getcwd()) == 'system':
        return os.path.dirname(os.getcwd())
    return os.getcwd()

class CustomDataset(Dataset):
    def __init__(self, data, labels):
        self.data = torch.Tensor(data)
        self.labels = torch.LongTensor(labels)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

def read_config(dataset):
    """Read the configuration file for the dataset."""
    try:
        config_path = os.path.join(get_project_root(), 'dataset', dataset, 'config.json')
        logger.info(f"Looking for config at: {config_path}")
        if not os.path.exists(config_path):
            logger.error(f"Config file not found at {config_path}")
            return {}
        
        with open(config_path, 'r') as f:
            config = ujson.load(f)
            logger.info(f"Successfully loaded config for dataset {dataset}")
            logger.info(f"Config contents: {config}")
            return config
    except Exception as e:
        logger.error(f"Error reading config for dataset {dataset}: {str(e)}")
        return {}

def read_client_data(dataset, client_id, is_train=True, few_shot=0):
    """Read regular (non-task) client data."""
    try:
        if is_train:
            data_path = os.path.join(get_project_root(), 'dataset', dataset, 'train', f'{client_id}.npz')
        else:
            data_path = os.path.join(get_project_root(), 'dataset', dataset, 'test', f'{client_id}.npz')

        if not os.path.exists(data_path):
            logger.warning(f"Data file not found at {data_path}")
            return None

        data = np.load(data_path, allow_pickle=True)['data'].item()
        
        if few_shot > 0:  # If few-shot learning is enabled
            if len(data['y']) > few_shot:
                few_idx = np.random.choice(len(data['y']), few_shot, replace=False)
                data = {'x': data['x'][few_idx], 'y': data['y'][few_idx]}

        if len(data['x']) == 0:  # If no data available
            return None

        return CustomDataset(data['x'], data['y'])
    except Exception as e:
        logger.error(f"Error reading client data for {dataset}, client {client_id}: {str(e)}")
        return None

def read_client_task_data(dataset, client_id, task_id, is_train=True):
    """Read task-specific client data."""
    try:
        # For task data, we always read from the task_data directory
        if is_train:
            data_path = os.path.join(get_project_root(), 'dataset', dataset, 'train', 'task_data', f'client_{client_id}_task_{task_id}.npz')
        else:
            # For testing, we might want to use the regular test data filtered by task classes
            # But for now, we'll use the same task_data directory
            data_path = os.path.join(get_project_root(), 'dataset', dataset, 'train', 'task_data', f'client_{client_id}_task_{task_id}.npz')

        # Add absolute path debugging
        abs_path = os.path.abspath(data_path)
        logger.info(f"Attempting to read task data from {abs_path}")
        logger.info(f"Current working directory: {os.getcwd()}")
        
        if not os.path.exists(data_path):
            logger.warning(f"Task data file not found at {abs_path}")
            return None

        data = np.load(data_path, allow_pickle=True)
        logger.info(f"Successfully loaded npz file with keys: {data.files}")
        
        # The task data files store 'x' and 'y' directly, not in a nested dictionary
        x_data = data['x']
        y_data = data['y']

        if len(x_data) == 0:  # If no data available
            logger.warning(f"Empty data found for client {client_id}, task {task_id}")
            return None

        logger.info(f"Successfully loaded data for client {client_id}, task {task_id} with {len(x_data)} samples")
        logger.info(f"Data shapes - x: {x_data.shape}, y: {y_data.shape}")
        
        return CustomDataset(x_data, y_data)
    except Exception as e:
        logger.error(f"Error reading task data for {dataset}, client {client_id}, task {task_id}: {str(e)}")
        logger.error(f"Stack trace:", exc_info=True)
        return None
import copy
import torch
import torch.nn as nn
import numpy as np
import os
from torch.utils.data import DataLoader
from sklearn.preprocessing import label_binarize
from sklearn import metrics
# from utils.data_utils import read_client_data # Original import
from utils.data_utils import read_client_data, read_client_task_data, read_config # Updated import


class Client(object):
    """
    Base class for clients in federated learning.
    """

    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        torch.manual_seed(0)
        self.model = copy.deepcopy(args.model)
        self.algorithm = args.algorithm
        self.dataset = args.dataset
        self.device = args.device
        self.id = id  # integer
        self.save_folder_name = args.save_folder_name

        self.num_classes = args.num_classes
        self.train_samples = train_samples # Total samples for this client across all tasks
        self.test_samples = test_samples   # Total samples for this client across all tasks
        self.batch_size = args.batch_size
        self.learning_rate = args.local_learning_rate
        self.local_epochs = args.local_epochs
        self.few_shot = args.few_shot

        # New: Task-related attributes
        # Ensure config is read safely. This is called *per client*, could be optimized if config is truly global
        # but for now, keep it as is based on existing pattern.
        config = read_config(self.dataset) 
        self.task_type = config.get('task_type', 'iid_tasks')
        self.num_tasks = config.get('num_tasks', 1)
        # Ensure task_classes_map keys are integers, as they are saved as strings in JSON.
        self.task_classes_map = {int(k): v for k, v in config.get('task_classes_map', {}).items()} 

        # Stores DataLoaders for each task's test data for this client
        self.test_dataloaders_by_task = {} 
        # Stores DataLoaders for each task's train data for this client (for incremental learning)
        self.train_dataloaders_by_task = {}


        # check BatchNorm
        self.has_BatchNorm = False
        for layer in self.model.children():
            if isinstance(layer, nn.BatchNorm2d):
                self.has_BatchNorm = True
                break

        self.train_slow = kwargs['train_slow']
        self.send_slow = kwargs['send_slow']
        self.train_time_cost = {'num_rounds': 0, 'total_cost': 0.0}
        self.send_time_cost = {'num_rounds': 0, 'total_cost': 0.0}

        self.loss = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)
        self.learning_rate_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=self.optimizer, 
            gamma=args.learning_rate_decay_gamma
        )
        self.learning_rate_decay = args.learning_rate_decay

    # Modified to load data for a specific task
    def load_train_data(self, batch_size=None, current_task_id=0):
        """Load training data for the client, either task-specific or all data."""
        if batch_size is None:
            batch_size = self.batch_size
        
        # If already loaded for this task, return it
        if current_task_id in self.train_dataloaders_by_task:
            print(f"DEBUG: Client {self.id} using cached dataloader for task {current_task_id}")
            return self.train_dataloaders_by_task[current_task_id]

        print(f"DEBUG: Client {self.id} loading data for task {current_task_id}")
        print(f"DEBUG: Task type is {self.task_type}")
        print(f"DEBUG: Dataset is {self.dataset}")

        try:
            if self.task_type == 'class_incremental':
                # For class-incremental, load data specific to the current task ID
                train_data = read_client_task_data(self.dataset, self.id, current_task_id, is_train=True)
                if train_data is None:
                    print(f"DEBUG: Client {self.id} got None train_data for task {current_task_id}")
                    return None
                elif len(train_data) == 0:
                    print(f"DEBUG: Client {self.id} got empty dataset for task {current_task_id}")
                    return None
                print(f"DEBUG: Client {self.id} successfully loaded {len(train_data)} samples for task {current_task_id}")
            else:
                # For IID/Non-IID tasks (single task scenario), load all client data
                train_data = read_client_data(self.dataset, self.id, is_train=True, few_shot=self.few_shot)
                if train_data is None:
                    print(f"DEBUG: Client {self.id} got None train_data")
                    return None
                elif len(train_data) == 0:
                    print(f"DEBUG: Client {self.id} got empty dataset")
                    return None
                print(f"DEBUG: Client {self.id} successfully loaded {len(train_data)} samples")

            # Create DataLoader only if we have valid data
            dataloader = DataLoader(train_data, batch_size, drop_last=True, shuffle=True)
            self.train_dataloaders_by_task[current_task_id] = dataloader
            print(f"DEBUG: Client {self.id} created DataLoader with {len(dataloader.dataset)} samples for task {current_task_id}")
            return dataloader
            
        except Exception as e:
            print(f"ERROR: Client {self.id} failed to load data for task {current_task_id}")
            print(f"ERROR details: {str(e)}")
            import traceback
            traceback.print_exc()
            return None

    # Modified to load data for a specific task, OR a list of tasks
    def load_test_data(self, batch_size=None, task_ids=None):
        if batch_size is None:
            batch_size = self.batch_size
        
        if task_ids is None: # Default: load all test data for the client (for overall eval)
            test_data = read_client_data(self.dataset, self.id, is_train=False, few_shot=self.few_shot)
            # --- FIX: Check if test_data is empty before creating DataLoader ---
            if not test_data:
                return None
            return DataLoader(test_data, batch_size, drop_last=False, shuffle=False) 
        
        # For evaluation on specific tasks (e.g., for forgetting metrics)
        combined_test_data = []
        task_id_list = [task_ids] if isinstance(task_ids, int) else task_ids 

        for task_id in task_id_list:
            # Check if this specific task's dataloader is already cached as None or a DataLoader
            if task_id in self.test_dataloaders_by_task:
                cached_dataloader = self.test_dataloaders_by_task[task_id]
                if cached_dataloader is not None and cached_dataloader.dataset: # If valid cached data
                    combined_test_data.extend(cached_dataloader.dataset)
                continue # Move to next task_id

            # If not cached, try to read data for this task
            if self.task_type == 'class_incremental':
                test_data_for_task = read_client_task_data(self.dataset, self.id, task_id, is_train=False)
            else:
                # For non-incremental, task_id=0 means all test data. Other task_ids (if used) would be empty.
                test_data_for_task = read_client_data(self.dataset, self.id, is_train=False, few_shot=self.few_shot) if task_id == 0 else []

            # --- FIX: Check if test_data_for_task is empty before creating DataLoader and caching ---
            if not test_data_for_task:
                self.test_dataloaders_by_task[task_id] = None # Cache None if no data for this task
            else:
                dataloader_for_task = DataLoader(test_data_for_task, batch_size, drop_last=False, shuffle=False)
                self.test_dataloaders_by_task[task_id] = dataloader_for_task
                combined_test_data.extend(dataloader_for_task.dataset) # Add to combined list

        # --- FIX: Check if combined_test_data is empty before creating DataLoader ---
        if not combined_test_data:
             return None 
        
        return DataLoader(combined_test_data, batch_size, drop_last=False, shuffle=False)
        

    def set_parameters(self, model):
        for new_param, old_param in zip(model.parameters(), self.model.parameters()):
            old_param.data = new_param.data.clone()

    def clone_model(self, model, target):
        for param, target_param in zip(model.parameters(), target.parameters()):
            target_param.data = param.data.clone()
            # target_param.grad = param.grad.clone()

    def update_parameters(self, model, new_params):
        for param, new_param in zip(model.parameters(), new_params):
            param.data = new_param.data.clone()

    def test_metrics(self, task_ids=None): # Modified to accept task_ids
        """
        Evaluates the client's model on specified task(s) test data.
        If task_ids is None, evaluates on all client's test data.
        If task_ids is an int, evaluates on that specific task.
        If task_ids is a list, evaluates on the combined data of those tasks.
        """
        testloader = self.load_test_data(task_ids=task_ids)
        # --- FIX: Handle if testloader is None (no data for evaluation) ---
        if testloader is None or not testloader.dataset: 
            return 0, 0, 0.0 # acc, num, auc

        self.model.eval()

        test_acc = 0
        test_num = 0
        y_prob = []
        y_true = []
        
        with torch.no_grad():
            for x, y in testloader:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                output = self.model(x)

                test_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
                test_num += y.shape[0]

                y_prob.append(output.detach().cpu().numpy())
                
                # Handling num_classes for AUC calculation when classes might be missing
                # It's safer to use the actual classes present in the current batch
                # or ensure the range covers all possible classes in the dataset.
                # For classification, using all num_classes is usually fine if model outputs for all.
                # The original `nc = self.num_classes; if self.num_classes == 2: nc += 1; ... lb = lb[:, :2]` 
                # logic is a bit odd. `label_binarize` with `classes=np.arange(self.num_classes)`
                # should handle multiclass correctly without the `nc += 1` hack.
                # If only one class is present in y for a batch, AUC can still fail.
                
                lb = label_binarize(y.detach().cpu().numpy(), classes=np.arange(self.num_classes))
                
                y_true.append(lb)

        y_prob = np.concatenate(y_prob, axis=0)
        y_true = np.concatenate(y_true, axis=0)

        # Check for single-class predictions in y_true, which breaks AUC
        # This check is crucial. If y_true.shape[1] is 0 or 1 for multi-class problem, AUC fails.
        # It also fails if all values in y_true for a binary problem are the same.
        if y_true.shape[0] == 0 or y_true.shape[1] == 0: # No samples or no classes
             auc = 0.0
        elif y_true.shape[1] == 1: # Only one unique class in this entire aggregated test set
            auc = 0.0 # AUC is undefined for single class
        elif np.all(y_true == 0) or np.all(y_true == 1): # All labels are the same (e.g. all 0s or all 1s in a binary classification)
            auc = 0.0
        else:
            try:
                # Ensure y_true has at least two classes to compute AUC.
                # If it's a multi-class problem and current_batch_labels only has one class, AUC might throw error.
                # `average='micro'` handles multi-class AUC by flattening.
                auc = metrics.roc_auc_score(y_true, y_prob, average='micro')
            except ValueError:
                # This can happen if y_true contains only one class after binarization,
                # e.g., if a client's test set for a task only has one class, or if it's a binary problem but all samples are of one class.
                auc = 0.0 

        return test_acc, test_num, auc

    def train_metrics(self, current_task_id=0): # Modified to accept task_id
        """
        Evaluates training loss on the current task's training data.
        """
        trainloader = self.load_train_data(current_task_id=current_task_id)
        # --- FIX: Handle if trainloader is None (no data for evaluation) ---
        if trainloader is None or not trainloader.dataset: # Handle empty dataset
            return 0.0, 0 # loss, num

        self.model.eval()

        train_num = 0
        losses = 0
        with torch.no_grad():
            for x, y in trainloader:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                output = self.model(x)
                loss = self.loss(output, y)
                train_num += y.shape[0]
                losses += loss.item() * y.shape[0]

        return losses, train_num
    
    # The following methods are for saving/loading client-specific items.
    # No changes needed here for task management.
    def save_item(self, item, item_name, item_path=None):
        if item_path == None:
            item_path = self.save_folder_name
        if not os.path.exists(item_path):
            os.makedirs(item_path)
        torch.save(item, os.path.join(item_path, "client_" + str(self.id) + "_" + item_name + ".pt"))

    def load_item(self, item_name, item_path=None):
        if item_path == None:
            item_path = self.save_folder_name
        return torch.load(os.path.join(item_path, "client_" + str(self.id) + "_" + item_name + ".pt"))
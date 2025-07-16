import os
import ujson
import numpy as np
import gc
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from PIL import Image

batch_size = 10
train_ratio = 0.75 # merge original training set and test set, then split it manually. 
alpha = 0.1
 # for Dirichlet distribution. 100 for exdir

def check(config_path, train_path, test_path, num_clients, niid=False, 
          balance=True, partition=None, task_type=None, classes_per_task=None): # Added task_type, classes_per_task
    # check existing dataset
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = ujson.load(f)
        if config['num_clients'] == num_clients and \
            config['non_iid'] == niid and \
            config['balance'] == balance and \
            config['partition'] == partition and \
            config['alpha'] == alpha and \
            config['batch_size'] == batch_size and \
            config.get('task_type') == task_type and \
            config.get('classes_per_task') == classes_per_task: # Check task_type and classes_per_task
            print("\nDataset already generated.\n")
            return True

    dir_path = os.path.dirname(train_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    dir_path = os.path.dirname(test_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    return False

def separate_data(data, num_clients, num_classes, niid=False, balance=False, partition=None, 
                  class_per_client=None, task_type=None, classes_per_task=None):
    X = [[] for _ in range(num_clients)]
    y = [[] for _ in range(num_clients)]
    statistic = [[] for _ in range(num_clients)]

    dataset_content, dataset_label = data
    
    least_samples = int(min(batch_size / (1-train_ratio), len(dataset_label) / num_clients / 2))

    dataidx_map = {} #记录每个client分到了哪几种label的数据以及对应的下标
    
    # --- Start of Data Distribution (I.1, I.2, I.3) ---
    # Determine the actual partition strategy based on niid parameter
    partition_to_use = partition
    balance_to_use = balance
    class_per_client_to_use = class_per_client

    if not niid: # I.1 IID
        partition_to_use = 'pat' 
        class_per_client_to_use = num_classes 
        balance_to_use = True 
        print(f"Generating IID data distribution (partition: {partition_to_use}).")
    else: # niid is True, use provided parameters for non-IID
        print(f"Generating Non-IID data distribution (partition: {partition_to_use}).")

    if partition_to_use == 'pat': # I.1: IID, I.2: Non-IID with overlap (if class_per_client < num_classes)
        idxs = np.array(range(len(dataset_label))) 
        idx_for_each_class = []
        for i in range(num_classes):
            idx_for_each_class.append(idxs[dataset_label == i])

        assigned_clients_per_class = [[] for _ in range(num_classes)] 

        # First, assign classes to clients based on `class_per_client_to_use`
        for client_id in range(num_clients):
            if class_per_client_to_use < num_classes: # Non-IID with overlap (I.2)
                selected_classes_for_client = np.random.choice(num_classes, class_per_client_to_use, replace=False)
                for class_id in selected_classes_for_client:
                    assigned_clients_per_class[class_id].append(client_id)
            else: # IID (I.1), all clients get all classes
                for class_id in range(num_classes):
                    assigned_clients_per_class[class_id].append(client_id)

        # Now, distribute samples for each class to the assigned clients
        for class_id in range(num_classes):
            selected_clients_for_this_class = assigned_clients_per_class[class_id]
            
            if not selected_clients_for_this_class:
                continue 

            num_all_samples_in_class = len(idx_for_each_class[class_id])
            
            # If the class has too few samples for `least_samples` per client
            if num_all_samples_in_class < len(selected_clients_for_this_class) * least_samples:
                print(f"Warning: Class {class_id} has too few samples ({num_all_samples_in_class}) for {len(selected_clients_for_this_class)} clients with min {least_samples} samples each. Adjusting distribution.")
                # Distribute as evenly as possible
                samples_per_client_list = np.array_split(idx_for_each_class[class_id], len(selected_clients_for_this_class))
                for client_idx, client_id in enumerate(selected_clients_for_this_class):
                    if client_id not in dataidx_map.keys():
                        dataidx_map[client_id] = samples_per_client_list[client_idx].tolist()
                    else:
                        dataidx_map[client_id].extend(samples_per_client_list[client_idx].tolist())
                continue

            if balance_to_use: # IID (I.1) or balanced Non-IID with overlap (I.2)
                num_per = num_all_samples_in_class // len(selected_clients_for_this_class)
                for i, client_id in enumerate(selected_clients_for_this_class):
                    start_idx = i * num_per
                    end_idx = start_idx + num_per
                    if client_id not in dataidx_map.keys():
                        dataidx_map[client_id] = idx_for_each_class[class_id][start_idx:end_idx].tolist()
                    else:
                        dataidx_map[client_id].extend(idx_for_each_class[class_id][start_idx:end_idx].tolist())
                
                remaining_samples_idx = idx_for_each_class[class_id][len(selected_clients_for_this_class) * num_per:]
                for i, idx in enumerate(remaining_samples_idx):
                    client_id = selected_clients_for_this_class[i % len(selected_clients_for_this_class)]
                    dataidx_map[client_id].append(idx)

            else: # Unbalanced Non-IID with overlap (I.2)
                proportions = np.random.dirichlet(np.repeat(alpha, len(selected_clients_for_this_class)))
                proportions = (np.cumsum(proportions) * num_all_samples_in_class).astype(int)[:-1]
                
                splits = np.split(idx_for_each_class[class_id], proportions)
                for client_idx, client_id in enumerate(selected_clients_for_this_class):
                    if client_id not in dataidx_map.keys():
                        dataidx_map[client_id] = splits[client_idx].tolist()
                    else:
                        dataidx_map[client_id].extend(splits[client_idx].tolist())
    
    elif partition == "no_overlap": # I.3: Non-IID, No Overlap
        if num_clients * class_per_client_to_use > num_classes:
            raise ValueError("Not enough unique classes for 'no_overlap' partition with current settings. Decrease class_per_client or num_clients.")
        
        classes_per_client_list = []
        all_classes = list(range(num_classes))
        np.random.shuffle(all_classes) 

        current_class_idx = 0
        for i in range(num_clients):
            if current_class_idx >= num_classes:
                # If we've exhausted all classes, wrap around
                current_class_idx = 0
            
            # Take the next class_per_client_to_use classes
            end_idx = min(current_class_idx + class_per_client_to_use, num_classes)
            assigned_classes = all_classes[current_class_idx:end_idx]
            
            # If we don't have enough classes left, wrap around
            if len(assigned_classes) < class_per_client_to_use:
                needed = class_per_client_to_use - len(assigned_classes)
                assigned_classes.extend(all_classes[:needed])
            
            classes_per_client_list.append(assigned_classes)
            current_class_idx = end_idx
            
            client_data_indices = []
            for cls in assigned_classes:
                client_data_indices.extend(np.where(dataset_label == cls)[0].tolist())
            dataidx_map[i] = client_data_indices
        
        print(f"Generating Non-IID data distribution with NO OVERLAP (partition: {partition_to_use}).")
    
    elif partition == "dir": # I.2: Non-IID with overlap via Dirichlet distribution
        # https://github.com/IBM/probabilistic-federated-neural-matching/blob/master/experiment.py
        min_size = 0
        K = num_classes
        N = len(dataset_label)

        try_cnt = 1
        while min_size < least_samples:
            if try_cnt > 1:
                print(f'Client data size does not meet the minimum requirement {least_samples}. Try allocating again for the {try_cnt}-th time.')

            idx_batch = [[] for _ in range(num_clients)]
            for k in range(K):
                idx_k = np.where(dataset_label == k)[0]
                np.random.shuffle(idx_k)
                proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
                proportions = np.array([p*(len(idx_j)<N/num_clients) for p,idx_j in zip(proportions,idx_batch)])
                proportions = proportions/proportions.sum()
                proportions = (np.cumsum(proportions)*len(idx_k)).astype(int)[:-1]
                idx_batch = [idx_j + idx.tolist() for idx_j,idx in zip(idx_batch,np.split(idx_k,proportions))]
                min_size = min([len(idx_j) for idx_j in idx_batch])
            try_cnt += 1

        for j in range(num_clients):
            dataidx_map[j] = idx_batch[j]
        print(f"Generating Non-IID data distribution with OVERLAP via Dirichlet (partition: {partition}, alpha: {alpha}).")

    elif partition == 'exdir': # I.2: Non-IID with overlap via ExDir
        # for Non-IID with overlap I.2
        r'''This strategy comes from https://arxiv.org/abs/2311.03154
        See details in https://github.com/TsingZ0/PFLlib/issues/139

        This version in PFLlib is slightly different from the original version 
        Some changes are as follows:
        n_nets -> num_clients, n_class -> num_classes
        '''
        C = class_per_client_to_use
        
        '''The first level: allocate labels to clients
        clientidx_map (dict, {label: clientidx}), e.g., C=2, num_clients=5, num_classes=10
            {0: [0, 1], 1: [1, 2], 2: [2, 3], 3: [3, 4], 4: [4, 5], 5: [5, 6], 6: [6, 7], 7: [7, 8], 8: [8, 9], 9: [9, 0]}
        '''
        min_size_per_label = 0
        min_require_size_per_label = max(C * num_clients // num_classes // 2, 1)
        if min_require_size_per_label < 1:
            raise ValueError
        clientidx_map = {}
        while min_size_per_label < min_require_size_per_label:
            for k in range(num_classes):
                clientidx_map[k] = []
            for i in range(num_clients):
                labelidx = np.random.choice(range(num_classes), C, replace=False)
                for k in labelidx:
                    clientidx_map[k].append(i)
            min_size_per_label = min([len(clientidx_map[k]) for k in range(num_classes)])
        
        '''The second level: allocate data idx'''
        dataidx_map = {}
        y_train = dataset_label
        min_size = 0
        min_require_size = 10
        K = num_classes
        N = len(y_train)
        print("\n*****clientidx_map*****")
        print(clientidx_map)
        print("\n*****Number of clients per label*****")
        print([len(clientidx_map[i]) for i in range(len(clientidx_map))])

        while min_size < min_require_size:
            idx_batch = [[] for _ in range(num_clients)]
            for k in range(K):
                idx_k = np.where(y_train == k)[0]
                np.random.shuffle(idx_k)
                proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
                proportions = np.array([p * (len(idx_j) < N / num_clients and j in clientidx_map[k]) for j, (p, idx_j) in enumerate(zip(proportions, idx_batch))])
                proportions = proportions / proportions.sum()
                proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))] 
                min_size = min([len(idx_j) for idx_j in idx_batch])

        for j in range(num_clients):
            np.random.shuffle(idx_batch[j])
            dataidx_map[j] = idx_batch[j]
        print(f"Generating Non-IID data distribution with OVERLAP via ExDir (partition: {partition}, alpha: {alpha}).")
    
    else:
        raise NotImplementedError(f"Partition type '{partition}' not implemented.")
    
    # Ensure all clients have data
    for client_id in range(num_clients):
        if client_id not in dataidx_map:
            dataidx_map[client_id] = []
    
    # --- End of Data Distribution ---
    print("\n--- DEBUG: Dataidx_map contents after initial partitioning ---")
    for client_id_debug in range(num_clients):
        if client_id_debug in dataidx_map and len(dataidx_map[client_id_debug]) > 0:
            current_client_indices_debug = np.array(dataidx_map[client_id_debug])
            current_client_labels_debug = dataset_label[current_client_indices_debug]
            unique_labels_debug, counts_debug = np.unique(current_client_labels_debug, return_counts=True)
            print(f"Client {client_id_debug} (Initial): Total samples = {len(current_client_indices_debug)}, Unique Labels = {unique_labels_debug}, Counts = {counts_debug}")
        else:
            print(f"Client {client_id_debug} (Initial): No data assigned in dataidx_map.")
    print("--- END DEBUG: Initial Partitioning ---")

    # --- Start of Task Sequence (II.1, II.2, II.3) ---
    client_task_data = [{} for _ in range(num_clients)] 
    
    num_tasks_final = 1 
    task_classes_map_final = {0: list(range(num_classes))} 

    if task_type == 'class_incremental':
        if not classes_per_task or classes_per_task <= 0:
            raise ValueError("For 'class_incremental' task_type, 'classes_per_task' must be specified and positive.")
        
        # Use sequential class assignment instead of random shuffling for better debugging
        all_classes = list(range(num_classes))
        num_tasks_final = num_classes // classes_per_task
        if num_classes % classes_per_task != 0:
            num_tasks_final += 1 

        task_classes_map_final = {} 
        for t_idx in range(num_tasks_final):
            start_class_idx = t_idx * classes_per_task
            end_class_idx = min((t_idx + 1) * classes_per_task, num_classes)
            current_task_classes = all_classes[start_class_idx:end_class_idx]
            task_classes_map_final[t_idx] = current_task_classes
        
        # --- DEBUG: Print Task Class Map ---
        print("\n--- DEBUG: Task Class Map ---")
        print(f"Total tasks: {num_tasks_final}")
        for t_idx, classes in task_classes_map_final.items():
            print(f"Task {t_idx}: Classes {classes}")
        print("--- END DEBUG: Task Class Map ---")

        print(f"Generating Class-Incremental Tasks: {num_tasks_final} tasks, {classes_per_task} classes per task.")
        for client_id in range(num_clients):
            # Ensure client_overall_indices is a numpy array
            client_overall_indices = np.array(dataidx_map.get(client_id, [])) # Use .get with default [] for safety
            
            # If client has no overall data from initial partitioning, skip directly
            if len(client_overall_indices) == 0:
                print(f"DEBUG: Client {client_id} has NO overall data, skipping task assignment.")
                for t_idx in range(num_tasks_final): # Initialize empty tasks for this client
                    client_task_data[client_id][t_idx] = {'x': np.array([]), 'y': np.array([])}
                continue # Move to next client

            client_overall_labels = dataset_label[client_overall_indices] 

            print(f"\nDEBUG: Processing Client {client_id} for Class-Incremental Tasks.")
            print(f"  Client {client_id} overall labels in its data: {np.unique(client_overall_labels)}")

            for t_idx in range(num_tasks_final):
                current_task_classes = task_classes_map_final[t_idx]
                task_specific_indices = []
                
                print(f"  DEBUG:   For Task {t_idx} (Classes: {current_task_classes}):")

                for cls in current_task_classes:
                    # Find indices from client_overall_indices that correspond to current task's classes
                    # This filters the client's overall data based on the current task's classes
                    indices_for_class_in_client = client_overall_indices[client_overall_labels == cls]
                    if len(indices_for_class_in_client) > 0:
                        task_specific_indices.extend(indices_for_class_in_client.tolist())
                        print(f"    DEBUG:     Client {client_id} HAS data for Class {cls} in Task {t_idx} (count: {len(indices_for_class_in_client)})")
                    else:
                        print(f"    DEBUG:     Client {client_id} has NO data for Class {cls} in Task {t_idx}")
                
                if task_specific_indices: 
                    # Convert to numpy array for indexing
                    task_specific_indices = np.array(task_specific_indices)
                    client_task_data[client_id][t_idx] = {
                        'x': dataset_content[task_specific_indices], 
                        'y': dataset_label[task_specific_indices]
                    }
                    print(f"  DEBUG:   Client {client_id} assigned {len(task_specific_indices)} samples for Task {t_idx}.")
                else: 
                    client_task_data[client_id][t_idx] = {'x': np.array([]), 'y': np.array([])}
                    print(f"  DEBUG:   Client {client_id} has NO samples for Task {t_idx} after filtering for classes.")

    elif task_type == 'iid_tasks': # II.1: IID Tasks
        print(f"Generating IID Tasks: Each task contains all classes evenly distributed.")
        # In this mode, the entire client data is considered one "task" (Task 0)
        # So client_task_data for task 0 will contain all their data
        for client_id in range(num_clients):
            client_overall_indices = np.array(dataidx_map.get(client_id, [])) # Ensure it's numpy array
            if len(client_overall_indices) > 0:
                client_task_data[client_id][0] = {
                    'x': dataset_content[client_overall_indices],
                    'y': dataset_label[client_overall_indices]
                }
            else:
                client_task_data[client_id][0] = {'x': np.array([]), 'y': np.array([])}

    elif task_type == 'non_iid_tasks': # II.2: Non-IID Tasks (within tasks, data is non-IID)
        print(f"Generating Non-IID Tasks: Each task contains all classes, but with imbalanced distribution.")
        # Similar to iid_tasks, but the data distribution per client is already non-IID from the partition step.
        for client_id in range(num_clients):
            client_overall_indices = np.array(dataidx_map.get(client_id, [])) # Ensure it's numpy array
            if len(client_overall_indices) > 0:
                client_task_data[client_id][0] = {
                    'x': dataset_content[client_overall_indices],
                    'y': dataset_label[client_overall_indices]
                }
            else:
                client_task_data[client_id][0] = {'x': np.array([]), 'y': np.array([])}

    else:
        raise NotImplementedError(f"Task type '{task_type}' not implemented.")

    # --- End of Task Sequence ---

    final_X = [[] for _ in range(num_clients)]
    final_y = [[] for _ in range(num_clients)]
    final_statistic = [[] for _ in range(num_clients)]

    for client in range(num_clients):
        # Ensure dataidx_map[client] is a numpy array for consistent indexing
        current_client_indices = np.array(dataidx_map.get(client, []))
        if len(current_client_indices) > 0:
            final_X[client] = dataset_content[current_client_indices]
            final_y[client] = dataset_label[current_client_indices]
        else:
            final_X[client] = np.array([])
            final_y[client] = np.array([])

        for i in np.unique(final_y[client]):
            final_statistic[client].append((int(i), int(sum(final_y[client]==i))))
            
    del data
    # gc.collect() # Consider if gc.collect() is truly needed or if it causes issues

    for client in range(num_clients):
        if len(final_X[client]) > 0:
            print(f"Client {client}\t Size of data: {len(final_X[client])}\t Labels: ", np.unique(final_y[client]))
            print(f"\t\t Samples of labels: ", [i for i in final_statistic[client]])
        else:
            print(f"Client {client}\t Size of data: 0\t Labels: []")
        print("-" * 50)
    
    return final_X, final_y, final_statistic, client_task_data, num_tasks_final, task_classes_map_final


def split_data(X, y):
    # Split dataset
    train_data, test_data = [], []
    num_samples = {'train':[], 'test':[]}

    for i in range(len(y)):
        # Handle empty arrays for clients that might have no data for some tasks in sparse partitions
        if len(X[i]) == 0:
            train_data.append({'x': np.array([]), 'y': np.array([])})
            num_samples['train'].append(0)
            test_data.append({'x': np.array([]), 'y': np.array([])})
            num_samples['test'].append(0)
            continue

        # --- Start of fix for ValueError: The least populated class has only 1 member ---
        # Identify classes with only one member
        unique_labels, counts = np.unique(y[i], return_counts=True)
        single_sample_labels = unique_labels[counts == 1]

        if len(single_sample_labels) > 0:
            # If there are classes with only one sample, train_test_split with stratify will fail.
            # Option 1 (Recommended if small): Remove these single-sample classes from the split.
            # This might change the distribution slightly but prevents crash.
            # Filter out samples belonging to single-sample classes
            mask = np.isin(y[i], single_sample_labels, invert=True)
            X_filtered = X[i][mask]
            y_filtered = y[i][mask]

            # Split the filtered data with stratify
            if len(y_filtered) > 0: # Ensure there's still data after filtering
                X_train_filtered, X_test_filtered, y_train_filtered, y_test_filtered = train_test_split(
                    X_filtered, y_filtered, train_size=train_ratio, shuffle=True, stratify=y_filtered)
            else: # No data left after filtering
                X_train_filtered, X_test_filtered, y_train_filtered, y_test_filtered = np.array([]), np.array([]), np.array([]), np.array([])

            X_train_single = X[i][~mask]
            y_train_single = y[i][~mask]

            if len(X_train_filtered) > 0:
                X_train = np.concatenate((X_train_filtered, X_train_single))
                y_train = np.concatenate((y_train_filtered, y_train_single))
            else:
                X_train = X_train_single
                y_train = y_train_single
            
            X_test = X_test_filtered
            y_test = y_test_filtered

            # Shuffle after concatenation to mix.
            if len(y_train) > 0:
                perm = np.random.permutation(len(y_train))
                X_train, y_train = X_train[perm], y_train[perm]
        else:
            # No classes with only one member, proceed with stratify
            X_train, X_test, y_train, y_test = train_test_split(
                X[i], y[i], train_size=train_ratio, shuffle=True, stratify=y[i])

        train_data.append({'x': X_train, 'y': y_train})
        num_samples['train'].append(len(y_train))
        test_data.append({'x': X_test, 'y': y_test})
        num_samples['test'].append(len(y_test))

    print("Total number of samples:", sum(num_samples['train'] + num_samples['test']))
    print("The number of train samples:", num_samples['train'])
    print("The number of test samples:", num_samples['test'])
    print()
    del X, y
    # gc.collect()

    return train_data, test_data

def save_file(config_path, train_path, test_path, train_data, test_data, num_clients, 
              num_classes, statistic, niid=False, balance=True, partition=None,
              task_type=None, classes_per_task=None, num_tasks=None, task_classes_map=None, # Added task related params
              client_task_data=None): # Added client_task_data
    
    config = {
        'num_clients': num_clients, 
        'num_classes': num_classes, 
        'non_iid': niid, 
        'balance': balance, 
        'partition': partition, 
        'Size of samples for labels in clients': statistic, 
        'alpha': alpha, 
        'batch_size': batch_size,
        'task_type': task_type, # New
        'classes_per_task': classes_per_task, # New
        'num_tasks': num_tasks, # New
        'task_classes_map': task_classes_map, # New: A dictionary mapping task_id to list of classes
    }

    # gc.collect()
    print("Saving to disk.\n")

    # Save overall client data (as before)
    for idx, train_dict in enumerate(train_data):
        with open(train_path + str(idx) + '.npz', 'wb') as f:
            np.savez_compressed(f, data=train_dict)
    for idx, test_dict in enumerate(test_data):
        with open(test_path + str(idx) + '.npz', 'wb') as f:
            np.savez_compressed(f, data=test_dict)
    
    # Save task-specific data if applicable
    if client_task_data:
        # Create task_data directories for both train and test
        train_task_data_path = os.path.dirname(train_path) + "/task_data/"
        test_task_data_path = os.path.dirname(test_path) + "/task_data/"
        os.makedirs(train_task_data_path, exist_ok=True)
        os.makedirs(test_task_data_path, exist_ok=True)
        
        # Process each client's data
        for client_id, tasks_dict in enumerate(client_task_data):
            for task_id, data_dict in tasks_dict.items():
                if data_dict['x'].size > 0:  # Only save if client has data for this task
                    # Split data into train and test sets
                    x_data = data_dict['x']
                    y_data = data_dict['y']
                    
                    # Check class distribution
                    unique_labels, counts = np.unique(y_data, return_counts=True)
                    min_samples = np.min(counts)
                    
                    if min_samples < 2:
                        # If any class has less than 2 samples, use simple random split
                        num_samples = len(y_data)
                        num_train = int(0.75 * num_samples)  # Use same ratio as in split_data
                        indices = np.random.permutation(num_samples)
                        train_indices = indices[:num_train]
                        test_indices = indices[num_train:]
                        
                        x_train = x_data[train_indices]
                        x_test = x_data[test_indices]
                        y_train = y_data[train_indices]
                        y_test = y_data[test_indices]
                    else:
                        # Use stratified split if all classes have enough samples
                        from sklearn.model_selection import train_test_split
                        x_train, x_test, y_train, y_test = train_test_split(
                            x_data, y_data, 
                            train_size=0.75,  # Use same ratio as in split_data
                            stratify=y_data,  # Maintain class distribution
                            random_state=42    # For reproducibility
                        )
                    
                    # Save train data
                    with open(os.path.join(train_task_data_path, f'client_{client_id}_task_{task_id}.npz'), 'wb') as f:
                        np.savez_compressed(f, x=x_train, y=y_train)
                    
                    # Save test data
                    with open(os.path.join(test_task_data_path, f'client_{client_id}_task_{task_id}.npz'), 'wb') as f:
                        np.savez_compressed(f, x=x_test, y=y_test)
    
    with open(config_path, 'w') as f:
        ujson.dump(config, f)

    print("Finish generating dataset.\n")


class ImageDataset(Dataset):
    def __init__(self, dataframe, image_folder, transform=None):
        """
        Args:
            dataframe (pd.DataFrame): DataFrame containing file names and labels.
                                      Assumes 'file_name' and 'class' columns.
            image_folder (str): Path to the folder containing the images.
            transform (callable, optional): Optional transform to be applied to the image.
        """
        self.dataframe = dataframe
        self.image_folder = image_folder
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        # Get the file name from the DataFrame
        img_name = self.dataframe.iloc[idx]['file_name']
        img_label = self.dataframe.iloc[idx]['class']
        img_path = os.path.join(self.image_folder, img_name)
        
        # Load the image using PIL
        image = Image.open(img_path).convert('RGB')  # Ensure RGB if not grayscale
        
        if self.transform:
            image = self.transform(image)
        
        return image, img_label
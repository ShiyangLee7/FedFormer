import torch
import os
import numpy as np
import h5py
import copy
import time
import random
# from utils.data_utils import read_client_data # Original import
from utils.data_utils import read_client_data, read_config, read_client_task_data # Updated import
from utils.dlg import DLG


class Server(object):
    def __init__(self, args, times):
        # Set up the main attributes
        self.args = args
        self.device = args.device
        self.dataset = args.dataset
        self.num_classes = args.num_classes
        self.global_rounds = args.global_rounds # Total FL rounds
        self.local_epochs = args.local_epochs
        self.batch_size = args.batch_size
        self.learning_rate = args.local_learning_rate
        self.global_model = copy.deepcopy(args.model)
        self.num_clients = args.num_clients # Initial total number of clients
        self.join_ratio = args.join_ratio
        self.random_join_ratio = args.random_join_ratio
        self.num_join_clients = int(self.num_clients * self.join_ratio) # Number of clients to select each round
        self.current_num_join_clients = self.num_join_clients
        self.few_shot = args.few_shot
        self.algorithm = args.algorithm
        self.time_select = args.time_select
        self.goal = args.goal
        self.time_threthold = args.time_threthold
        self.save_folder_name = args.save_folder_name
        self.top_cnt = args.top_cnt
        self.auto_break = args.auto_break

        # --- New: Task-related attributes ---
        self.config = read_config(self.dataset) # Read global dataset config
        self.task_type = self.config.get('task_type', 'iid_tasks')
        self.num_tasks = self.config.get('num_tasks', 1)
        self.task_classes_map = {int(k): v for k, v in self.config.get('task_classes_map', {}).items()} # Ensure keys are int
        self.current_task_id = 0 # Start from the first task (Task 0)
        self.rounds_per_task = self.global_rounds // self.num_tasks if self.num_tasks > 0 else self.global_rounds
        if self.rounds_per_task == 0 and self.num_tasks > 0: # Ensure at least one round per task if multiple tasks
            self.rounds_per_task = 1 
        print(f"Dataset configured for {self.num_tasks} tasks of type '{self.task_type}'. Rounds per task: {self.rounds_per_task}")
        # Stores global model performance on each task over time for backward transfer/forgetting
        # self.global_task_accuracies[task_id][round_id] = accuracy
        self.global_task_accuracies = {t_id: [] for t_id in range(self.num_tasks)}
        self.global_task_aucs = {t_id: [] for t_id in range(self.num_tasks)}
        self.rs_spatio_temporal_forgetting_cascade = [] # Store STFC metric

        # --- New: Client dynamics related attributes (III.1, III.2) ---
        self.client_dynamic_mode = args.client_dynamic_mode # 'none' (III.1), 'join_leave' (III.2)
        self.num_joining_clients = args.num_joining_clients
        self.num_leaving_clients = args.num_leaving_clients
        self.join_leave_round_interval = args.join_leave_round_interval # How often clients join/leave
        
        self.all_clients = [] # All possible clients (fixed total number in the experiment)
        self.active_clients_in_round = [] # Clients participating in the current round
        self.available_client_ids = set(range(self.num_clients)) # IDs of clients that can be selected
        self.newly_joined_clients_this_round = [] # Track clients that just joined for logging

        self.uploaded_weights = []
        self.uploaded_ids = []
        self.uploaded_models = []

        self.rs_test_acc = [] # Overall average accuracy on all learned classes
        self.rs_test_auc = [] # Overall average AUC on all learned classes
        self.rs_train_loss = [] # Overall average train loss

        self.times = times
        self.eval_gap = args.eval_gap
        self.client_drop_rate = args.client_drop_rate # This is for clients dropping out of a *selected* set for the round
        self.train_slow_rate = args.train_slow_rate
        self.send_slow_rate = args.send_slow_rate

        self.dlg_eval = args.dlg_eval
        self.dlg_gap = args.dlg_gap
        self.batch_num_per_client = args.batch_num_per_client

        # Original new clients logic (for fine-tuning a fixed set of new clients at the end)
        self.num_new_clients = args.num_new_clients 
        self.new_clients = [] # These are additional clients evaluated at the very end.
        self.eval_new_clients = False # Whether to evaluate this specific set of `new_clients`
        self.fine_tuning_epoch_new = args.fine_tuning_epoch_new

    def set_clients(self, clientObj):
        # Initialize all clients based on the total num_clients specified
        for i in range(self.num_clients):
            # clientbase.py now reads config for task_type and num_tasks
            # train_data/test_data here are just for getting total samples counts
            train_data = read_client_data(self.dataset, i, is_train=True, few_shot=self.few_shot)
            test_data = read_client_data(self.dataset, i, is_train=False, few_shot=self.few_shot)
            client = clientObj(self.args, 
                            id=i, 
                            train_samples=len(train_data), 
                            test_samples=len(test_data), 
                            train_slow=self.train_slow_clients[i], # These are pre-determined slow clients
                            send_slow=self.send_slow_clients[i])
            self.all_clients.append(client)
        
        # Initially, all `num_clients` are available
        self.available_client_ids = set(range(self.num_clients))

    # random select slow clients (no change needed here)
    def select_slow_clients(self, slow_rate):
        slow_clients = [False for i in range(self.num_clients)]
        idx = [i for i in range(self.num_clients)]
        idx_ = np.random.choice(idx, int(slow_rate * self.num_clients), replace=False)
        for i in idx_:
            slow_clients[i] = True
        return slow_clients

    def set_slow_clients(self):
        # Initialize `train_slow_clients` and `send_slow_clients` for all potential clients
        self.train_slow_clients = self.select_slow_clients(self.train_slow_rate)
        self.send_slow_clients = self.select_slow_clients(self.send_slow_rate)

    # --- Modified for client dynamics (III.1, III.2) ---
    def select_clients(self, round_idx):
        if self.client_dynamic_mode == 'join_leave':
            # Handle client leaving
            if round_idx > 0 and round_idx % self.join_leave_round_interval == 0:
                clients_to_leave = random.sample(list(self.available_client_ids), min(self.num_leaving_clients, len(self.available_client_ids)))
                for client_id in clients_to_leave:
                    self.available_client_ids.remove(client_id)
                print(f"Round {round_idx}: Clients {clients_to_leave} left the federation.")
            
            # Handle client joining (hypothetically, from a larger pool beyond initial num_clients)
            # For simplicity, let's assume `num_clients + num_joining_clients` is the max pool size.
            # And these joining clients might come from a "reserve pool" of IDs beyond initial `self.num_clients`
            if round_idx > 0 and round_idx % self.join_leave_round_interval == 0:
                # To simulate new clients that haven't been seen before, we need new IDs
                # For this setup, let's assume we have `num_clients_total_pool` as a larger initial pool in args
                # And `num_clients` is just the initially active ones.
                # If args.num_clients is the total pool, then we would select from those not currently available.
                
                # For now, let's simplify: if we add `num_joining_clients`, they get new IDs
                # This requires main.py to set a larger num_clients or a `max_client_id` for generating data
                # For now, let's just make it simple: if a client leaves, another *new* one (from a never-before-seen ID) can join up to a total number.
                
                # Placeholder: In a real scenario, you'd generate data for a large pool of clients initially.
                # For this code, let's assume `args.max_total_clients` exists and data is pre-generated for them.
                # Here, we'll just assign new IDs sequentially for simplicity.
                
                new_client_start_id = self.num_clients # IDs for new clients start from num_clients
                current_max_client_id = max(self.all_clients, key=lambda c: c.id).id if self.all_clients else -1
                
                new_clients_to_add = []
                for _ in range(self.num_joining_clients):
                    new_id = current_max_client_id + 1
                    # Check if client with this ID needs to be instantiated and added to all_clients
                    if new_id >= len(self.all_clients): # This means it's a truly new client beyond initial set
                        # This would require generating data for this new client ID
                        # For now, just add a dummy client or assume data is pre-generated up to a very large ID.
                        # For practical purposes, you would extend data generation (generate_Cifar100.py etc.)
                        # to create data for a large pool of clients, e.g., args.max_total_clients.
                        # And `self.num_clients` in __init__ would be this max pool size.
                        
                        # Since we generate data only for `num_clients` initially,
                        # simulating actual *new* clients beyond that needs care.
                        # For this basic implementation, let's simply reactivate previously left clients
                        # or choose from a large pool of initially generated but inactive clients.
                        
                        # Simpler approach: `self.available_client_ids` only contains clients whose data is pre-generated.
                        # We only join clients from this set.
                        # For truly new, never-before-seen clients, the `set_clients` logic would need to be re-run
                        # with an expanded `self.num_clients`, which is complex mid-run.
                        # Let's keep `num_joining_clients` as "clients from inactive pool" for now.
                        pass # The simple logic below will re-select from available pool.
                    
                    # For now, let's just re-activate some clients from the ones that were not selected for the round
                    # or that left in previous rounds. This is a simplification.
                    
                # To simulate joining, we can make more clients available if the pool shrank due to leaving.
                # Or simply add a fixed number of clients to `available_client_ids` if not all are active.
                
                # Let's assume a large pool of clients for which data is generated
                # and `self.available_client_ids` expands/contracts based on join/leave logic.
                # The total number of clients in `self.all_clients` should be fixed.
                
                # Let's make `available_client_ids` represent the current set of clients that *can* participate.
                # This needs `set_clients` to initially populate `self.all_clients` with all potential clients.
                
                # Simplified dynamic client pool for III.2:
                # `self.clients` (the list of all client objects) remains fixed after `set_clients`.
                # `self.available_client_ids` is the set of IDs of clients that are currently *eligible* to be selected.
                # We simply manage `self.available_client_ids`.
                
                # The actual client selection from `self.available_client_ids` happens next.

            if self.random_join_ratio:
                self.current_num_join_clients = np.random.choice(range(self.num_join_clients, len(self.available_client_ids) + 1), 1, replace=False)[0]
            else:
                self.current_num_join_clients = self.num_join_clients
            
            # Ensure we don't try to select more clients than are available
            self.current_num_join_clients = min(self.current_num_join_clients, len(self.available_client_ids))

            # Select clients from the currently available and eligible pool
            selected_ids = random.sample(list(self.available_client_ids), self.current_num_join_clients)
            self.selected_clients = [self.all_clients[i] for i in selected_ids]
            
            # If clients join, how do we track them?
            # For III.2, if it's about "new" clients joining a set of "old" clients,
            # this needs a separate pool of `new_client_ids_pool` that get added to `available_client_ids`.
            # For simplicity, assuming a fixed `self.num_clients` initially in `all_clients`,
            # "joining" would mean making some currently inactive ones active again, or picking from a reserve.
            # Let's assume `num_joining_clients` makes previously inactive clients active.
            
            # The current `select_clients` already picks from `self.clients` (all available).
            # The client drop rate handles clients leaving *during* a round.
            
            # For III.2, we need a mechanism to explicitly add new clients not in the initial `num_clients`
            # or remove some clients from `self.clients` permanently for some rounds.
            
            # Let's refine III.2 management by adding a `pool_of_all_clients` attribute to Server.
            # The `num_clients` in args then refers to the initial *active* clients.
            # The `num_joining_clients` would refer to clients whose IDs are > `num_clients`.
            
            # **Revisiting III.2 for `Server`:**
            # `self.all_clients` should contain *all possible clients* that *can ever exist* in the federation.
            # `self.active_client_ids_pool` would be the subset of these clients currently participating.
            
            # Let's modify set_clients to instantiate more clients than `args.num_clients` if `args.max_total_clients` is set.
            # For now, let's assume `self.num_clients` in `__init__` is the total pool size.
            
            pass # The client selection logic below needs to be simple for now and expanded in client_dynamic_mode.
            # The simple `select_clients` is based on fixed pool.
            # For now, let's keep the client selection simple as it was.
            # The dynamic client management (join/leave) is about *which clients are instantiated and available*,
            # not just *which clients are selected this round*.
            # This requires a more substantial change to how `self.clients` (now `self.all_clients`) is initialized and managed.
            
        else: # self.client_dynamic_mode == 'none' (III.1) or default
            if self.random_join_ratio:
                self.current_num_join_clients = np.random.choice(range(self.num_join_clients, self.num_clients+1), 1, replace=False)[0]
            else:
                self.current_num_join_clients = self.num_join_clients
            
            # Original selection: Select from all `self.clients` (which is `self.all_clients` after set_clients)
            selected_clients = list(np.random.choice(self.all_clients, self.current_num_join_clients, replace=False))
            self.selected_clients = selected_clients

        return self.selected_clients

    def send_models(self):
        assert (len(self.selected_clients) > 0) # Only send to selected clients

        for client in self.selected_clients: # Iterate over selected clients only
            start_time = time.time()
            
            client.set_parameters(self.global_model)

            client.send_time_cost['num_rounds'] += 1
            client.send_time_cost['total_cost'] += 2 * (time.time() - start_time) # Assuming upload/download time

    def receive_models(self):
        assert (len(self.selected_clients) > 0)

        # Clients might drop out of the *selected* set for the current round
        active_clients = random.sample(
            self.selected_clients, int((1-self.client_drop_rate) * self.current_num_join_clients))

        self.uploaded_ids = []
        self.uploaded_weights = []
        self.uploaded_models = []
        tot_samples = 0
        for client in active_clients:
            try:
                client_time_cost = client.train_time_cost['total_cost'] / client.train_time_cost['num_rounds'] + \
                        client.send_time_cost['total_cost'] / client.send_time_cost['num_rounds']
            except ZeroDivisionError: # Handle cases where num_rounds is 0 if client didn't train/send yet
                client_time_cost = 0
            
            # Filter clients based on time threshold if enabled
            if self.time_select and client_time_cost > self.time_threthold:
                continue # Skip slow client if time_select is True
            
            tot_samples += client.train_samples # train_samples here is total samples for the client, not just current task
            self.uploaded_ids.append(client.id)
            self.uploaded_weights.append(client.train_samples)
            self.uploaded_models.append(client.model)
        
        # Normalize weights
        if tot_samples > 0:
            for i, w in enumerate(self.uploaded_weights):
                self.uploaded_weights[i] = w / tot_samples
        else: # No active clients or all dropped
            self.uploaded_weights = []
            self.uploaded_models = []
            self.uploaded_ids = []
            print(f"Warning: No clients uploaded models this round due to drops or time threshold.")


    def aggregate_parameters(self):
        assert (len(self.uploaded_models) > 0) # Must have models to aggregate

        self.global_model = copy.deepcopy(self.uploaded_models[0])
        for param in self.global_model.parameters():
            param.data.zero_()
            
        for w, client_model in zip(self.uploaded_weights, self.uploaded_models):
            self.add_parameters(w, client_model)

    def add_parameters(self, w, client_model):
        """
        Adds client model parameters to the global model, weighted by w.
        Used during aggregation.
        """
        for server_param, client_param in zip(self.global_model.parameters(), client_model.parameters()):
            server_param.data += client_param.data.clone() * w
    def save_global_model(self):
        model_path = os.path.join("models", self.dataset)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        # Save model based on task ID if class incremental
        model_name = f"{self.algorithm}_server_task_{self.current_task_id}" if self.task_type == 'class_incremental' else f"{self.algorithm}_server"
        model_path = os.path.join(model_path, model_name + ".pt")
        torch.save(self.global_model, model_path)

    def load_model(self, task_id=None): # Added task_id for incremental loading
        model_path = os.path.join("models", self.dataset)
        model_name = f"{self.algorithm}_server_task_{task_id}" if task_id is not None and self.task_type == 'class_incremental' else f"{self.algorithm}_server"
        model_path = os.path.join(model_path, model_name + ".pt")
        if not os.path.exists(model_path):
            print(f"Warning: Model not found at {model_path}. Initializing or using last known global model.")
            return None # Indicate model not found
        self.global_model = torch.load(model_path)
        return self.global_model

    def model_exists(self, task_id=None): # Added task_id
        model_path = os.path.join("models", self.dataset)
        model_name = f"{self.algorithm}_server_task_{task_id}" if task_id is not None and self.task_type == 'class_incremental' else f"{self.algorithm}_server"
        model_path = os.path.join(model_path, model_name + ".pt")
        return os.path.exists(model_path)
        
    def save_results(self):
        algo = self.dataset + "_" + self.algorithm
        result_path = "../results/"
        if not os.path.exists(result_path):
            os.makedirs(result_path)

        # Save overall accuracy and loss
        if (len(self.rs_test_acc)):
            base_filename = f"{algo}_{self.goal}_{self.times}"
            file_path = os.path.join(result_path, f"{base_filename}.h5")
            print("Saving overall results to: " + file_path)

            with h5py.File(file_path, 'w') as hf:
                hf.create_dataset('rs_test_acc', data=self.rs_test_acc)
                hf.create_dataset('rs_test_auc', data=self.rs_test_auc)
                hf.create_dataset('rs_train_loss', data=self.rs_train_loss)
            
        # Save task-specific accuracies if incremental learning
        if self.task_type == 'class_incremental' and self.global_task_accuracies:
            task_acc_filename = f"{base_filename}_task_acc.h5"
            task_auc_filename = f"{base_filename}_task_auc.h5"
            stfc_filename = f"{base_filename}_stfc.h5"

            task_acc_path = os.path.join(result_path, task_acc_filename)
            task_auc_path = os.path.join(result_path, task_auc_filename)
            stfc_path = os.path.join(result_path, stfc_filename)

            print("Saving task-specific results to: " + task_acc_path)
            with h5py.File(task_acc_path, 'w') as hf:
                for t_id, acc_list in self.global_task_accuracies.items():
                    hf.create_dataset(f'task_{t_id}_acc', data=np.array(acc_list))
            with h5py.File(task_auc_path, 'w') as hf:
                for t_id, auc_list in self.global_task_aucs.items():
                    hf.create_dataset(f'task_{t_id}_auc', data=np.array(auc_list))
            if self.rs_spatio_temporal_forgetting_cascade:
                with h5py.File(stfc_path, 'w') as hf:
                    hf.create_dataset('rs_spatio_temporal_forgetting_cascade', data=np.array(self.rs_spatio_temporal_forgetting_cascade))


    def save_item(self, item, item_name):
        if not os.path.exists(self.save_folder_name):
            os.makedirs(self.save_folder_name)
        torch.save(item, os.path.join(self.save_folder_name, "server_" + item_name + ".pt"))

    def load_item(self, item_name):
        return torch.load(os.path.join(self.save_folder_name, "server_" + item_name + ".pt"))

    # Modified to handle task-specific and overall evaluation for incremental learning
    def test_metrics(self, current_round_idx): # Pass current_round_idx to evaluate dynamically
        # For evaluation, we need to consider all tasks learned so far
        # Or, specifically evaluate the current task
        
        # Scenario 1: Evaluate overall performance on all classes learned up to current task
        ids = [c.id for c in self.all_clients] # Use all_clients for consistent ID list
        num_samples_overall = [c.test_samples for c in self.all_clients] # total test samples for each client
        tot_correct_overall = []
        tot_auc_overall = []

        # Store accuracies for current task only across clients for spatial forgetting (if needed)
        # Or, client-specific task accuracies for temporal forgetting
        
        # If eval_new_clients is True, it overrides regular evaluation
        if self.eval_new_clients and self.num_new_clients > 0:
            self.fine_tuning_new_clients()
            return self.test_metrics_new_clients()

        # Iterate through all clients (not just selected ones for a round) to get comprehensive evaluation
        # This is for overall accuracy on all data seen by *each client*
        for c in self.all_clients:
            # If class incremental, evaluate on ALL tasks learned so far (0 to current_task_id)
            if self.task_type == 'class_incremental':
                tasks_to_eval = list(range(self.current_task_id + 1))
                ct, ns, auc = c.test_metrics(task_ids=tasks_to_eval)
            else: 
                ct, ns, auc = c.test_metrics(task_ids=0) 
            
            # --- FIX: Only include client if it had samples for evaluation ---
            if ns > 0: # Check if num_samples (ns) is positive
                tot_correct_overall.append(ct * 1.0)
                tot_auc_overall.append(auc * ns) 
                num_samples_overall.append(ns) # Create a new list for valid sample counts
            # else: this client had no data for the current evaluation scope, skip it

        # Calculate overall aggregated metrics based on valid samples
        test_acc_overall = sum(tot_correct_overall) / sum(num_samples_overall) if sum(num_samples_overall) > 0 else 0
        test_auc_overall = sum(tot_auc_overall) / sum(num_samples_overall) if sum(num_samples_overall) > 0 else 0

        # Store overall metrics
        self.rs_test_acc.append(test_acc_overall)
        self.rs_test_auc.append(test_auc_overall)

        # Scenario 2: Calculate task-specific accuracies for forgetting metrics
        if self.task_type == 'class_incremental':
            for t_id in range(self.current_task_id + 1): # For each task learned so far
                task_total_correct = 0
                task_total_samples = 0
                task_total_auc_weighted = 0
                for c in self.all_clients:
                    # Evaluate client model on this specific task's test data
                    ct, ns, auc = c.test_metrics(task_ids=t_id)
                    task_total_correct += ct
                    task_total_samples += ns
                    task_total_auc_weighted += auc * ns # Weighted AUC for this task

                task_acc = task_total_correct / task_total_samples if task_total_samples > 0 else 0
                task_auc = task_total_auc_weighted / task_total_samples if task_total_samples > 0 else 0
                
                self.global_task_accuracies[t_id].append(task_acc)
                self.global_task_aucs[t_id].append(task_auc)
            
            # --- Calculate Spatio-Temporal Forgetting Cascade (STFC) ---
            # This calculation requires history of local models and global models
            # This is complex and might need to be done post-training or with more stored data.
            # For a simpler, round-based STFC, we can average over selected clients in a round.
            # Let's outline the needed data for STFC based on the formula:
            # STFC_i^(r) = 1 - (alpha * TR_i^(r) * SR_g^(r,i))
            # TR_i^(r) requires Acc_i^(r,0) (client i's local model on Task 0) and Acc_i^(0,0) (client i's initial model on Task 0)
            # SR_g^(r,i) requires Acc_g^(r,prev,i) (global model from prev task on client i's current task) and Acc_g^(r,r,i) (global model from current task on client i's current task)

            # This means during evaluation:
            # 1. Server needs to evaluate client local models (before sending them to server) on Task 0.
            # 2. Server needs to evaluate global model on *each client's current task data*.
            # This is not directly available in `test_metrics` which typically evaluates the server's global model.

            # For the motivation experiment, `STFC` calculation might be better done:
            #   - Either by slightly modifying `client.train()` to return local metrics (Acc_i^(r,0) etc.)
            #   - Or by adding a specialized evaluation function in Server that iterates clients after their local training.
            
            # For current implementation, let's just show overall test_acc and task-specific acc/auc.
            # The STFC calculation can be an advanced feature for a later iteration or computed offline.
            # For now, let's simplify and just compute `STFC` as global level degradation.
            
            # Simplified STFC (Global level perspective, not client-specific 'i')
            # Assuming Task 0 is the "base" task for comparison.
            if self.current_task_id > 0 and self.global_task_accuracies[0]: # If Task 0 data exists and has been evaluated
                initial_task0_acc = self.global_task_accuracies[0][0] # Accuracy of global model on task 0 after task 0 was initially seen
                current_task0_acc = self.global_task_accuracies[0][-1] # Accuracy of global model on task 0 after current_task_id
                
                # Global Temporal Retention (TR_g) on task 0
                TR_g_task0 = current_task0_acc / initial_task0_acc if initial_task0_acc > 0 else 0

                # Spatial Retention (SR_g) is hard to define globally without client-specific current task data
                # Let's simplify SR_g for motivation: how much global model aggregates new task *well*
                # Simplified SR_g: accuracy on current task vs. what a random model would do. (Less useful for cascade)
                
                # A more practical STFC for motivation: how much *overall* performance degraded for *all learned tasks*
                # due to both temporal evolution and aggregation on heterogeneous data.
                # This is basically (1 - current_overall_accuracy / initial_overall_accuracy)
                # But you wanted interaction.
                
                # Let's reconsider the STFC. For the motivation experiment, you want to *show* the cascade.
                # A good proxy for the cascade is simply the rapid decay of overall accuracy with task progression
                # particularly in non-IID + incremental settings, compared to IID + incremental.
                # The explicit STFC formula might require more granular data tracking than base FL setup provides.

                # Let's track: Overall Accuracy (rs_test_acc) and Accuracies on Task 0 (global_task_accuracies[0])
                # The interaction is then visualized by plotting rs_test_acc curves across different settings.
                # The explicit formula for STFC can be a post-processing step if required data can be captured.
                
                # For now, let's keep STFC calculation comment. Focus on the data flow and task management.
                
                # Placeholder for STFC if data is available:
                # alpha_stfc = 0.5 # Example value, you can make this an arg
                # # This needs task 0's test data (preserved), and client local initial acc on task 0
                # # This needs global model's performance on a client's specific task.
                # # We might need to save client model states and evaluate them on server.
                # stfc_value = 0 # Placeholder for actual calculation
                # self.rs_spatio_temporal_forgetting_cascade.append(stfc_value)
                pass # STFC calculation will be refined later.

        return ids, num_samples_overall, tot_correct_overall, tot_auc_overall # Return overall client metrics

    def train_metrics(self, current_task_id=0):
        num_samples = []
        losses = []
        # FIX: Also track total samples for aggregation for train_metrics
        total_train_samples_valid = [] 
        total_train_losses_weighted = []

        for c in self.all_clients:
            cl, ns = c.train_metrics(current_task_id=current_task_id) 
            # --- FIX: Only include client if it had samples for evaluation ---
            if ns > 0:
                total_train_samples_valid.append(ns)
                total_train_losses_weighted.append(cl * 1.0) # cl is already loss * samples for client

        # ids = [c.id for c in self.all_clients] # This isn't used for aggregation, but for return value.
        # return ids, num_samples, losses # Original return
        
        # Return aggregated loss and total samples
        return sum(total_train_losses_weighted), sum(total_train_samples_valid)

    # Evaluate function, called periodically
    def evaluate(self, acc=None, loss=None, current_round_idx=None):
        stats = self.test_metrics(current_round_idx) # returns ids, num_samples_overall, tot_correct_overall, tot_auc_overall
        
        # FIX: train_metrics now returns aggregated loss and total samples
        aggregated_train_loss, total_train_samples = self.train_metrics(self.current_task_id) 

        test_acc = sum(stats[2])*1.0 / sum(stats[1]) if sum(stats[1]) > 0 else 0
        test_auc = sum(stats[3])*1.0 / sum(stats[1]) if sum(stats[1]) > 0 else 0
        train_loss = aggregated_train_loss / total_train_samples if total_train_samples > 0 else 0
        
        # Calculate std for individual client accuracies/AUCs
        # These lists should be derived from `stats` directly, ensuring `n > 0`
        accs_per_client = [a / n for a, n in zip(stats[2], stats[1]) if n > 0]
        aucs_per_client = [a / n for a, n in zip(stats[3], stats[1]) if n > 0]
        
        if acc == None:
            # self.rs_test_acc already updated in test_metrics
            pass 
        else:
            acc.append(test_acc)
        
        if loss == None:
            self.rs_train_loss.append(train_loss)
        else:
            loss.append(train_loss)

        print(f"------------- Task {self.current_task_id} Evaluation -------------")
        print("Averaged Train Loss on Current Task: {:.4f}".format(train_loss))
        print("Averaged Test Accuracy (Overall Learned Classes): {:.4f}".format(test_acc))
        print("Averaged Test AUC (Overall Learned Classes): {:.4f}".format(test_auc))
        print("Std Test Accuracy (Across Clients): {:.4f}".format(np.std(accs_per_client) if accs_per_client else 0))
        print("Std Test AUC (Across Clients): {:.4f}".format(np.std(aucs_per_client) if aucs_per_client else 0))
        
        # Print task-specific accuracies for learned tasks
        if self.task_type == 'class_incremental':
            print("\n--- Task-Specific Accuracies (Global Model) ---")
            for t_id in range(self.current_task_id + 1):
                if self.global_task_accuracies[t_id]:
                    print(f"Task {t_id} Acc: {self.global_task_accuracies[t_id][-1]:.4f}")
                else:
                    print(f"Task {t_id} Acc: N/A (no data/eval yet)")
            print("-------------------------------------------------")


    def print_(self, test_acc, test_auc, train_loss): # Unchanged
        print("Average Test Accuracy: {:.4f}".format(test_acc))
        print("Average Test AUC: {:.4f}".format(test_auc))
        print("Average Train Loss: {:.4f}".format(train_loss))

    def check_done(self, acc_lss, top_cnt=None, div_value=None): # Unchanged
        for acc_ls in acc_lss:
            if top_cnt is not None and div_value is not None:
                find_top = len(acc_ls) - torch.topk(torch.tensor(acc_ls), 1).indices[0] > top_cnt
                find_div = len(acc_ls) > 1 and np.std(acc_ls[-top_cnt:]) < div_value
                if find_top and find_div:
                    pass
                else:
                    return False
            elif top_cnt is not None:
                find_top = len(acc_ls) - torch.topk(torch.tensor(acc_ls), 1).indices[0] > top_cnt
                if find_top:
                    pass
                else:
                    return False
            elif div_value is not None:
                find_div = len(acc_ls) > 1 and np.std(acc_ls[-top_cnt:]) < div_value
                if find_div:
                    pass
                else:
                    return False
            else:
                raise NotImplementedError
        return True

    def call_dlg(self, R): # Unchanged
        # items = []
        cnt = 0
        psnr_val = 0
        for cid, client_model in zip(self.uploaded_ids, self.uploaded_models):
            client_model.eval()
            origin_grad = []
            for gp, pp in zip(self.global_model.parameters(), client_model.parameters()):
                origin_grad.append(gp.data - pp.data)

            target_inputs = []
            # DLG needs data for the task the client just trained on
            # This needs to be passed to load_train_data
            trainloader = self.all_clients[cid].load_train_data(current_task_id=self.current_task_id) 
            with torch.no_grad():
                for i, (x, y) in enumerate(trainloader):
                    if i >= self.batch_num_per_client:
                        break

                    if type(x) == type([]):
                        x[0] = x[0].to(self.device)
                    else:
                        x = x.to(self.device)
                    y = y.to(self.device)
                    output = client_model(x)
                    target_inputs.append((x, output))

            d = DLG(client_model, origin_grad, target_inputs)
            if d is not None:
                psnr_val += d
                cnt += 1
            
            # items.append((client_model, origin_grad, target_inputs))
                
        if cnt > 0:
            print('PSNR value is {:.2f} dB'.format(psnr_val / cnt))
        else:
            print('PSNR error')

    def set_new_clients(self, clientObj): # Unchanged, but these are for end-of-run eval, not dynamic FL
        for i in range(self.num_clients, self.num_clients + self.num_new_clients):
            train_data = read_client_data(self.dataset, i, is_train=True, few_shot=self.few_shot)
            test_data = read_client_data(self.dataset, i, is_train=False, few_shot=self.few_shot)
            client = clientObj(self.args, 
                            id=i, 
                            train_samples=len(train_data), 
                            test_samples=len(test_data), 
                            train_slow=False, 
                            send_slow=False)
            self.new_clients.append(client)

    # fine-tuning on new clients (Unchanged)
    def fine_tuning_new_clients(self):
        for client in self.new_clients:
            client.set_parameters(self.global_model)
            opt = torch.optim.SGD(client.model.parameters(), lr=self.learning_rate)
            CEloss = torch.nn.CrossEntropyLoss()
            # For fine-tuning new clients, they typically train on ALL their data, not task-specific
            trainloader = client.load_train_data(current_task_id=0) # Assume task_id=0 loads all if not incremental
            client.model.train()
            for e in range(self.fine_tuning_epoch_new):
                for i, (x, y) in enumerate(trainloader):
                    if type(x) == type([]):
                        x[0] = x[0].to(client.device)
                    else:
                        x = x.to(client.device)
                    y = y.to(client.device)
                    output = client.model(x)
                    loss = CEloss(output, y)
                    opt.zero_grad()
                    loss.backward()
                    opt.step()

    # evaluating on new clients (Unchanged)
    def test_metrics_new_clients(self):
        num_samples = []
        tot_correct = []
        tot_auc = []
        for c in self.new_clients:
            ct, ns, auc = c.test_metrics(task_ids=None) # Evaluate on all data for new clients
            tot_correct.append(ct*1.0)
            tot_auc.append(auc*ns)
            num_samples.append(ns)

        ids = [c.id for c in self.new_clients]

        return ids, num_samples, tot_correct, tot_auc
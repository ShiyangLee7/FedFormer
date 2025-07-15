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
        self.num_clients = args.num_clients  # Initial number of active clients
        self.reserve_clients = args.reserve_clients  # Number of additional clients that can join
        self.total_clients = self.num_clients + self.reserve_clients  # Total number of clients
        self.join_ratio = args.join_ratio
        self.random_join_ratio = args.random_join_ratio
        self.num_join_clients = int(self.num_clients * self.join_ratio)
        self.current_num_join_clients = self.num_join_clients
        self.few_shot = args.few_shot
        self.algorithm = args.algorithm
        self.time_select = args.time_select
        self.goal = args.goal
        self.time_threthold = args.time_threthold
        self.save_folder_name = args.save_folder_name
        self.top_cnt = args.top_cnt
        self.auto_break = args.auto_break

        # --- Task-related attributes ---
        self.config = read_config(self.dataset)
        self.task_type = self.config.get('task_type', 'iid_tasks')
        self.num_tasks = self.config.get('num_tasks', 1)
        self.task_classes_map = {int(k): v for k, v in self.config.get('task_classes_map', {}).items()}
        self.current_task_id = 0

        # Initialize task accuracy tracking
        # Only initialize tasks that exist in task_classes_map
        self.global_task_accuracies = {t_id: [] for t_id in self.task_classes_map.keys()}
        self.global_task_aucs = {t_id: [] for t_id in self.task_classes_map.keys()}
        self.rs_spatio_temporal_forgetting_cascade = []

        print("\nTask Configuration:")
        print(f"Task Type: {self.task_type}")
        print(f"Number of Tasks: {self.num_tasks}")
        print(f"Task Classes Map: {self.task_classes_map}")

        # --- New: Client dynamics related attributes (III.1, III.2) ---
        self.client_dynamic_mode = args.client_dynamic_mode # 'none' (III.1), 'join_leave' (III.2)
        self.num_joining_clients = args.num_joining_clients
        self.num_leaving_clients = args.num_leaving_clients
        self.join_leave_round_interval = args.join_leave_round_interval # How often clients join/leave
        
        # Track active and reserve client pools
        self.active_client_ids = set(range(self.num_clients))  # Initially active clients
        self.reserve_client_ids = set(range(self.num_clients, self.total_clients))  # Reserve pool
        self.all_clients = []  # Will be populated in set_clients
        self.selected_clients = []  # Clients selected for current round
        
        # Track client dynamics history
        self.client_join_history = []  # [(round_idx, client_id, 'join'), ...]
        self.client_leave_history = []  # [(round_idx, client_id, 'leave'), ...]

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
        """Initialize all clients including reserve pool"""
        for i in range(self.total_clients):  # Initialize both active and reserve clients
            train_data = read_client_data(self.dataset, i, is_train=True, few_shot=self.few_shot)
            test_data = read_client_data(self.dataset, i, is_train=False, few_shot=self.few_shot)
            client = clientObj(self.args, 
                            id=i, 
                            train_samples=len(train_data), 
                            test_samples=len(test_data), 
                            train_slow=self.train_slow_clients[i], 
                            send_slow=self.send_slow_clients[i])
            self.all_clients.append(client)
        
        print(f"\nInitialized {self.total_clients} clients:")
        print(f"  Active pool: {len(self.active_client_ids)} clients (IDs: {self.active_client_ids})")
        print(f"  Reserve pool: {len(self.reserve_client_ids)} clients (IDs: {self.reserve_client_ids})")

    # random select slow clients (no change needed here)
    def select_slow_clients(self, slow_rate):
        """Modified to handle total_clients instead of num_clients"""
        slow_clients = [False for _ in range(self.total_clients)]
        idx = [i for i in range(self.total_clients)]
        idx_ = np.random.choice(idx, int(slow_rate * self.total_clients), replace=False)
        for i in idx_:
            slow_clients[i] = True
        return slow_clients

    def set_slow_clients(self):
        """Initialize slow client flags for all clients including reserve pool"""
        self.train_slow_clients = self.select_slow_clients(self.train_slow_rate)
        self.send_slow_clients = self.select_slow_clients(self.send_slow_rate)

    def select_clients(self, round_idx):
        """Modified client selection to handle dynamic join/leave"""
        if self.client_dynamic_mode == 'join_leave' and round_idx > 0 and round_idx % self.join_leave_round_interval == 0:
            # Handle client leaving
            if self.num_leaving_clients > 0 and len(self.active_client_ids) > self.num_clients // 2:  # Ensure minimum clients remain
                clients_to_leave = random.sample(list(self.active_client_ids), 
                                              min(self.num_leaving_clients, len(self.active_client_ids) - self.num_clients // 2))
                for client_id in clients_to_leave:
                    self.active_client_ids.remove(client_id)
                    self.reserve_client_ids.add(client_id)
                    self.client_leave_history.append((round_idx, client_id, 'leave'))
                print(f"\nRound {round_idx}: Clients {clients_to_leave} left the federation.")
            
            # Handle client joining from reserve pool
            if self.num_joining_clients > 0 and len(self.reserve_client_ids) > 0:
                clients_to_join = random.sample(list(self.reserve_client_ids), 
                                             min(self.num_joining_clients, len(self.reserve_client_ids)))
                for client_id in clients_to_join:
                    self.active_client_ids.add(client_id)
                    self.reserve_client_ids.remove(client_id)
                    self.client_join_history.append((round_idx, client_id, 'join'))
                print(f"\nRound {round_idx}: Clients {clients_to_join} joined the federation.")

        # Update number of clients to select based on current active pool
        if self.random_join_ratio:
            self.current_num_join_clients = np.random.choice(
                range(self.num_join_clients, len(self.active_client_ids) + 1), 1, replace=False)[0]
        else:
            self.current_num_join_clients = min(self.num_join_clients, len(self.active_client_ids))

        # Select clients from active pool
        selected_ids = random.sample(list(self.active_client_ids), self.current_num_join_clients)
        self.selected_clients = [self.all_clients[i] for i in selected_ids]

        print(f"\nRound {round_idx}:")
        print(f"  Active clients: {len(self.active_client_ids)}")
        print(f"  Reserve clients: {len(self.reserve_client_ids)}")
        print(f"  Selected clients: {len(self.selected_clients)}")
        
        return self.selected_clients

    def send_models(self):
        assert (len(self.selected_clients) > 0) # Only send to selected clients

        for client in self.selected_clients: # Iterate over selected clients only
            start_time = time.time()
            
            print(f"\nDEBUG: Server sending model to client {client.id}")
           
            
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
            
            print(f"\nDEBUG: Server receiving model from client {client.id}")
            
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
            
        print("\nDEBUG: Server aggregating models")
        print(f"Number of models to aggregate: {len(self.uploaded_models)}")
        print(f"Model weights: {self.uploaded_weights}")
            
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
        """Modified to handle task-specific and overall evaluation for incremental learning"""
        # Only evaluate active clients
        active_clients = [self.all_clients[i] for i in self.active_client_ids]
        ids = [c.id for c in active_clients]
        num_samples_overall = [c.test_samples for c in active_clients]
        tot_correct_overall = []
        tot_auc_overall = []

        if self.eval_new_clients and self.num_new_clients > 0:
            self.fine_tuning_new_clients()
            return self.test_metrics_new_clients()

        print(f"\nEvaluating {len(active_clients)} active clients (IDs: {sorted(list(self.active_client_ids))})")

        # Iterate through active clients only for evaluation
        for c in active_clients:
            if self.task_type == 'class_incremental':
                tasks_to_eval = list(range(self.current_task_id + 1))
                ct, ns, auc = c.test_metrics(task_ids=tasks_to_eval)
            else:
                ct, ns, auc = c.test_metrics(task_ids=0)

            if ns > 0:
                tot_correct_overall.append(ct * 1.0)
                tot_auc_overall.append(auc * ns)
                num_samples_overall.append(ns)

        test_acc_overall = sum(tot_correct_overall) / sum(num_samples_overall) if sum(num_samples_overall) > 0 else 0
        test_auc_overall = sum(tot_auc_overall) / sum(num_samples_overall) if sum(num_samples_overall) > 0 else 0

        self.rs_test_acc.append(test_acc_overall)
        self.rs_test_auc.append(test_auc_overall)

        # Calculate task-specific accuracies
        if self.task_type == 'class_incremental':
            # Only evaluate tasks that exist in task_classes_map
            for t_id in range(min(self.current_task_id + 1, len(self.task_classes_map))):
                if t_id not in self.task_classes_map:
                    print(f"\nWarning: Task {t_id} not found in task_classes_map. Skipping evaluation.")
                    continue

                task_total_correct = 0
                task_total_samples = 0
                task_total_auc_weighted = 0

                # Only evaluate active clients for task-specific metrics
                for c in active_clients:
                    ct, ns, auc = c.test_metrics(task_ids=t_id)
                    task_total_correct += ct
                    task_total_samples += ns
                    task_total_auc_weighted += auc * ns

                if task_total_samples > 0:
                    task_acc = task_total_correct / task_total_samples
                    task_auc = task_total_auc_weighted / task_total_samples
                    
                    if t_id not in self.global_task_accuracies:
                        print(f"\nWarning: Initializing tracking for previously unknown task {t_id}")
                        self.global_task_accuracies[t_id] = []
                        self.global_task_aucs[t_id] = []
                    
                    self.global_task_accuracies[t_id].append(task_acc)
                    self.global_task_aucs[t_id].append(task_auc)
                else:
                    print(f"\nWarning: No samples found for task {t_id}")

        return ids, num_samples_overall, tot_correct_overall, tot_auc_overall # Return overall client metrics

    def train_metrics(self, current_task_id=0):
        """Modified to only evaluate active clients"""
        total_train_samples_valid = []
        total_train_losses_weighted = []

        # Only evaluate active clients
        active_clients = [self.all_clients[i] for i in self.active_client_ids]
        print(f"\nEvaluating training metrics for {len(active_clients)} active clients (IDs: {sorted(list(self.active_client_ids))})")

        for c in active_clients:
            cl, ns = c.train_metrics(current_task_id=current_task_id)
            if ns > 0:
                total_train_samples_valid.append(ns)
                total_train_losses_weighted.append(cl * 1.0)

        return sum(total_train_losses_weighted), sum(total_train_samples_valid)

    # Evaluate function, called periodically
    def evaluate(self, acc=None, loss=None, current_round_idx=None):
        stats = self.test_metrics(current_round_idx) # returns ids, num_samples_overall, tot_correct_overall, tot_auc_overall
        
        # FIX: train_metrics now returns aggregated loss and total samples
        aggregated_train_loss, total_train_samples = self.train_metrics(self.current_task_id) 

        test_acc = sum(stats[2])*1.0 / sum(stats[1]) if sum(stats[1]) > 0 else 0
        test_auc = sum(stats[3])*1.0 / sum(stats[1]) if sum(stats[1]) > 0 else 0
        train_loss = aggregated_train_loss / total_train_samples if total_train_samples > 0 else 0
        
        print("\nDEBUG: Evaluation Statistics:")
        print(f"  Total test samples: {sum(stats[1])}")
        print(f"  Total correct predictions: {sum(stats[2])}")
        print(f"  Individual client test samples: {stats[1]}")
        print(f"  Individual client correct predictions: {stats[2]}")
        print(f"  Individual client accuracies: {[c/n if n > 0 else 0 for c, n in zip(stats[2], stats[1])]}")
        
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

        print(f"\n------------- Task {self.current_task_id} Evaluation -------------")
        print("Averaged Train Loss on Current Task: {:.4f}".format(train_loss))
        
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
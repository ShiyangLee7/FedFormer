import copy
import torch
import numpy as np
import time
from flcore.clients.clientbase import Client


class clientAVG(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)

    def train(self, current_task_id=0):
        trainloader = self.load_train_data(current_task_id=current_task_id)
        
        # --- FIX: Check if trainloader is None (empty dataset) ---
        if trainloader is None: 
            print(f"Client {self.id} has no training data for task {current_task_id}. Skipping local training.")
            # Still update time costs to avoid ZeroDivisionError in server, but with 0 cost
            self.train_time_cost['num_rounds'] += 1
            return # Exit function if no data to train on

        if not trainloader.dataset or len(trainloader.dataset) == 0:
            print(f"Client {self.id} has empty dataset for task {current_task_id}. Skipping local training.")
            self.train_time_cost['num_rounds'] += 1
            return

        print(f"Client {self.id} starting training for task {current_task_id} with {len(trainloader.dataset)} samples")
        
        self.model.train()
        
        start_time = time.time()

        max_local_epochs = self.local_epochs
        if self.train_slow:
            max_local_epochs = np.random.randint(1, max_local_epochs // 2)

        for epoch in range(max_local_epochs):
            for i, (x, y) in enumerate(trainloader):
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                if self.train_slow:
                    time.sleep(0.1 * np.abs(np.random.rand()))
                output = self.model(x)
                loss = self.loss(output, y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        if self.learning_rate_decay:
            self.learning_rate_scheduler.step()

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time
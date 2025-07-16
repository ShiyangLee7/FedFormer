#!/usr/bin/env python
import copy
import torch
import argparse
import os
import time
import warnings
import numpy as np
import torchvision
import logging

from flcore.servers.serveravg import FedAvg
# from flcore.servers.serverpFedMe import pFedMe
# (other server imports)

from flcore.trainmodel.models import * 
from flcore.trainmodel.bilstm import *
from flcore.trainmodel.resnet import *
from flcore.trainmodel.alexnet import *
from flcore.trainmodel.mobilenet_v2 import *
from flcore.trainmodel.transformer import *

from utils.result_utils import average_data
from utils.mem_utils import MemReporter
from utils.data_utils import read_config # Import read_config to get task info
from flcore.clients.clientavg import clientAVG

logger = logging.getLogger()
logger.setLevel(logging.ERROR)

warnings.simplefilter("ignore")
torch.manual_seed(0)


def run(args):

    time_list = []
    reporter = MemReporter()
    model_str = args.model

    # Read dataset configuration to get task details and total classes
    dataset_config = read_config(args.dataset)
    task_type = dataset_config.get('task_type', 'iid_tasks')
    num_tasks = dataset_config.get('num_tasks', 1)
    # Update args.num_classes based on dataset config
    args.num_classes = dataset_config.get('num_classes', args.num_classes) 
    print(f"Detected total classes from config: {args.num_classes}")

    # Calculate rounds per task. Ensure at least 1 round per task.
    rounds_per_task = args.global_rounds // num_tasks if num_tasks > 0 else args.global_rounds
    if num_tasks > 0 and rounds_per_task == 0:
        rounds_per_task = 1 
        print(f"Warning: Global rounds ({args.global_rounds}) are fewer than tasks ({num_tasks}). Setting 1 round per task.")


    for i in range(args.prev, args.times): # Outer loop for multiple experimental runs
        print(f"\n============= Running time: {i}th =============")
        print("Creating server and clients ...")
        start = time.time()

        # Generate args.model (Model definition logic remains unchanged)
        if model_str == "MLR": 
            if "MNIST" in args.dataset:
                args.model = Mclr_Logistic(1*28*28, num_classes=args.num_classes).to(args.device)
            elif "Cifar10" in args.dataset:
                args.model = Mclr_Logistic(3*32*32, num_classes=args.num_classes).to(args.device)
            else:
                args.model = Mclr_Logistic(60, num_classes=args.num_classes).to(args.device)

        elif model_str == "CNN": 
            if "MNIST" in args.dataset:
                args.model = FedAvgCNN(in_features=1, num_classes=args.num_classes).to(args.device)
            elif "Cifar10" in args.dataset:
                args.model = FedAvgCNN(in_features=3, num_classes=args.num_classes, dim=1600).to(args.device)
            elif "Omniglot" in args.dataset:
                args.model = FedAvgCNN(in_features=1, num_classes=args.num_classes, dim=33856).to(args.device)
            elif "Digit5" in args.dataset:
                args.model = Digit5CNN().to(args.device)
            else:
                args.model = FedAvgCNN(in_features=3, num_classes=args.num_classes, dim=10816).to(args.device)

        elif model_str == "DNN": 
            if "MNIST" in args.dataset:
                args.model = DNN(1*28*28, 100, num_classes=args.num_classes).to(args.device)
            elif "Cifar10" in args.dataset:
                args.model = DNN(3*32*32, 100, num_classes=args.num_classes).to(args.device)
            else:
                args.model = DNN(60, 20, num_classes=args.num_classes).to(args.device)
        
        elif model_str == "ResNet18":
            args.model = torchvision.models.resnet18(pretrained=False, num_classes=args.num_classes).to(args.device)
        
        elif model_str == "ResNet10":
            args.model = resnet10(num_classes=args.num_classes).to(args.device)
        
        elif model_str == "ResNet34":
            args.model = torchvision.models.resnet34(pretrained=False, num_classes=args.num_classes).to(args.device)

        elif model_str == "AlexNet":
            args.model = alexnet(pretrained=False, num_classes=args.num_classes).to(args.device)
            
        elif model_str == "GoogleNet":
            args.model = torchvision.models.googlenet(pretrained=False, aux_logits=False, 
                                                      num_classes=args.num_classes).to(args.device)
            
        elif model_str == "MobileNet":
            args.model = mobilenet_v2(pretrained=False, num_classes=args.num_classes).to(args.device)
            
        elif model_str == "LSTM":
            args.model = LSTMNet(hidden_dim=args.feature_dim, vocab_size=args.vocab_size, num_classes=args.num_classes).to(args.device)

        elif model_str == "BiLSTM":
            args.model = BiLSTM_TextClassification(input_size=args.vocab_size, hidden_size=args.feature_dim, 
                                                   output_size=args.num_classes, num_layers=1, 
                                                   embedding_dropout=0, lstm_dropout=0, attention_dropout=0, 
                                                   embedding_length=args.feature_dim).to(args.device)

        elif model_str == "fastText":
            args.model = fastText(hidden_dim=args.feature_dim, vocab_size=args.vocab_size, num_classes=args.num_classes).to(args.device)

        elif model_str == "TextCNN":
            args.model = TextCNN(hidden_dim=args.feature_dim, max_len=args.max_len, vocab_size=args.vocab_size, 
                                 num_classes=args.num_classes).to(args.device)

        elif model_str == "Transformer":
            args.model = TransformerModel(ntoken=args.vocab_size, d_model=args.feature_dim, nhead=8, nlayers=2, 
                                          num_classes=args.num_classes, max_len=args.max_len).to(args.device)
        
        elif model_str == "AmazonMLP":
            args.model = AmazonMLP().to(args.device)

        elif model_str == "HARCNN":
            if args.dataset == 'HAR':
                args.model = HARCNN(9, dim_hidden=1664, num_classes=args.num_classes, conv_kernel_size=(1, 9), 
                                    pool_kernel_size=(1, 2)).to(args.device)
            elif args.dataset == 'PAMAP2':
                args.model = HARCNN(9, dim_hidden=3712, num_classes=args.num_classes, conv_kernel_size=(1, 9), 
                                    pool_kernel_size=(1, 2)).to(args.device)

        else:
            raise NotImplementedError

        print(args.model)

        if args.algorithm == "FedAvg":
            
            # Save the original fc layer
            args.head = copy.deepcopy(args.model.fc)
            
            # Replace fc with Identity
            args.model.fc = nn.Identity()
            
            # Create base model (without fc) and head (fc only)
            base_model = args.model
            head_model = args.head
            
            # Create the split model
            args.model = BaseHeadSplit(base_model, head_model)

            # Initialize server
            server = FedAvg(args, i) # args now contains new task/dynamic client info

        # other algorithms
        # elif args.algorithm == "Local":
        #     server = Local(args, i)
        # ...

        else:
            raise NotImplementedError

        # --- Main training loop modified for incremental tasks and client dynamics ---
        server.Budget = [] # Reset budget for each run
        
        # Initial evaluation before any training
        print(f"\n-------------Round number: 0 (Initial Evaluation before training)-------------")
        server.evaluate(current_round_idx=0) # Pass round_idx
        server.current_task_id = 0 # Ensure task ID is reset for each new run

        # Training loop starts from round 0
        for round_idx in range(args.global_rounds):
            s_t = time.time()
            
            # --- Task Progression Logic ---
            if task_type == 'class_incremental' and round_idx > 0 and round_idx % rounds_per_task == 0 and server.current_task_id < num_tasks:
                server.current_task_id += 1
                print(f"\n============= Advancing to Task {server.current_task_id} (Round {round_idx}) ==============")

            # Select and train clients
            server.selected_clients = server.select_clients(round_idx)
            server.send_models()

            # Local training on clients for the current task
            for client in server.selected_clients:
                client.train(current_task_id=server.current_task_id)

            # Receive and aggregate models
            server.receive_models()
            if server.dlg_eval and round_idx % server.dlg_gap == 0:
                server.call_dlg(round_idx)
            server.aggregate_parameters()

            # Evaluate after aggregation
            if round_idx % args.eval_gap == 0:
                print(f"\n-------------Round number: {round_idx}-------------")
                print("\nEvaluate global model")
                server.evaluate(current_round_idx=round_idx)

            server.Budget.append(time.time() - s_t)
            print('-'*25, 'time cost', '-'*25, server.Budget[-1])

            if server.auto_break and server.check_done(acc_lss=[server.rs_test_acc], top_cnt=server.top_cnt):
                break
        
        # After all global rounds (and tasks) are completed for this run
        print("\nBest accuracy.")
        print(max(server.rs_test_acc))
        print("\nAverage time cost per round.")
        print(sum(server.Budget[1:])/len(server.Budget[1:]))

        server.save_results() # Save all collected results (overall and task-specific)
        server.save_global_model() # Save final global model

        if args.num_new_clients > 0: # This is for a separate evaluation on completely new clients after training
            server.eval_new_clients = True
            server.set_new_clients(clientAVG)
            print(f"\n-------------Fine tuning round-------------")
            print("\nEvaluate new clients")
            server.evaluate() # This evaluate is for the fixed `new_clients` list

        time_list.append(time.time()-start) # Track total time for this run

    print(f"\nAverage total time cost for {args.times} runs: {round(np.average(time_list), 2)}s.")
    
    # Global average across multiple runs
    average_data(dataset=args.dataset, algorithm=args.algorithm, goal=args.goal, times=args.times)

    print("All done!")

    reporter.report()


if __name__ == "__main__":
    total_start = time.time()

    parser = argparse.ArgumentParser()
    # general
    
    parser.add_argument('-go', "--goal", type=str, default="test", 
                        help="The goal for this experiment")
    parser.add_argument('-dev', "--device", type=str, default="cuda",
                        choices=["cpu", "cuda"])
    parser.add_argument('-did', "--device_id", type=str, default="0")
    parser.add_argument('-data', "--dataset", type=str, default="MNIST")

    parser.add_argument('-ncl', "--num_classes", type=int, default=100, # 可以设置为100或一个更大的通用值
                    help="Total number of classes in the dataset (will be updated from dataset config)")
    parser.add_argument('-m', "--model", type=str, default="CNN")
    parser.add_argument('-lbs', "--batch_size", type=int, default=10)
    parser.add_argument('-lr', "--local_learning_rate", type=float, default=0.005,
                        help="Local learning rate")
    parser.add_argument('-ld', "--learning_rate_decay", type=bool, default=False)
    parser.add_argument('-ldg', "--learning_rate_decay_gamma", type=float, default=0.99)
    parser.add_argument('-gr', "--global_rounds", type=int, default=20,
                        help="Total communication rounds")
    parser.add_argument('-tc', "--top_cnt", type=int, default=100, 
                        help="For auto_break")
    parser.add_argument('-ls', "--local_epochs", type=int, default=20, 
                        help="Multiple update steps in one local epoch.")
    parser.add_argument('-algo', "--algorithm", type=str, default="FedAvg")
    parser.add_argument('-jr', "--join_ratio", type=float, default=1.0,
                        help="Ratio of clients per round")
    parser.add_argument('-rjr', "--random_join_ratio", type=bool, default=False,
                        help="Random ratio of clients per round")
    parser.add_argument('-nc', "--num_clients", type=int, default=10,
                        help="Initial number of active clients")
    parser.add_argument('-rc', "--reserve_clients", type=int, default=5,
                        help="Number of reserve clients that can join during training")
    parser.add_argument('-pv', "--prev", type=int, default=0,
                        help="Previous Running times")
    parser.add_argument('-t', "--times", type=int, default=1,
                        help="Running times")
    parser.add_argument('-eg', "--eval_gap", type=int, default=1,
                        help="Rounds gap for evaluation")
    parser.add_argument('-sfn', "--save_folder_name", type=str, default='items')
    parser.add_argument('-ab', "--auto_break", type=bool, default=False)
    parser.add_argument('-dlg', "--dlg_eval", type=bool, default=False)
    parser.add_argument('-dlgg', "--dlg_gap", type=int, default=100)
    parser.add_argument('-bnpc', "--batch_num_per_client", type=int, default=2)
    parser.add_argument('-nnc', "--num_new_clients", type=int, default=0) # For end-of-run eval
    parser.add_argument('-ften', "--fine_tuning_epoch_new", type=int, default=0)
    parser.add_argument('-fd', "--feature_dim", type=int, default=512)
    parser.add_argument('-vs', "--vocab_size", type=int, default=80, 
                        help="Set this for text tasks. 80 for Shakespeare. 32000 for AG_News and SogouNews.")
    parser.add_argument('-ml', "--max_len", type=int, default=200)
    parser.add_argument('-fs', "--few_shot", type=int, default=0)
    
    # --- New arguments for Task Sequence (II) ---
    parser.add_argument('-tt', "--task_type", type=str, default="iid_tasks", 
                        choices=["iid_tasks", "non_iid_tasks", "class_incremental"],
                        help="Type of task sequence (e.g., IID, Non-IID, Class-Incremental)")
    parser.add_argument('-cpt', "--classes_per_task", type=int, default=None,
                        help="Number of new classes per task for 'class_incremental' task_type")

    # --- New arguments for Client Dynamics (III) ---
    parser.add_argument('-cdm', "--client_dynamic_mode", type=str, default="none",
                        choices=["none", "join_leave"],
                        help="Mode for client dynamics: 'none' (fixed clients), 'join_leave'")
    parser.add_argument('-njc', "--num_joining_clients", type=int, default=0,
                        help="Number of new clients joining the federation at intervals (for 'join_leave' mode)")
    parser.add_argument('-nlc', "--num_leaving_clients", type=int, default=0,
                        help="Number of clients leaving the federation at intervals (for 'join_leave' mode)")
    parser.add_argument('-jlri', "--join_leave_round_interval", type=int, default=50,
                        help="Interval (in global rounds) at which clients join/leave")


    # practical (remain unchanged)
    parser.add_argument('-cdr', "--client_drop_rate", type=float, default=0.0,
                        help="Rate for clients that train but drop out")
    parser.add_argument('-tsr', "--train_slow_rate", type=float, default=0.0,
                        help="The rate for slow clients when training locally")
    parser.add_argument('-ssr', "--send_slow_rate", type=float, default=0.0,
                        help="The rate for slow clients when sending global model")
    parser.add_argument('-ts', "--time_select", type=bool, default=False,
                        help="Whether to group and select clients at each round according to time cost")
    parser.add_argument('-tth', "--time_threthold", type=float, default=10000,
                        help="The threthold for droping slow clients")
    # pFedMe / PerAvg / FedProx / FedAMP / FedPHP / GPFL / FedCAC (remain unchanged)
    parser.add_argument('-bt', "--beta", type=float, default=0.0)
    parser.add_argument('-lam', "--lamda", type=float, default=1.0,
                        help="Regularization weight")
    parser.add_argument('-mu', "--mu", type=float, default=0.0)
    parser.add_argument('-K', "--K", type=int, default=5,
                        help="Number of personalized training steps for pFedMe")
    parser.add_argument('-lrp', "--p_learning_rate", type=float, default=0.01,
                        help="personalized learning rate to caculate theta aproximately using K steps")
    # ... (rest of the arguments remain unchanged)
    parser.add_argument('-M', "--M", type=int, default=5, help="Server only sends M client models to one client at each round")
    parser.add_argument('-itk', "--itk", type=int, default=4000, help="The iterations for solving quadratic subproblems")
    parser.add_argument('-alk', "--alphaK", type=float, default=1.0, help="lambda/sqrt(GLOABL-ITRATION) according to the paper")
    parser.add_argument('-sg', "--sigma", type=float, default=1.0)
    parser.add_argument('-al', "--alpha", type=float, default=1.0) # Note: this alpha is for APFL/FedCross, not your STFC alpha.
    parser.add_argument('-pls', "--plocal_epochs", type=int, default=1)
    parser.add_argument('-tau', "--tau", type=float, default=1.0)
    parser.add_argument('-fte', "--fine_tuning_epochs", type=int, default=10)
    parser.add_argument('-dlr', "--dr_learning_rate", type=float, default=0.0)
    parser.add_argument('-L', "--L", type=float, default=1.0)
    parser.add_argument('-nd', "--noise_dim", type=int, default=512)
    parser.add_argument('-glr', "--generator_learning_rate", type=float, default=0.005)
    parser.add_argument('-hd', "--hidden_dim", type=int, default=512)
    parser.add_argument('-se', "--server_epochs", type=int, default=1000)
    parser.add_argument('-lf', "--localize_feature_extractor", type=bool, default=False)
    parser.add_argument('-slr', "--server_learning_rate", type=float, default=1.0)
    parser.add_argument('-et', "--eta", type=float, default=1.0)
    parser.add_argument('-s', "--rand_percent", type=int, default=80)
    parser.add_argument('-p', "--layer_idx", type=int, default=2, help="More fine-graind than its original paper.")
    parser.add_argument('-mlr', "--mentee_learning_rate", type=float, default=0.005)
    parser.add_argument('-Ts', "--T_start", type=float, default=0.95)
    parser.add_argument('-Te', "--T_end", type=float, default=0.98)
    parser.add_argument('-mo', "--momentum", type=float, default=0.1)
    parser.add_argument('-klw', "--kl_weight", type=float, default=0.0)
    parser.add_argument('-fsb', "--first_stage_bound", type=int, default=0)
    parser.add_argument('-ca', "--fedcross_alpha", type=float, default=0.99)
    parser.add_argument('-cmss', "--collaberative_model_select_strategy", type=int, default=1)


    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.device_id

    # 添加GPU设备检查
    if args.device == "cuda" and not torch.cuda.is_available():
        print("Warning: CUDA is not available, using CPU instead")
        args.device = "cpu"
    else:
        print(f"Using device: {args.device}")
        if args.device == "cuda":
            print(f"GPU Device Name: {torch.cuda.get_device_name(0)}")
            # 设置CUDA设备
            torch.cuda.set_device(int(args.device_id))

    print("=" * 50)
    for arg in vars(args):
        print(arg, '=',getattr(args, arg))
    print("=" * 50)

    # (Profiling section commented out as in original)
    # with torch.profiler.profile( ... ) as prof:
    # with torch.autograd.profiler.profile(profile_memory=True) as prof:
    run(args)
    
    # print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=20))
    # print(f"\nTotal time cost: {round(time.time()-total_start, 2)}s.")
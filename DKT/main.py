# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.

# Modified for DKT by Xinyuan Gao, thanks for the code of Arthur Douillard
import argparse
import copy
import datetime
import json
import os
import statistics
import time
import warnings
from pathlib import Path
import yaml

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from continuum.metrics import Logger
from continuum.tasks import split_train_val
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer

import continual.utils as utils
from continual import factory, scaler
from continual.classifier import Classifier
from continual.rehearsal import Memory, get_finetuning_dataset
from continual.datasets import build_dataset
from continual.engine import eval_and_log, train_one_epoch
from continual.losses import bce_with_logits, soft_bce_with_logits

warnings.filterwarnings("ignore")

local_rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local_rank)

def get_args_parser():
    parser = argparse.ArgumentParser('DKT training and evaluation script', add_help=False)
    parser.add_argument('--batch-size', default=128, type=int)
    parser.add_argument('--incremental-batch-size', default=None, type=int)
    parser.add_argument('--epochs', default=500, type=int)
    parser.add_argument('--base-epochs', default=500, type=int,
                        help='Number of epochs for base task')
    parser.add_argument('--no-amp', default=False, action='store_true',
                        help='Disable mixed precision')

    # Model parameters
    parser.add_argument('--model', default='')
    parser.add_argument('--input-size', default=32, type=int, help='images input size')
    parser.add_argument('--patch-size', default=16, type=int)
    parser.add_argument('--embed-dim', default=768, type=int)
    parser.add_argument('--depth', default=12, type=int)
    parser.add_argument('--num-heads', default=12, type=int)
    parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                        help='Dropout rate (default: 0.)')
    parser.add_argument('--drop-path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')
    parser.add_argument('--norm', default='layer', choices=['layer', 'scale'],
                        help='Normalization layer type')

    # Optimizer parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--opt-eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--clip-grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')

    # Learning rate schedule parameters
    parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',
                        help='LR scheduler (default: "cosine"')
    parser.add_argument('--lr', type=float, default=5e-4, metavar='LR',
                        help='learning rate (default: 5e-4)')
    parser.add_argument("--incremental-lr", default=None, type=float,
                        help="LR to use for incremental task (t > 0)")
    parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
                        help='learning rate noise on/off epoch percentages')
    parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
                        help='learning rate noise limit percent (default: 0.67)')
    parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
                        help='learning rate noise std-dev (default: 1.0)')
    parser.add_argument('--warmup-lr', type=float, default=1e-6, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--incremental-warmup-lr', type=float, default=None, metavar='LR',
                        help='warmup learning rate (default: 1e-6) for task T > 0')
    parser.add_argument('--min-lr', type=float, default=1e-5, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')
    parser.add_argument('--decay-epochs', type=float, default=30, metavar='N',
                        help='epoch interval to decay LR')
    parser.add_argument('--warmup-epochs', type=int, default=5, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--cooldown-epochs', type=int, default=10, metavar='N',
                        help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
    parser.add_argument('--patience-epochs', type=int, default=10, metavar='N',
                        help='patience epochs for Plateau LR scheduler (default: 10')
    parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                        help='LR decay rate (default: 0.1)')

    # Augmentation parameters
    parser.add_argument('--color-jitter', type=float, default=0.4, metavar='PCT',
                        help='Color jitter factor (default: 0.4)')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + \
                             "(default: rand-m9-mstd0.5-inc1)'),
    parser.add_argument('--smoothing', type=float, default=0.1, help='Label smoothing (default: 0.1)')
    parser.add_argument('--train-interpolation', type=str, default='bicubic',
                        help='Training interpolation (random, bilinear, bicubic default: "bicubic")')

    parser.add_argument('--repeated-aug', action='store_true')
    parser.add_argument('--no-repeated-aug', action='store_false', dest='repeated_aug')
    parser.set_defaults(repeated_aug=True)

    # * Random Erase params
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--resplit', action='store_true', default=False,
                        help='Do not random erase first (clean) augmentation split')

    # Distillation parameters
    parser.add_argument('--auto-kd', default=False, action='store_true',
                        help='Balance kd factor as WA https://arxiv.org/abs/1911.07053')
    parser.add_argument('--kd', default=0., type=float)
    parser.add_argument('--distillation-tau', default=1.0, type=float,
                        help='Temperature for the KD')
    parser.add_argument('--resnet', default=False, action='store_true')

    # Dataset parameters
    parser.add_argument('--data-path', default='', type=str,
                        help='dataset path')
    parser.add_argument('--data-set', default='IMNET', choices=['CIFAR', 'IMNET', 'INAT', 'INAT19'],
                        type=str, help='Image Net dataset path')
    parser.add_argument('--data-path-subTrain', default='', type=str,
                        help='dataset path subT')
    parser.add_argument('--data-path-subVal', default='', type=str,
                        help='dataset path subV')
    parser.add_argument('--inat-category', default='name',
                        choices=['kingdom', 'phylum', 'class', 'order', 'supercategory', 'family', 'genus', 'name'],
                        type=str, help='semantic granularity')
    parser.add_argument('--output-dir', default='',
                        help='Dont use that')
    parser.add_argument('--output-basedir', default='./checkponts/',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--dist-eval', action='store_true', default=False, help='Enabling distributed evaluation')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin-mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no-pin-mem', action='store_false', dest='pin_mem',
                        help='')
    parser.set_defaults(pin_mem=True)

    # Continual Learning parameters
    parser.add_argument("--initial-increment", default=50, type=int,
                        help="Base number of classes")
    parser.add_argument("--increment", default=10, type=int,
                        help="Number of new classes per incremental task")
    parser.add_argument('--class-order', default=None, type=int, nargs='+',
                        help='Class ordering, a list of class ids.')

    parser.add_argument("--eval-every", default=50, type=int,
                        help="Eval model every X epochs, if None only eval at the task end")
    parser.add_argument('--debug', default=False, action='store_true',
                        help='Only do one batch per epoch')
    parser.add_argument('--retrain-scratch', default=False, action='store_true',
                        help='Retrain from scratch on all data after each step (JOINT).')
    parser.add_argument('--max-task', default=None, type=int,
                        help='Max task id to train on')
    parser.add_argument('--name', default='', help='Name to display for screen')
    parser.add_argument('--options', default=[], nargs='*')

    # DKT related
    parser.add_argument('--DKT', action='store_true', default=False,
                        help='Enable super DKT god mode.')
    parser.add_argument('--duplex-clf', default=True,
                        help='Duplex Classifier')

    # Rehearsal memory
    parser.add_argument('--memory-size', default=2000, type=int,
                        help='Total memory size in number of stored (image, label).')
    parser.add_argument('--distributed-memory', default=False, action='store_true',
                        help='Use different rehearsal memory per process.')
    parser.add_argument('--global-memory', default=False, action='store_false', dest='distributed_memory',
                        help='Use same rehearsal memory for all process.')
    parser.set_defaults(distributed_memory=True)
    parser.add_argument('--oversample-memory', default=1, type=int,
                        help='Amount of time we repeat the same rehearsal.')
    parser.add_argument('--oversample-memory-ft', default=1, type=int,
                        help='Amount of time we repeat the same rehearsal for finetuning, only for old classes not new classes.')
    parser.add_argument('--rehearsal-test-trsf', default=False, action='store_true',
                        help='Extract features without data augmentation.')
    parser.add_argument('--rehearsal-modes', default=1, type=int,
                        help='Select N on a single gpu, but with mem_size/N.')
    parser.add_argument('--fixed-memory', default=False, action='store_true',
                        help='Dont fully use memory when no all classes are seen as in Hou et al. 2019')
    parser.add_argument('--rehearsal', default="random",
                        choices=[
                            'random',
                            'closest_token', 'closest_all',
                            'icarl_token', 'icarl_all',
                            'furthest_token', 'furthest_all'
                        ],
                        help='Method to herd sample for rehearsal.')
    parser.add_argument('--sep-memory', default=False, action='store_true',
                        help='Dont merge memory w/ task dataset but keep it alongside')
    parser.add_argument('--replay-memory', default=0, type=int,
                        help='Replay memory according to Guido rule [NEED DOC]')

    # Finetuning
    parser.add_argument('--finetuning', default='', choices=['balanced'],
                        help='Whether to do a finetuning after each incremental task. Backbone are frozen.')
    parser.add_argument('--finetuning-mode', default=30, type=int,
                        help='Number of epochs to spend in finetuning.')
    parser.add_argument('--finetuning-lr', default=5e-5, type=float,
                        help='LR during finetuning, will be kept constant.')
    parser.add_argument('--finetuning-teacher', default=False, action='store_true',
                        help='Use teacher/old model during finetuning for all kd related.')
    parser.add_argument('--finetuning-resetclf', default=False, action='store_true',
                        help='Reset classifier before finetuning phase (similar to GDumb/DER).')
    parser.add_argument('--only-ft', default=False, action='store_true',
                        help='Only train on FT data')
    parser.add_argument('--ft-no-sampling', default=False, action='store_true',
                        help='Dont use particular sampling for the finetuning phase.')

    # What to freeze
    parser.add_argument('--freeze-task', default=[], nargs="*", type=str,
                        help='What to freeze before every incremental task (t > 0).')
    parser.add_argument('--freeze-ft', default=[], nargs="*", type=str,
                        help='What to freeze before every finetuning (t > 0).')
    parser.add_argument('--freeze-eval', default=False, action='store_true',
                        help='Frozen layers are put in eval. Important for stoch depth')

    # Logs
    parser.add_argument('--log-path', default="logs")
    parser.add_argument('--log-category', default="misc")

    # Classification
    parser.add_argument('--bce-loss', default=False, action='store_true')

    # distributed training parameters
    parser.add_argument('--local_rank', default=None, type=int)
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

    # Resuming
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start-task', default=0, type=int, help='resume from checkpoint')
    parser.add_argument('--start-epoch', default=0, type=int, help='resume from checkpoint')
    parser.add_argument('--save-every-epoch', default=None, type=int)

    parser.add_argument('--validation', default=0.0, type=float,
                        help='Use % of the training set as val, replacing the test.')

    return parser


def main(args):
    print(args)
    logger = Logger(list_subsets=['train', 'test'])

    use_distillation = args.auto_kd
    device = torch.device(args.device)

    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    scenario_train, args.nb_classes = build_dataset(is_train=True, args=args)
    scenario_val, _ = build_dataset(is_train=False, args=args)

    model = factory.get_backbone(args)
    model.head = Classifier(
        model.embed_dim, args.nb_classes, args.initial_increment,
        args.increment, len(scenario_train)
    )
    model.to(device)
    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model_without_ddp.parameters() if p.requires_grad)

    if args.name:
        log_path = os.path.join(args.log_dir, f"logs_{args.trial_id}.json")
        long_log_path = os.path.join(args.log_dir, f"long_logs_{args.trial_id}.json")

        if utils.is_main_process():
            os.system("echo '\ek{}\e\\'".format(args.name))
            os.makedirs(args.log_dir, exist_ok=True)
            with open(os.path.join(args.log_dir, f"config_{args.trial_id}.json"), 'w+') as f:
                config = vars(args)
                config["nb_parameters"] = n_parameters
                json.dump(config, f, indent=2)
            with open(log_path, 'w+') as f:
                pass  # touch
            with open(long_log_path, 'w+') as f:
                pass  # touch
        log_store = {'results': {}}

        args.output_dir = os.path.join(
            args.output_basedir,
            f"{datetime.datetime.now().strftime('%y-%m-%d')}_{args.data_set}-{args.initial_increment}-{args.increment}_{args.name}_{args.trial_id}"
        )
    else:
        log_store = None
        log_path = long_log_path = None
    if args.output_dir and utils.is_main_process():
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    if args.distributed:
        torch.distributed.barrier()

    print('number of params:', n_parameters)

    loss_scaler = scaler.ContinualScaler(args.no_amp)


    if args.bce_loss:
        criterion = bce_with_logits

    teacher_model = None

    output_dir = Path(args.output_dir)

    memory = None
    if args.memory_size > 0:
        memory = Memory(
            args.memory_size, scenario_train.nb_classes, args.rehearsal, args.fixed_memory, args.rehearsal_modes
        )

    nb_classes = args.initial_increment
    base_lr = args.lr
    accuracy_list = []
    start_time = time.time()

    if args.debug:
        args.base_epochs = 1
        args.epochs = 1

    args.increment_per_task = [args.initial_increment] + [args.increment for _ in range(len(scenario_train) - 1)]

    dataset_true_val = None

    for task_id, dataset_train in enumerate(scenario_train):
        # if args.data_set.lower() == 'imagenet1000' and task_id > 7:
        # ## We have observed that some devices sometimes experience gradient explosions in ImageNet1000,  you can change the bs to 128 and resume the previous task model.
        #     args.batch_size = 128
        if args.max_task == task_id:
            print(f"Stop training because of max task")
            break
        print(f"Starting task id {task_id}/{len(scenario_train) - 1}")

        dataset_val = scenario_val[:task_id + 1]
        if args.validation > 0.:
            if task_id == 0:
                dataset_train, dataset_val = split_train_val(dataset_train, args.validation)
                dataset_true_val = dataset_val
            else:
                dataset_train, dataset_val = split_train_val(dataset_train, args.validation)
                dataset_true_val.concat(dataset_val)
            dataset_val = dataset_true_val

        for i in range(3):
            assert abs(dataset_train.trsf.transforms[-1].mean[i] - dataset_val.trsf.transforms[-1].mean[i]) < 0.0001
            assert abs(dataset_train.trsf.transforms[-1].std[i] - dataset_val.trsf.transforms[-1].std[i]) < 0.0001

        loader_memory = None
        if task_id > 0 and memory is not None:
            dataset_memory = memory.get_dataset(dataset_train)
            loader_memory = factory.InfiniteLoader(factory.get_train_loaders(
                dataset_memory, args,
                args.replay_memory if args.replay_memory > 0 else args.batch_size
            ))
            if not args.sep_memory:
                previous_size = len(dataset_train)

                for _ in range(args.oversample_memory):
                    dataset_train.add_samples(*memory.get())
                print(f"{len(dataset_train) - previous_size} samples added from memory.")

            if args.only_ft:
                dataset_train = get_finetuning_dataset(dataset_train, memory, 'balanced')

        if use_distillation and task_id > 0:
            teacher_model = copy.deepcopy(model_without_ddp)
            teacher_model.freeze(['all'])
            teacher_model.eval()

        if args.DKT:
            model_without_ddp = factory.update_DKT(model_without_ddp, task_id, args)

        if task_id > 0 and not args.DKT:
            model_without_ddp.head.add_classes()

        if task_id > 0:
            model_without_ddp.freeze(args.freeze_task)

        if args.retrain_scratch:
            model_without_ddp.init_params()
            dataset_train = scenario_train[:task_id + 1]

        loader_train, loader_val = factory.get_loaders(dataset_train, dataset_val, args)

        if task_id > 0 and args.incremental_batch_size:
            args.batch_size = args.incremental_batch_size

        if args.incremental_lr is not None and task_id > 0:
            linear_scaled_lr = args.incremental_lr * args.batch_size * utils.get_world_size() / 256.0
        else:
            linear_scaled_lr = base_lr * args.batch_size * utils.get_world_size() / 256.0

        args.lr = linear_scaled_lr
        optimizer = create_optimizer(args, model_without_ddp)
        lr_scheduler, _ = create_scheduler(args, optimizer)
        # ----------------------------------------------------------------------

        skipped_task = False
        initial_epoch = epoch = 0
        if args.resume and args.start_task > task_id:
            utils.load_first_task_model(model_without_ddp, loss_scaler, task_id, args)
            print("Skipping first task")
            epochs = 0
            train_stats = {"task_skipped": str(task_id)}
            skipped_task = True
        elif args.base_epochs is not None and task_id == 0:
            epochs = args.base_epochs
        else:
            epochs = args.epochs

        if args.distributed:
            del model
            model = torch.nn.parallel.DistributedDataParallel(
                model_without_ddp, device_ids=[args.gpu], find_unused_parameters=True)
            torch.distributed.barrier()
        else:
            model = model_without_ddp

        model_without_ddp.nb_epochs = epochs
        model_without_ddp.nb_batch_per_epoch = len(loader_train)

        print(f"Start training for {epochs - initial_epoch} epochs")
        max_accuracy = 0.0
        for epoch in range(initial_epoch, epochs):
            if args.distributed:
                loader_train.sampler.set_epoch(epoch)

            train_stats = train_one_epoch(
                model, criterion, loader_train,
                optimizer, device, epoch, task_id, loss_scaler,
                args.clip_grad,
                debug=args.debug,
                args=args,
                teacher_model=teacher_model,
                model_without_ddp=model_without_ddp,
                loader_memory=loader_memory
            )

            lr_scheduler.step(epoch)

            if args.save_every_epoch is not None and epoch % args.save_every_epoch == 0:
                if os.path.isdir(args.resume):
                    with open(os.path.join(args.resume, 'save_log.txt'), 'w+') as f:
                        f.write(f'task={task_id}, epoch={epoch}\n')

                    checkpoint_paths = [os.path.join(args.resume, f'checkpoint_{task_id}.pth')]
                    for checkpoint_path in checkpoint_paths:
                        if (task_id < args.start_task and args.start_task > 0) and os.path.isdir(
                                args.resume) and os.path.exists(checkpoint_path):
                            continue

                        utils.save_on_master({
                            'model': model_without_ddp.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'lr_scheduler': lr_scheduler.state_dict(),
                            'epoch': epoch,
                            'task_id': task_id,
                            'scaler': loss_scaler.state_dict(),
                            'args': args,
                        }, checkpoint_path)

            if args.eval_every and (epoch % args.eval_every == 0 or (args.finetuning and epoch == epochs - 1)):
                eval_and_log(
                    args, output_dir, model, model_without_ddp, optimizer, lr_scheduler,
                    epoch, task_id, loss_scaler, max_accuracy,
                    [], n_parameters, device, loader_val, train_stats, None, long_log_path,
                    logger, model_without_ddp.epoch_log()
                )
                logger.end_epoch()

        if memory is not None and args.distributed_memory:
            task_memory_path = os.path.join(args.resume, f'dist_memory_{task_id}.npz')
            if torch.distributed.get_rank() > 0:
                torch.distributed.barrier()
            elif torch.distributed.get_rank() == 0:
                if os.path.isdir(args.resume) and os.path.exists(task_memory_path):
                    memory.load(task_memory_path)
                else:
                    task_set_to_rehearse = scenario_train[task_id]
                    if args.rehearsal_test_trsf:
                        task_set_to_rehearse.trsf = scenario_val[task_id].trsf

                    memory.add(task_set_to_rehearse, model, args.initial_increment if task_id == 0 else args.increment)

                    if args.resume != '':
                        memory.save(task_memory_path)
                    else:
                        memory.save(os.path.join(args.output_dir, f'dist_memory_{task_id}.npz'))
                torch.distributed.barrier()

        if torch.distributed.get_rank() > 0:
            task_memory_path = os.path.join(args.output_dir, f'dist_memory_{task_id}.npz')
            memory.load(task_memory_path)

        # FINETUNING

        if args.finetuning and memory and (
                task_id > 0 or scenario_train.nb_classes == args.initial_increment) and not skipped_task:
            dataset_finetune = get_finetuning_dataset(dataset_train, memory, args.finetuning, args.oversample_memory_ft,
                                                      task_id)
            print(f'Finetuning phase of type {args.finetuning} with {len(dataset_finetune)} samples.')

            loader_finetune, loader_val = factory.get_loaders(dataset_finetune, dataset_val, args, finetuning=True)
            print(f'Train-ft and val loaders of lengths: {len(loader_finetune)} and {len(loader_val)}.')
            if args.finetuning_resetclf:
                model_without_ddp.reset_classifier()

            model_without_ddp.freeze(args.freeze_ft)

            if args.distributed:
                del model
                model = torch.nn.parallel.DistributedDataParallel(model_without_ddp, device_ids=[args.gpu],
                                                                  find_unused_parameters=True)
                torch.distributed.barrier()
            else:
                model = model_without_ddp

            model_without_ddp.begin_finetuning()

            if args.increment == 100:  # because the ImageNet1000 use four GPUs
                args.lr = args.finetuning_lr * args.batch_size * utils.get_world_size() / 512.0
            else:
                args.lr = args.finetuning_lr * args.batch_size * utils.get_world_size() / 256.0
            optimizer = create_optimizer(args, model_without_ddp)
            # the fine-tuning is divided into two parts, the first part is the first 30% steps, the second part is the remaining steps
            if args.data_set.lower() == 'cifar':
                if args.increment == 20:
                    if task_id < 2:
                        ft_ep = 1
                    else:
                        ft_ep = 10
                elif args.increment == 10:
                    if task_id < 3:
                        ft_ep = 1
                    else:
                        ft_ep = 10
                else:
                    if task_id < 6:
                        ft_ep = 1
                    else:
                        ft_ep = 10
            else:
                if args.increment == 10:
                    if task_id < 3:
                        ft_ep = 5
                    else:
                        ft_ep = 10
                else:
                    if task_id < 3:
                        ft_ep = 5
                    else:
                        ft_ep = 10
            for epoch in range(ft_ep):
                if args.distributed and hasattr(loader_finetune.sampler, 'set_epoch'):
                    loader_finetune.sampler.set_epoch(epoch)
                train_stats = train_one_epoch(
                    model, criterion, loader_finetune,
                    optimizer, device, epoch, task_id, loss_scaler,
                    args.clip_grad,
                    debug=args.debug,
                    args=args,
                    teacher_model=teacher_model if args.finetuning_teacher else None,
                    model_without_ddp=model_without_ddp
                )

                if epoch % 10 == 0 or epoch == ft_ep - 1:
                    eval_and_log(
                        args, output_dir, model, model_without_ddp, optimizer, lr_scheduler,
                        epoch, task_id, loss_scaler, max_accuracy,
                        [], n_parameters, device, loader_val, train_stats, None, long_log_path,
                        logger, model_without_ddp.epoch_log()
                    )
                    logger.end_epoch()

            model_without_ddp.end_finetuning()

        eval_and_log(
            args, output_dir, model, model_without_ddp, optimizer, lr_scheduler,
            epoch, task_id, loss_scaler, max_accuracy,
            accuracy_list, n_parameters, device, loader_val, train_stats, log_store, log_path,
            logger, model_without_ddp.epoch_log(), skipped_task
        )
        logger.end_task()

        nb_classes += args.increment

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))
    print(f'Setting {args.data_set} with {args.initial_increment}-{args.increment}')
    print(f"All accuracies: {accuracy_list}")
    print(f"Average Incremental Accuracy: {statistics.mean(accuracy_list)}")
    if args.name:
        print(f"Experiment name: {args.name}")
        log_store['summary'] = {"avg": statistics.mean(accuracy_list)}
        if log_path is not None and utils.is_main_process():
            with open(log_path, 'a+') as f:
                f.write(json.dumps(log_store['summary']) + '\n')


def load_options(args, options):
    varargs = vars(args)

    name = []
    for o in options:
        with open(o) as f:
            new_opts = yaml.safe_load(f)

        for k, v in new_opts.items():
            if k not in varargs:
                raise ValueError(f'Option {k}={v} doesnt exist!')
        varargs.update(new_opts)
        name.append(o.split("/")[-1].replace('.yaml', ''))

    return '_'.join(name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('DKT training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    utils.init_distributed_mode(args)

    if args.options:
        name = load_options(args, args.options)
        if not args.name:
            args.name = name

    args.log_dir = os.path.join(
        args.log_path, args.data_set.lower(), args.log_category,
        datetime.datetime.now().strftime('%y-%m'),
        f"week-{int(datetime.datetime.now().strftime('%d')) // 7 + 1}",
        f"{int(datetime.datetime.now().strftime('%d'))}_{args.name}"
    )

    if isinstance(args.class_order, list) and isinstance(args.class_order[0], list):
        print(f'Running {len(args.class_order)} different class orders.')
        class_orders = copy.deepcopy(args.class_order)

        for i, order in enumerate(class_orders, start=1):
            print(f'Running class ordering {i}/{len(class_orders)}.')
            args.trial_id = i
            args.class_order = order
            main(args)
    else:
        args.trial_id = 1
        main(args)

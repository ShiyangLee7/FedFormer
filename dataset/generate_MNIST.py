import numpy as np
import os
import sys
import random
import torch
import torchvision
import torchvision.transforms as transforms
from utils.dataset_utils import check, separate_data, split_data, save_file

random.seed(1)
np.random.seed(1)
num_clients = 20
dir_path = "MNIST/"

# Allocate data to users
def generate_dataset(dir_path, num_clients, niid, balance, partition):
    # 检查目录是否存在，不存在则创建
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        
    # 设置训练和测试数据目录
    config_path = dir_path + "config.json" # 记录数据分配，包括数据集的划分方式，客户端数量，是否iid，是否balance等
    train_path = dir_path + "train/"
    test_path = dir_path + "test/"

    if check(config_path, train_path, test_path, num_clients, niid, balance, partition):
        '''
        检查是否已经生成过数据集，如果config.json存在，并且里面的参数与当前参数一致，则不重新生成，直接return
        如果config.json不存在或里面的参数与当前参数不一致，则下载数据集，并重新生成config.json
        config.json的参数:
        num_clients: 客户端数量
        non_iid: 是否非独立同分布
        balance: 是否平衡
        partition: 数据集划分方式
        Size of samples for labels in clients: 每个客户端的每个标签的样本数，每个客户端的每个标签的样本数是一个列表，列表中每个元素是一个二元组，第一个元素是标签，第二个元素是样本数
        alpha: 迪利克雷分布的参数，用于控制每个客户端的每个标签的样本数
        '''
        return

    # Get MNIST data
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])

    trainset = torchvision.datasets.MNIST(
        root=dir_path+"rawdata", train=True, download=True, transform=transform)
    testset = torchvision.datasets.MNIST(
        root=dir_path+"rawdata", train=False, download=True, transform=transform)
    
    # 设置上和常规联邦学习略有不同，常规联邦学习是每个客户端的样本数是相同的，而这里每个客户端的样本数是不同的
    # PFL和联邦增量学习都采用这种划分方式
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=len(trainset.data), shuffle=False)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=len(testset.data), shuffle=False)

    # 将数据部分和标签部分分开
    for _, train_data in enumerate(trainloader, 0):
        trainset.data, trainset.targets = train_data
    for _, test_data in enumerate(testloader, 0):
        testset.data, testset.targets = test_data

    dataset_image = []
    dataset_label = []

    # 把trainset和testset的数据合并
    dataset_image.extend(trainset.data.cpu().detach().numpy())
    dataset_image.extend(testset.data.cpu().detach().numpy())
    # 把trainset和testset的标签合并
    dataset_label.extend(trainset.targets.cpu().detach().numpy())
    dataset_label.extend(testset.targets.cpu().detach().numpy())

    # 将数据部分和标签部分转换为numpy数组
    dataset_image = np.array(dataset_image)
    dataset_label = np.array(dataset_label)

    num_classes = len(set(dataset_label))
    print(f'Number of classes: {num_classes}')

    # X:记录了每个client拥有的数据的原始内容的索引
    # y:记录了每个client拥有的数据的原始标签的索引
    # statistic:记录了每个client拥有的数据的原始标签的分布（每个标签的样本数）
    X, y, statistic = separate_data((dataset_image, dataset_label), num_clients, num_classes, 
                                    niid, balance, partition, class_per_client=2)
    train_data, test_data = split_data(X, y)
    save_file(config_path, train_path, test_path, train_data, test_data, num_clients, num_classes, 
        statistic, niid, balance, partition)


if __name__ == "__main__":
    niid = True if sys.argv[1] == "noniid" else False
    balance = True if sys.argv[2] == "balance" else False
    partition = sys.argv[3] if sys.argv[3] != "-" else None
    #  partition: "iid", "noniid" 迪利克雷等划分方式

    generate_dataset(dir_path, num_clients, niid, balance, partition)
import os
import torch
import csv
from tqdm import tqdm

def init_setting(seed, alg, dataset, name):
    torch.manual_seed(seed)   
    if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint')
    if not os.path.isdir('results'):
        os.mkdir('results')
    filename = (alg + "_" + dataset + "_" + name)

    pth_name = ('checkpoint/' + filename)
    res_name = ('results/' + filename + '_res.csv')

    tab_name = ('results/' +  alg + '.csv')
    if not os.path.exists(tab_name):
        with open(tab_name, 'w') as logfile:
            logwriter = csv.writer(logfile, delimiter=',')
            logwriter.writerow(["Alg", "Data", "OOD", "Name", "seed", "epoch",
                "Train Acc", "Test Acc", "auroc", "auprIn", "auprOut", "tpr95", "detection", "ECE", "ys"])

    return pth_name, res_name

def save_model(pth_name, net):
    pth_name = (pth_name + '.pth')
    torch.save(net.state_dict(), pth_name)

def load_model(pth_name, net):
    pth_name = (pth_name + '.pth')
    net.load_state_dict(torch.load(pth_name))

def train_model(epoch, trainloader, net, optimizer, criterion):
    net.train()
    train_loss_list = []

    for i in range(epoch):
        train_loss = 0
        correct = 0
        total = 0
        for idx, (inputs, targets) in enumerate(trainloader):
            data, label = inputs.cuda(), targets.cuda()
            optimizer.zero_grad()
            outputs = net(data)
            loss = criterion(outputs, label)
            train_loss += loss.item()
            loss.backward()
            optimizer.step_and_update_lr()

        train_loss = train_loss/(idx + 1)
        train_loss_list.append(train_loss)

    return train_loss_list
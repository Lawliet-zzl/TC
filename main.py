import Ndata
import os
import copy
import csv
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import argparse
import evaluation
from tqdm import tqdm
from transformer.Models import Transformer
from transformer.Optim import ScheduledOptim
from sklearn.metrics import accuracy_score, precision_score, recall_score

__author__ = "Zhilin Zhao"

parser = argparse.ArgumentParser(description='PyTorch')
parser.add_argument('-seed', default=42, type=int, help='random seed')

parser.add_argument('-d_feature', type=int, default=8)
parser.add_argument('-d_model', type=int, default=128, help='d_k * n_head 64 * 8 = 512')
parser.add_argument('-d_inner_hid', type=int, default=0)
parser.add_argument('-d_k', type=int, default=16, help='64 for NLP')
parser.add_argument('-d_v', type=int, default=16, help='64 for NLP')
parser.add_argument('-n_head', type=int, default=8)
parser.add_argument('-n_layers', type=int, default=6)
parser.add_argument('-dropout', type=float, default=0.1)
parser.add_argument('-scale_emb_or_prj', type=str, default='prj')
parser.add_argument('-warmup','--n_warmup_steps', type=int, default=4000)
parser.add_argument('-lr_mul', type=float, default=2.0, help='2.0')
parser.add_argument('-d_y', type=int, default=0)

parser.add_argument('-dataset', type=str, default='iris')
parser.add_argument('-OOD', type=int, default=-1)
parser.add_argument('-batch_size', type=int, default=128)
parser.add_argument('-epoch', type=int, default=1)
parser.add_argument('-name', type=str, default='0')
parser.add_argument('-precision', default=100000, type=float)

parser.add_argument('-q', type=float, default=0.5)
parser.add_argument('-temp', type=float, default=1000)
parser.add_argument('-alg', type=str, default='TC')
parser.add_argument('-order', type=int, default=100)

args = parser.parse_args()

class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, eps=0.1, reduction='mean'):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.eps = eps
        self.reduction = reduction

    def forward(self, output, target):
        c = output.size()[-1]
        log_preds = F.log_softmax(output, dim=-1)
        if self.reduction=='sum':
            loss = -log_preds.sum()
        else:
            loss = -log_preds.sum(dim=-1)
            if self.reduction=='mean':
                loss = loss.mean()
        return loss*self.eps/c + (1-self.eps) * F.nll_loss(log_preds, target, reduction=self.reduction)

def init_setting():
    torch.manual_seed(args.seed)   
    if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint')
    if not os.path.isdir('results'):
        os.mkdir('results')
    filename = (args.alg + "_" + args.dataset + "_" + args.name)

    pth_name = ('checkpoint/' + filename)
    index_name = ('results/' + filename + '_index.csv')
    res_name = ('results/' + filename + '_res.csv')
    return pth_name, index_name, res_name

def save_model(pth_name, net, c):
    pth_name = (pth_name + '_' + str(c) + '.pth')
    torch.save(net.state_dict(), pth_name)

def load_model(pth_name, net, c):
    pth_name = (pth_name + '_' + str(c) + '.pth')
    net.load_state_dict(torch.load(pth_name))

def test_model(testloader, w_index, s_index, net):
    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for idx, (inputs, targets) in enumerate(testloader):
            data, label = inputs.cuda(), targets.cuda()
    
            w_data = data[:,w_index]
            s_data = data[:,s_index]
            outputs = net(w_data,s_data)
            _, predicted = torch.max(outputs.data, 1)
            total += label.size(0)
            correct += predicted.eq(label).sum().item()
        acc = 100.*correct/total
    return acc

def test_model_ens(testloader, index_list, nets, c):
    correct = 0
    total = 0
    with torch.no_grad():
        for idx, (inputs, targets) in enumerate(testloader):
            data, label = inputs.cuda(), targets.cuda()

            outputs_ens = 0
            for i in range(c):
                w_data = data[:,index_list[0][i]]
                s_data = data[:,index_list[1][i]]
                net = nets[i]
                net.eval()
                outputs = net(w_data,s_data)
                outputs = F.softmax(outputs.data, dim=1)
                outputs_ens += outputs

            _, predicted = torch.max(outputs_ens.data / c, dim=1)
            total += label.size(0)
            correct += predicted.eq(label).sum().item()
        acc = 100.*correct/total
    return acc

def OOD_score_MSP(dataloader, w_index, s_index, net):
    net.eval()
    res = np.array([])
    with torch.no_grad():
        for idx, (inputs, targets) in enumerate(dataloader):
            data, label = inputs.cuda(), targets.cuda()
            w_data = data[:,w_index]
            s_data = data[:,s_index]
            outputs = net(w_data,s_data)
            softmax_vals, predicted = torch.max(F.softmax(outputs.data / args.temp, dim=1), dim=1)
            res = np.append(res, softmax_vals.cpu().numpy())
    return res 

def OOD_score_MSP_ens(dataloader, index_list, nets, c):
    res = np.array([])
    with torch.no_grad():
        for idx, (inputs, targets) in enumerate(dataloader):
            data, label = inputs.cuda(), targets.cuda()
            softmax_vals_ens = 0
            for i in range(c):
                w_data = data[:,index_list[0][i]]
                s_data = data[:,index_list[1][i]]
                net = nets[i]
                net.eval()
                outputs = net(w_data,s_data)
                softmax_vals = F.softmax(outputs.data / args.temp, dim=1)
                softmax_vals_ens += softmax_vals
            softmax_vals_ens = torch.max(softmax_vals_ens / c, dim=1)
            res = np.append(res, softmax_vals_ens[0].cpu().numpy())
    return res

# def OOD_score_MSP_ens(dataloader, index_list, nets, c):
#     res = np.array([])
#     with torch.no_grad():
#         for idx, (inputs, targets) in enumerate(dataloader):
#             data, label = inputs.cuda(), targets.cuda()
#             softmax_vals_ens = 0
#             for i in range(c):
#                 w_data = data[:,index_list[0][i]]
#                 s_data = data[:,index_list[1][i]]
#                 net = nets[i]
#                 net.eval()
#                 outputs = net(w_data,s_data)
#                 softmax_vals = outputs.data
#                 softmax_vals_ens += softmax_vals
#             softmax_vals_ens = torch.max(F.softmax(softmax_vals_ens.data /  args.temp, dim=1), dim=1)[0]
#             res = np.append(res, softmax_vals_ens.cpu().numpy())
#     return res

class ECELoss(nn.Module):
    def __init__(self, n_bins=20, temperature = 1):
        super(ECELoss, self).__init__()
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]
        self.temperature = temperature
    def forward(self, logits, labels):
        confidences, predictions = torch.max(logits, 1)
        accuracies = predictions.eq(labels)
        ece = torch.zeros(1).cuda()
        ys = []
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
                accuracy_in_bin = accuracy_in_bin.item()
            else:
                accuracy_in_bin = 0
            ys.append(accuracy_in_bin)
        return ece, ys

def Calibrated(dataloader, w_index, s_index, net, bins=20, temperature = 1):
    net.eval()
    ece_criterion = evaluation.ECELoss(n_bins=bins, temperature = temperature).cuda()

    logits_list = []
    labels_list = []
    with torch.no_grad():
        for idx, (inputs, targets) in enumerate(dataloader):
            inputs= inputs.cuda()
            w_data = inputs[:,w_index]
            s_data = inputs[:,s_index]
            outputs = net(w_data,s_data)
            softmaxes = F.softmax(outputs, dim=1)
            logits_list.append(softmaxes.cpu())
            labels_list.append(targets.cpu())
        logits = torch.cat(logits_list).cuda()
        labels = torch.cat(labels_list).cuda()
    ece, ys = ece_criterion(logits, labels)
    ece = ece.item()
    return ece, ys

def Calibrated_ens(dataloader, index_list, nets, c, bins=20, temperature = 1):
    ece_criterion = evaluation.ECELoss(n_bins=bins, temperature = temperature).cuda()
    logits_list = []
    labels_list = []
    with torch.no_grad():
        for idx, (inputs, targets) in enumerate(dataloader):
            inputs= inputs.cuda()
            softmaxes = 0
            for i in range(c):
                w_data = inputs[:,index_list[0][i]]
                s_data = inputs[:,index_list[1][i]]
                net = nets[i]
                net.eval()
                outputs = net(w_data,s_data)
                softmax_vals = F.softmax(outputs.data, dim=1)
                softmaxes += softmax_vals
            softmaxes = softmaxes / c

            logits_list.append(softmaxes.cpu())
            labels_list.append(targets.cpu())
        logits = torch.cat(logits_list).cuda()
        labels = torch.cat(labels_list).cuda()
    ece, ys = ece_criterion(logits, labels)
    ece = ece.item()
    return ece, ys

def train_model(trainloader, w_index, s_index, net, optimizer, criterion):
    net.train()
    train_loss_list = []
    train_acc_list = []
    ave_losses = []

    for epoch in range(0, args.epoch):
        train_loss = 0
        correct = 0
        total = 0
        for idx, (inputs, targets) in enumerate(trainloader):
            data, label = inputs.cuda(), targets.cuda()

            optimizer.zero_grad()
            w_data = data[:,w_index]
            s_data = data[:,s_index]
            outputs = net(w_data, s_data)
            loss = criterion(outputs, label)
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += label.size(0)
            correct += (predicted == label).sum().item()
            loss.backward()
            optimizer.step_and_update_lr()
            # torch.cuda.empty_cache()
            # net.zero_grad()

        train_loss = train_loss/(idx + 1)
        train_acc = 100.*correct/total
        train_loss_list.append(train_loss)
        train_acc_list.append(train_acc)

        # if (epoch + 1) % 10 == 0:
        #     print("Epoch: ", epoch + 1, ", Training Loss: ", train_loss, ", Training Acc: ", train_acc)

    return train_loss_list, train_acc_list

def write_index(index_list, index_name):
    with open(index_name, 'w') as logfile:
        logwriter = csv.writer(logfile, delimiter=',')
        for i in range(len(index_list[0])):
            logwriter.writerow(index_list[0][i])
            logwriter.writerow(index_list[1][i])

def write_res(results, detection_results_s, losses, train_acc_s, ece, ys, res_name):
    with open(res_name, 'w') as logfile:
        logwriter = csv.writer(logfile, delimiter=',')
        for i in range(len(losses)):
            logwriter.writerow(losses[i])
        for i in range(len(train_acc_s)):
            logwriter.writerow(train_acc_s[i])
        logwriter.writerow(["Chain","Train Acc", "Test Acc", "auroc", "auprIn", "auprOut", "tpr95", "detection" , "ECE", "ys"])
        for i in range(int(len(results) / 2)):
            row = [str(i + 1), str(results[i*2]), str(results[i*2 + 1]), 
            detection_results_s[i][0], detection_results_s[i][1], detection_results_s[i][2], 
            detection_results_s[i][3], detection_results_s[i][4], ece[i], ys[i]]
            logwriter.writerow(row)

def init_list(dim):
    w_index = [True for n in range(dim)]
    s_index = [False for n in range(dim)]
    w_index_list = []
    s_index_list = []
    w_index_list.append(w_index)
    s_index_list.append(s_index)
    index_list = []
    index_list.append(w_index_list)
    index_list.append(s_index_list)
    return index_list

def load_index(index_name):
    f = csv.reader(open(index_name,'r'))
    w_index_list = []
    s_index_list = []
    cnt = 0
    for line in f:
        tmp = []
        for x in line:
            if x == 'True':
                tmp.append(True)
            else:
                tmp.append(False)
        if cnt % 2 == 0:
            w_index_list.append(tmp)
        else:
            s_index_list.append(tmp)
        cnt += 1
    index_list = []
    index_list.append(w_index_list)
    index_list.append(s_index_list)
    return index_list

def update_list(index_list, c, weights):
    weights = weights.cpu().detach().numpy()
    # num_features = int(len(weights) / 2)
    num_features = int(len(weights) * (1 - args.q))
    if num_features == 0:
        num_features = 1

    feature_index = np.argsort(weights)
    # feature_index = np.argsort(weights)[::-1]
    w_index_list = copy.deepcopy(index_list[0][c - 1])  # last one
    s_index_list = copy.deepcopy(index_list[1][0])  # first one

    for i in range(num_features):
        index = feature_index[i]
        cnt = -1
        for j in range(len(w_index_list)):
            cnt += int(index_list[0][c - 1][j])
            if cnt == index:
                w_index_list[j] = False
                s_index_list[j] = True
                break

    index_list[0].append(w_index_list)
    index_list[1].append(s_index_list)

def write_res_table(c, acc, detection_result, ece, ys):
    filename = ('results/' +  args.alg + '.csv')
    if not os.path.exists(filename):
        with open(filename, 'w') as logfile:
            logwriter = csv.writer(logfile, delimiter=',')
            logwriter.writerow(["Data", "OOD", "Name", "q", "feature", "dy", "epoch", "seed", "chain", "Test Acc", 
                "auroc", "auprIn", "auprOut", "tpr95", "detection", "ECE", "ys"])

    with open(filename, 'a') as logfile:
        logwriter = csv.writer(logfile, delimiter=',')
        logwriter.writerow([args.dataset, args.OOD, args.name, args.q, args.d_feature, args.d_y,
            args.epoch, args.seed, c, acc, 
            detection_result[0], detection_result[1], detection_result[2], detection_result[3], detection_result[4], 
            ece, ys])
def main():
    pth_name, index_name, res_name = init_setting()
    criterion =  nn.CrossEntropyLoss()
    # criterion = LabelSmoothingCrossEntropy()

    trainloader, testloader, OODloader, X_dim, y_dim = Ndata.get_loader_IOD(args.dataset, args.OOD, batch_size = args.batch_size)
    index_list = init_list(X_dim)
    c = 0
    results = []
    losses = []
    train_acc_s = []
    detection_result_s = []
    nets = []
    ece_list = []
    ys_list = []

    # print("Features: ", X_dim, ", labels: ", y_dim, ", epoch: ", args.epoch)
    print("Training: " + pth_name + ", Seed: " + str(args.seed))

    if args.d_y != 0:
        y_dim = args.d_y

    while(True):
        # print("Chain:", c + 1, index_list[0][c], index_list[1][c])
        net = Transformer(sum(index_list[0][c]), sum(index_list[1][c]), 
            num_feature = args.d_feature, num_outputs = y_dim, src_pad_idx=0, trg_pad_idx=0,
            d_k=args.d_k, d_v=args.d_v, d_model=args.d_model, d_inner=args.d_inner_hid,
            n_layers=args.n_layers, n_head=args.n_head, dropout=args.dropout,
            scale_emb_or_prj=args.scale_emb_or_prj).cuda()

        optimizer = ScheduledOptim(
            optim.Adam(net.parameters(), betas=(0.9, 0.98), eps=1e-09),
            args.lr_mul, args.d_model, args.n_warmup_steps)

        train_loss, train_acc_list = train_model(trainloader, index_list[0][c], index_list[1][c], net, optimizer, criterion)
        train_acc = test_model(trainloader, index_list[0][c], index_list[1][c], net)
        test_acc = test_model(testloader, index_list[0][c], index_list[1][c], net)
        ID_scores = OOD_score_MSP(testloader, index_list[0][c], index_list[1][c], net)
        OOD_scores = OOD_score_MSP(OODloader, index_list[0][c], index_list[1][c], net)

        if np.min([np.min(ID_scores),np.min(OOD_scores)]) == np.max([np.max(ID_scores),np.max(OOD_scores)]):
            break

        detection_result = evaluation.detect_performance(ID_scores, OOD_scores, precision=args.precision)
        ece, ys = Calibrated(testloader, index_list[0][c], index_list[1][c], net)

        # print("Chain:", c + 1, ", fearures: ", sum(index_list[0][c]) + sum(index_list[1][c]), ", labels: ", y_dim,
        #     ", train acc: ", train_acc, "test acc: ", test_acc, ", OOD detection: ", detection_result[0], ", ECE: ", ece)
        # print("Confidence: ", np.mean(ID_scores), np.mean(OOD_scores))

        results.append(train_acc)
        results.append(test_acc)
        detection_result_s.append(detection_result)
        nets.append(net)
        losses.append(train_loss)
        train_acc_s.append(train_acc_list)
        ece_list.append(ece)
        ys_list.append(ys)

        c += 1
        # save_model(pth_name, net, c)
        if sum(index_list[0][-1]) >= 2 and c < args.order:
            update_list(index_list, c, torch.norm(net.get_weights(), dim = 1))
        else:
            break
    
    acc = test_model_ens(testloader, index_list, nets, c)
    ID_scores = OOD_score_MSP_ens(testloader, index_list, nets, c)
    OOD_scores = OOD_score_MSP_ens(OODloader, index_list, nets, c)
    detection_result = evaluation.detect_performance(ID_scores, OOD_scores, precision=args.precision)
    ece, ys = Calibrated_ens(testloader, index_list, nets, c)
    results.append(0)
    results.append(acc)
    detection_result_s.append(detection_result)
    ece_list.append(ece)
    ys_list.append(ys)

    # print("Ensemble:", ", test acc: ", acc, ", OOD detection: ", detection_result[0], ", ECE: ", ece)
    # print("Confidence: ", np.mean(ID_scores), np.mean(OOD_scores))

    write_index(index_list, index_name)
    write_res_table(c, acc, detection_result, ece, ys)
    write_res(results, detection_result_s, losses, train_acc_s, ece_list, ys_list, res_name)
    print("Accuracy:", acc, "AUROC:", detection_result[0])
    # print("------------------------------------------------------------------------------------------")

if __name__ == '__main__':
    main()
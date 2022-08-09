from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
import torch
import Ndata
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import argparse

parser = argparse.ArgumentParser(description='PyTorch')
parser.add_argument('-dataset', type=str, default='iris')
parser.add_argument('-rho', type=float, default=0.2)
parser.add_argument('-OOD', type=int, default=-1)
parser.add_argument('-normalization', action='store_true', default=False)
parser.add_argument('-seed', default=0, type=int, help='random seed')
args = parser.parse_args()

def main():

    if args.seed != 0:
        torch.manual_seed(args.seed)
        
    print("Processing: " + args.dataset + ", Seed: " + str(args.seed))
    X, y, X_dim, y_dim, OOD_index = Ndata.load_data(args.dataset)
    if args.OOD != -1:
        OOD_index = args.OOD

    ID_X, ID_y, test_OOD_X, test_OOD_y = Ndata.ID_OOD_split(X, y, ood_label = OOD_index)
    train_ID_X, test_ID_X, train_ID_y, test_ID_y = train_test_split(ID_X, ID_y, test_size=args.rho)

    if args.normalization:
        train_ID_X, test_ID_X, test_OOD_X = Ndata.feature_normalize(train_ID_X, test_ID_X, test_OOD_X)

    Ndata.write_data(train_ID_X, train_ID_y, args.dataset, OOD_index, "train_ID")
    Ndata.write_data(test_ID_X, test_ID_y, args.dataset, OOD_index,"test_ID")
    Ndata.write_data(test_OOD_X, test_OOD_y, args.dataset, OOD_index,"test_OOD")

if __name__ == '__main__':
    main()
from algorithm import *
from model import *
import torch
import numpy as np
import scipy.sparse as sp
from scipy.sparse import csr_matrix
import pickle
from netrd.distance import netsimile
import networkx as nx
import os.path as osp
from scipy.sparse import coo_matrix
from tqdm import tqdm
import random
import warnings
from torch.optim import Adam
from utils import *
import argparse
warnings.filterwarnings("ignore")


def fit_GAE(no_samples, GAE, epoch, train_loader, train_features, device, lr, level_eval, dataset_eval, model_eval):

    best_avg = 0
    best_std = 0
    S_hat_samples, S_prime_samples, p_samples = gen_test_set(device, load_adj(dataset_eval), 10,
                                                             [level_eval],
                                                             method=model_eval)
    S_eval = load_adj(dataset_eval)
    adj_S = coo_matrix(S_eval.numpy())
    adj_norm_S = preprocess_graph(adj_S)
    adj_norm_S = torch.sparse.FloatTensor(torch.LongTensor(adj_norm_S[0].T),
                                        torch.FloatTensor(adj_norm_S[1]),
                                        torch.Size(adj_norm_S[2])).to(device)
    S_feat = generate_features([S_eval])[0]

    S_hat_features = generate_features(S_hat_samples[str(level_eval)])
    optimizer = Adam(GAE.parameters(), lr=lr,weight_decay=5e-4)
    for step in range(epoch):
        loss = 0
        for dataset in train_loader.keys():
            S = train_loader[dataset][0]
            initial_features = train_features[dataset]
            for i in range(len(train_loader[dataset])):
                adj_tensor = train_loader[dataset][i]
                adj = coo_matrix(adj_tensor.numpy())
                adj_norm = preprocess_graph(adj)
                pos_weight = float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()
                norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)

                adj_label = coo_matrix(S.numpy())
                adj_label = sparse_to_tuple(adj_label)

                adj_norm = torch.sparse.FloatTensor(torch.LongTensor(adj_norm[0].T),
                                                    torch.FloatTensor(adj_norm[1]),
                                                    torch.Size(adj_norm[2])).to(device)
                adj_label = torch.sparse.FloatTensor(torch.LongTensor(adj_label[0].T),
                                                    torch.FloatTensor(adj_label[1]),
                                                    torch.Size(adj_label[2])).to(device)

                initial_feature = initial_features[i].to(device)

                weight_mask = adj_label.to_dense().view(-1) == 1
                weight_tensor = torch.ones(weight_mask.size(0))
                weight_tensor[weight_mask] = pos_weight
                weight_tensor = weight_tensor.to(device)
                z = GAE(initial_feature, adj_norm)
                A_pred = torch.sigmoid(torch.matmul(z,z.t()))
                loss += norm * F.binary_cross_entropy(A_pred.view(-1), adj_label.to_dense().view(-1),
                                                           weight=weight_tensor)
        optimizer.zero_grad()
        loss = loss / no_samples
        loss.backward()
        optimizer.step()
        print("Epoch:", '%04d' % (step + 1), "train_loss= {0:.5f}".format(loss.item()), end = " ")
        S_emb = GAE(S_feat.to(device), adj_norm_S).detach()
        avg, std = test_matching(GAE, S_hat_samples[str(level_eval)], p_samples[str(level_eval)], S_hat_features, S_emb, device,
                              metric="accuracy")
        if(avg > best_avg):
            best_avg = avg
            best_std = std
        print("Current best result:" +str(best_avg)[:6]+"+-"+str(best_std)[:5])




def load_adj(dataset):
    if (dataset == "celegans"):
        S = torch.load("data/celegans.pt")
    elif(dataset == "arenas"):
        S = torch.load("data/arenas.pt")
    elif (dataset == "douban"):
        S = torch.load("data/douban.pt")
    elif(dataset == "Online"):
        S = torch.load("data/online.pt")
    elif(dataset == "Offline"):
        S = torch.load("data/offline.pt")
    elif (dataset == "ACM"):
        S = torch.load("data/ACM.pt")
    elif (dataset == "DBLP"):
        S = torch.load("data/DBLP.pt")
    else:
        filepath = "data/" + dataset + ".npz"
        loader = load_npz(filepath)
        data = loader["adj_matrix"]
        samples = data.shape[0]
        features = data.shape[1]
        values = data.data
        coo_data = data.tocoo()
        indices = torch.LongTensor([coo_data.row, coo_data.col])
        S = torch.sparse.FloatTensor(indices, torch.from_numpy(values).float(), [samples, features]).to_dense()
        if (not torch.all(S.transpose(0, 1) == S)):
            S = torch.add(S, S.transpose(0, 1))
        S = S.int()
        ones = torch.ones_like(S)
        S = torch.where(S > 1, ones, S)
    return S


def main(args):
    device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
    train_set = ["celegans","arenas","douban","cora"]
    test_set = [args.dataset]
    probability_model = args.model
    training_perturbation_level = args.level
    no_training_samples_per_graph = 10
    NUM_HIDDEN_LAYERS = 8
    HIDDEN_DIM = 16
    output_feature_size = 8
    lr = 0.001
    epoch = 200
    print("Loading training datasets")

    train_loader = {}
    original_graph_loader = {}
    for dataset in [*set(train_set+test_set)]:
        original_graph_loader[dataset] = load_adj(dataset)
    print("Generating training perturbations")

    for dataset in train_set:
        train_loader[dataset] = generate_purturbations(device, original_graph_loader[dataset],
                                                        perturbation_level = training_perturbation_level,
                                                        no_samples=no_training_samples_per_graph,
                                                       method = probability_model)
    model = GAE(NUM_HIDDEN_LAYERS,
               7,
               HIDDEN_DIM,
               output_feature_size, activation=F.relu,
                use_input_augmentation=True,
                use_output_augmentation=False,
                encoder="GIN",variational=False).to(device)

    print("Generating training features")
    train_features = {}
    for dataset in train_loader.keys():
        train_features[dataset] = generate_features(train_loader[dataset])
    print("Fitting T-GAE")
    fit_GAE(len(train_set)*(no_training_samples_per_graph+1),model,epoch, train_loader, train_features, device, lr, args.level, args.dataset, args.model)


def parse_args():
    parser = argparse.ArgumentParser(description="Run T-GAE for graph matching task")
    parser.add_argument('--dataset', type=str, default="celegans", help='Choose from {celegans, arenas, douban, cora, dblp, coauthor_cs}')
    parser.add_argument('--model', type=str, default="uniform", help='Choose from {uniform, degree}')
    parser.add_argument('--level', type=int, default=0, help='Choose from {0,0.01,0.05}')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
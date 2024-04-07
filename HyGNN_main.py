import os.path as osp
import argparse
import os
import sys
import time
import torch
import numpy as np 
import torch.nn as nn
import torch.nn.functional as F
from tqdm import *
import torch.cuda as cuda

import HYGNN
from dataset import *
from GNN_model import *
from config import *

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default='DD_A_our_3', help="dataset")
parser.add_argument("--dim", type=int, default=96, help="input embedding dimension")
parser.add_argument("--num_layers", type=int, default=6, help="num layers")
parser.add_argument("--hidden", type=int, default=32, help="hidden dimension")
parser.add_argument("--classes", type=int, default=22, help="number of output classes")
parser.add_argument("--epochs", type=int, default=200, help="number of epoches")
parser.add_argument("--model", type=str, default='gcn', help='GNN model', choices=['gcn', 'gin'])
parser.add_argument("--single_kernel", action='store_true', help="whether to profile a single SAG kernel")
args = parser.parse_args()
print(args)

#########################################
## Load Graph from files.
#########################################
dataset = args.dataset
path = osp.join("../", dataset + ".txt")
dataset = HYGNN_dataset(path, args.dim, args.classes, load_from_txt=True)
num_nodes = dataset.num_nodes
num_edges = dataset.num_edges
column_index =  dataset.column_index 
row_pointers = dataset.row_pointers

#########################################
## Compute TC-GNN related graph MetaData.
#########################################
num_row_windows = (num_nodes + BLK_H - 1) // BLK_H
edgeToColumn = torch.zeros(num_edges, dtype=torch.int)
edgeToRow = torch.zeros(num_edges, dtype=torch.int)
blockPartition = torch.zeros(num_row_windows, dtype=torch.int)
hybrid_type = torch.zeros(num_row_windows, dtype=torch.int)
row_nzr = torch.zeros(num_row_windows + 1, dtype=torch.int)
col_nzr = torch.zeros(16 * num_row_windows, dtype=torch.int)
output = torch.zeros(num_nodes * args.hidden, dtype=torch.float).reshape(num_nodes, args.hidden)

# preprocessing for generating meta-information
start = time.perf_counter()
HYGNN.preprocess(column_index, row_pointers, num_nodes,  \
                BLK_H,	BLK_W, blockPartition, edgeToColumn, edgeToRow, hybrid_type, row_nzr, col_nzr)
build_neighbor_parts = time.perf_counter() - start
print("Prep. (ms):\t{:.3f}".format(build_neighbor_parts*1e3))

column_index = column_index.cuda()
row_pointers = row_pointers.cuda()
blockPartition = blockPartition.cuda()
edgeToColumn = edgeToColumn.cuda()
edgeToRow = edgeToRow.cuda()
hybrid_type = hybrid_type.cuda()
row_nzr = row_nzr.cuda()
col_nzr = col_nzr.cuda()
output = output.cuda()

#########################################
## Single Satter-And-Gather (SAG) Profiling.
#########################################
if args.single_kernel:
    SAG_obj = SAG(row_pointers, column_index,\
                    blockPartition, edgeToColumn, edgeToRow, hybrid_type, row_nzr, col_nzr)
    X = dataset.x
    # with torch.autograd.profiler.profile(use_cuda=True) as prof:
    SAG_obj.profile(X)
        # exit(0)
    # print(prof.key_averages().table(sort_by="cuda_time_total"))
    exit(0)
#########################################
## Build GCN and AGNN Model
#########################################
if args.model == "gcn":
    class Net(torch.nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.conv1 = GCNConv(dataset.num_features, args.hidden, 1)

            self.hidden_layers = nn.ModuleList()
            for _ in range(args.num_layers -  2):
                self.hidden_layers.append(GCNConv(args.hidden, args.hidden, 0))
            
            self.conv2 = GCNConv(args.hidden, dataset.num_classes, 2)
            self.relu = nn.ReLU()

        def forward(self):
            x = dataset.x
            x = self.relu(self.conv1(x, row_pointers, column_index, blockPartition, edgeToColumn, edgeToRow, hybrid_type, row_nzr, col_nzr, output))
            x = F.dropout(x, training=self.training)
            for Gconv in self.hidden_layers:
                x = Gconv(x, row_pointers, column_index, blockPartition, edgeToColumn, edgeToRow, hybrid_type, row_nzr, col_nzr, output)
                x = self.relu(x)
            x = self.conv2(x, row_pointers, column_index, blockPartition, edgeToColumn, edgeToRow, hybrid_type, row_nzr, col_nzr, output)
            return F.log_softmax(x, dim=1)

elif args.model == "gin":
    class Net(torch.nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.conv1 = GINConv(dataset.num_features, args.hidden, 1)

            self.hidden_layers = nn.ModuleList()
            for _ in range(args.num_layers -  2):
                self.hidden_layers.append(GINConv(args.hidden, args.hidden, 0))
            
            self.conv2 = GINConv(args.hidden, dataset.num_classes, 2)
            self.relu = nn.ReLU()

        def forward(self):
            x = dataset.x
            x = self.relu(self.conv1(x, row_pointers, column_index, blockPartition, edgeToColumn, edgeToRow, hybrid_type, row_nzr, col_nzr, output))
            x = F.dropout(x, training=self.training)
            for Gconv in self.hidden_layers:
                x = Gconv(x, row_pointers, column_index, blockPartition, edgeToColumn, edgeToRow, hybrid_type, row_nzr, col_nzr, output)
                x = self.relu(x)
            x = self.conv2(x, row_pointers, column_index, blockPartition, edgeToColumn, edgeToRow, hybrid_type, row_nzr, col_nzr, output)
            return F.log_softmax(x, dim=1)


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model, dataset = Net().to(device), dataset.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Training 
def train():
    # start = time.perf_counter()
    model.train()
    # dur = time.perf_counter() - start
    # print("=> Forward aggregation (ms): {:.3f}".format(dur*1e3))
    optimizer.zero_grad()
    # torch.cuda.synchronize()
    # start = time.perf_counter()
    loss = F.nll_loss(model()[:], dataset.y[:])
    # torch.cuda.synchronize()
    # dur = time.perf_counter() - start
    # print("=> Forward aggregation (ms): {:.3f}".format(dur*1e3))
    
    # torch.cuda.synchronize()
    # start = time.perf_counter()
    
    loss.backward()
    
    # torch.cuda.synchronize()
    # dur = time.perf_counter() - start
    # print("=> Forward aggregation (ms): {:.3f}".format(dur*1e3))
    
    optimizer.step()

    
    

if __name__ == "__main__":

    # s = torch.cuda.Stream()
    # s.wait_stream(torch.cuda.current_stream())
    # train()
    # torch.cuda.current_stream().wait_stream(s)
    

    # g = torch.cuda.CUDAGraph()
    # with torch.cuda.graph(g):
    #     train()

    # dry run.
    for epoch in range(1, 10):
      train()
      # g.replay()

    # with torch.autograd.profiler.profile(use_cuda=True) as prof:
    # torch.cuda.synchronize()
    # start_train = time.perf_counter()
    
    for _ in tqdm(range(1, args.epochs + 1)):
        train()
        # g.replay()


    # torch.cuda.synchronize()
    # train_time = time.perf_counter() - start_train

    # print("Train (ms):\t{:6.3f}".format(train_time))
    # print(prof.key_averages().table(sort_by="cuda_time_total"))
#!/usr/bin/env python3
import torch
import sys
import math
import time 

from tqdm.std import tqdm
import HYGNN

n_heads = 1
n_output = 8
rd = 100

def gen_test_tensor(X_prime):
    n_rows = X_prime.size(0)
    n_cols = X_prime.size(1)
    
    X_new = []
    for i in range(n_rows):
        tmp = [i] * n_cols
        X_new.append(tmp)

    X_new = torch.FloatTensor(X_new).cuda()
    return X_new


class TCGNNFunction_SAG(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X, row_pointers, column_index, \
                blockPartition, edgeToColumn, edgeToRow, hybrid_type, row_nzr, col_nzr):

        ctx.save_for_backward(row_pointers, column_index, \
                                blockPartition, edgeToColumn, edgeToRow, hybrid_type, row_nzr, col_nzr)
        real_embedding_dim = rd

        # torch.cuda.synchronize()
        # start = time.perf_counter()

        # Basic Scatter and Gather
        X_out = HYGNN.forward_fixed32(X, row_pointers, column_index, \
                                blockPartition, edgeToColumn, edgeToRow, hybrid_type, row_nzr, col_nzr)[0]

        # torch.cuda.synchronize()
        # dur = time.perf_counter() - start
        # print("=> Forward aggregation (ms): {:.3f}".format(dur*1e3))

        return X_out

    @staticmethod
    def backward(ctx, d_output):
        row_pointers, column_index, \
            blockPartition, edgeToColumn, edgeToRow, hybrid_type, row_nzr, col_nzr = ctx.saved_tensors
        real_embedding_dim = rd
        # SAG backward.
        d_input = HYGNN.forward(d_output, row_pointers, column_index, \
                                blockPartition, edgeToColumn, edgeToRow, hybrid_type, row_nzr, col_nzr, real_embedding_dim)[0]

        return d_input, None, None, None, None, None, None


class TCGNNFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X, weights, row_pointers, column_index, blockPartition, edgeToColumn, edgeToRow, hybrid_type, row_nzr, col_nzr):
        # X = torch.sparse.mm(edge_coo, X)
        ctx.save_for_backward(X, weights, row_pointers, column_index, blockPartition, edgeToColumn, edgeToRow, hybrid_type, row_nzr, col_nzr)
        real_embedding_dim = rd
        # GEMM node update
        # start = time.perf_counter()
        X_prime = torch.mm(X, weights)
        # print(X.shape, weights.shape, X_prime.shape)
        # torch.cuda.synchronize()
        # dur = time.perf_counter() - start
        # print("=> Forward aggregation (ms): {:.3f}".format(dur*1e3))
        # print()
        # X_prime_t = torch.ones_like(X_prime)
        # X_prime_t = gen_test_tensor(X_prime)
        # print("=========Before AggreAGNNion========")
        # print(X_prime_t)
        # sys.exit(0)
        # SpMM: Neighbor AggreAGNNion.
        X_prime = HYGNN.forward(X_prime, row_pointers, column_index, blockPartition, edgeToColumn, edgeToRow, hybrid_type, row_nzr, col_nzr)[0]
        # print("==========After Aggreation=========")
        # print(X_prime)
        # sys.exit(0)

        return X_prime

    @staticmethod
    def backward(ctx, d_output):
        X, weights, row_pointers, column_index, blockPartition, edgeToColumn, edgeToRow, hybrid_type, row_nzr, col_nzr = ctx.saved_tensors
        real_embedding_dim = rd
        # SPMM backward propaAGNNion.
        # start = time.perf_counter()
        d_input_prime = HYGNN.forward(d_output, row_pointers, column_index, blockPartition, edgeToColumn, edgeToRow, hybrid_type, row_nzr, col_nzr)[0]
        # torch.cuda.synchronize()
        # dur = time.perf_counter() - start
        # print("=> Forward aggregation (ms): {:.3f}".format(dur*1e3))
        # print()
        # GEMM backward propaAGNNion.
        d_input = torch.mm(d_input_prime, weights.transpose(0,1))
        d_weights = torch.mm(X.transpose(0,1), d_input_prime)
        return d_input, d_weights, None, None, None, None, None, None, None, None

class TCGNNFunctionFixed32(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X, weights, row_pointers, column_index, blockPartition, edgeToColumn, edgeToRow, hybrid_type, row_nzr, col_nzr):
        ctx.save_for_backward(X, weights, row_pointers, column_index, blockPartition, edgeToColumn, edgeToRow, hybrid_type, row_nzr, col_nzr)
        
        torch.cuda.synchronize()
        start = time.perf_counter()
        
        X_prime = torch.mm(X, weights)
        
        # torch.cuda.synchronize()
        # start = time.perf_counter()

        X_prime = HYGNN.forward_fixed32(X_prime, row_pointers, column_index, blockPartition, edgeToColumn, edgeToRow, hybrid_type, row_nzr, col_nzr)[0]
        # X_prime = HYGNN.forward_fixed64(X_prime, row_pointers, column_index, blockPartition, edgeToColumn, edgeToRow, hybrid_type, row_nzr, col_nzr)[0]
        
        torch.cuda.synchronize()
        dur = time.perf_counter() - start
        print("=> Forward aggregation (ms): {:.3f}".format(dur*1e3))
        
        return X_prime

    @staticmethod
    def backward(ctx, d_output):

        # torch.cuda.synchronize()
        # start = time.perf_counter()

        X, weights, row_pointers, column_index, blockPartition, edgeToColumn, edgeToRow, hybrid_type, row_nzr, col_nzr = ctx.saved_tensors
        
        torch.cuda.synchronize()
        start = time.perf_counter()

        tmp = HYGNN.forward_fixed32_fused(d_output, row_pointers, column_index, blockPartition, edgeToColumn, edgeToRow, hybrid_type, row_nzr, col_nzr, weights.transpose(0, 1))
        d_input, d_input_prime = tmp[0], tmp[1]
        # d_input_prime = HYGNN.forward_fixed32(d_output, row_pointers, column_index, blockPartition, edgeToColumn, edgeToRow, hybrid_type, row_nzr, col_nzr)[0]
        
        # d_input = torch.mm(d_input_prime, weights.transpose(0,1))
        d_weights = torch.mm(X.transpose(0,1), d_input_prime)
        
        torch.cuda.synchronize()
        dur = time.perf_counter() - start
        print("=> Backward aggregation (ms): {:.3f}".format(dur*1e3))

        return d_input, d_weights, None, None, None, None, None, None, None, None

class TCGNNFunctionFinal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X, weights, row_pointers, column_index, blockPartition, edgeToColumn, edgeToRow, hybrid_type, row_nzr, col_nzr, output):
        ctx.save_for_backward(X, weights, row_pointers, column_index, blockPartition, edgeToColumn, edgeToRow, hybrid_type, row_nzr, col_nzr, output)
        
        torch.cuda.synchronize()
        start = time.perf_counter()
        
        X_prime = torch.mm(X, weights)
        X_prime = HYGNN.forward(X_prime, row_pointers, column_index, blockPartition, edgeToColumn, edgeToRow, hybrid_type, row_nzr, col_nzr)[0] # 最后一层维度是class数量
        
        torch.cuda.synchronize()
        dur = time.perf_counter() - start
        print("=> Forward aggregation (ms): {:.3f}".format(dur*1e3))

        return X_prime

    @staticmethod
    def backward(ctx, d_output):

        # torch.cuda.synchronize()
        # start = time.perf_counter()


        X, weights, row_pointers, column_index, blockPartition, edgeToColumn, edgeToRow, hybrid_type, row_nzr, col_nzr, output = ctx.saved_tensors
        
        torch.cuda.synchronize()
        start = time.perf_counter()

        tmp = HYGNN.forward_final_fused(d_output, row_pointers, column_index, blockPartition, edgeToColumn, edgeToRow, hybrid_type, row_nzr, col_nzr, weights.transpose(0, 1), output) # 最后一层是维度是class数量: node * class * hidden
        # d_input_prime = HYGNN.forward(d_output, row_pointers, column_index, blockPartition, edgeToColumn, edgeToRow, hybrid_type, row_nzr, col_nzr)[0]
        # d_input = torch.mm(d_input_prime, weights.transpose(0,1))
        d_input, d_input_prime = tmp[0], tmp[1]
        d_weights = torch.mm(X.transpose(0,1), d_input_prime)
        # print(d_input[0])
        
        torch.cuda.synchronize()
        dur = time.perf_counter() - start
        print("=> Backward aggregation (ms): {:.3f}".format(dur*1e3))

        return d_input, d_weights, None, None, None, None, None, None, None, None, None

class TCGNNFunctionFirst(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X, weights, row_pointers, column_index, blockPartition, edgeToColumn, edgeToRow, hybrid_type, row_nzr, col_nzr):
        ctx.save_for_backward(X, weights, row_pointers, column_index, blockPartition, edgeToColumn, edgeToRow, hybrid_type, row_nzr, col_nzr)
        
        torch.cuda.synchronize()
        start = time.perf_counter()

        X_prime = torch.mm(X, weights)
        # print(weights.shape, X_prime.shape)
        X_prime = HYGNN.forward_fixed32(X_prime, row_pointers, column_index, blockPartition, edgeToColumn, edgeToRow, hybrid_type, row_nzr, col_nzr)[0]
        # X_prime = HYGNN.forward_fixed64(X_prime, row_pointers, column_index, blockPartition, edgeToColumn, edgeToRow, hybrid_type, row_nzr, col_nzr)[0]
        
        torch.cuda.synchronize()
        dur = time.perf_counter() - start
        print("=> Forward aggregation (ms): {:.3f}".format(dur*1e3))

        return X_prime

    @staticmethod
    def backward(ctx, d_output):

        # torch.cuda.synchronize()
        # start = time.perf_counter()

        X, weights, row_pointers, column_index, blockPartition, edgeToColumn, edgeToRow, hybrid_type, row_nzr, col_nzr = ctx.saved_tensors
        # start = time.perf_counter()
        # print(d_output.shape)

        torch.cuda.synchronize()
        start = time.perf_counter()

        d_input_prime = HYGNN.forward_fixed32(d_output, row_pointers, column_index, blockPartition, edgeToColumn, edgeToRow, hybrid_type, row_nzr, col_nzr)[0]
        # d_input_prime = HYGNN.forward_fixed64(d_output, row_pointers, column_index, blockPartition, edgeToColumn, edgeToRow, hybrid_type, row_nzr, col_nzr)[0]
        # dur = time.perf_counter() - start
        # print("=> Forward aggregation (ms): {:.3f}".format(dur*1e3))

        # print(d_output.shape, d_input_prime.shape)
        
        d_input = torch.mm(d_input_prime, weights.transpose(0,1))
        
        d_weights = torch.mm(X.transpose(0,1), d_input_prime)
        
        torch.cuda.synchronize()
        dur = time.perf_counter() - start
        print("=> Backward aggregation (ms): {:.3f}".format(dur*1e3))
        
        return d_input, d_weights, None, None, None, None, None, None, None, None

#################### GIN #######################

class TCGNNFunction_GINFixed32(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X, weights, row_pointers, column_index, blockPartition, edgeToColumn, edgeToRow, hybrid_type, row_nzr, col_nzr):

        torch.cuda.synchronize()
        start = time.perf_counter()

        # SpMM: Neighbor AggreAGNNion.
        tmp = HYGNN.forward_fixed32_fused(X, row_pointers, column_index, blockPartition, edgeToColumn, edgeToRow, hybrid_type, row_nzr, col_nzr, weights)
        X_prime_new, X_prime = tmp[0], tmp[1]
        ctx.save_for_backward(X_prime, weights, row_pointers, column_index, blockPartition, edgeToColumn, edgeToRow, hybrid_type, row_nzr, col_nzr)

        # GEMM node update
        # X_prime = torch.mm(X_prime_new, weights)
        torch.cuda.synchronize()
        dur = time.perf_counter() - start
        print("=> Forward aggregation (ms): {:.3f}".format(dur*1e3))

        return X_prime_new

    @staticmethod
    def backward(ctx, d_output):
        X_prime, weights, row_pointers, column_index, blockPartition, edgeToColumn, edgeToRow, hybrid_type, row_nzr, col_nzr = ctx.saved_tensors

        torch.cuda.synchronize()
        start = time.perf_counter()

        # GEMM backward propaAGNNion.
        d_X_prime = torch.mm(d_output, weights.transpose(0,1))
        d_weights = torch.mm(X_prime.transpose(0,1), d_output)  

        # SPMM backward propaAGNNion.
        d_input = HYGNN.forward_fixed32(d_X_prime, row_pointers, column_index, blockPartition, edgeToColumn, edgeToRow, hybrid_type, row_nzr, col_nzr)[0]

        torch.cuda.synchronize()
        dur = time.perf_counter() - start
        print("=> Backward aggregation (ms): {:.3f}".format(dur*1e3))

        return d_input, d_weights, None, None, None, None, None, None, None, None
        # return None, d_weights, None, None, None, None, None, None

class TCGNNFunction_GINFirst(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X, weights, row_pointers, column_index, blockPartition, edgeToColumn, edgeToRow, hybrid_type, row_nzr, col_nzr):

        torch.cuda.synchronize()
        start = time.perf_counter()

        # SpMM: Neighbor AggreAGNNion.
        X_prime = HYGNN.forward(X, row_pointers, column_index, blockPartition, edgeToColumn, edgeToRow, hybrid_type, row_nzr, col_nzr)[0]

        ctx.save_for_backward(X_prime, weights, row_pointers, column_index, blockPartition, edgeToColumn, edgeToRow, hybrid_type, row_nzr, col_nzr)

        # GEMM node update
        X_prime = torch.mm(X_prime, weights)
        
        torch.cuda.synchronize()
        dur = time.perf_counter() - start
        print("=> Forward aggregation (ms): {:.3f}".format(dur*1e3))

        return X_prime

    @staticmethod
    def backward(ctx, d_output):
        X_prime, weights, row_pointers, column_index, blockPartition, edgeToColumn, edgeToRow, hybrid_type, row_nzr, col_nzr = ctx.saved_tensors

        torch.cuda.synchronize()
        start = time.perf_counter()

        # GEMM backward propaAGNNion.
        d_X_prime = torch.mm(d_output, weights.transpose(0,1))
        d_weights = torch.mm(X_prime.transpose(0,1), d_output)  

        # SPMM backward propaAGNNion.
        d_input = HYGNN.forward(d_X_prime, row_pointers, column_index, blockPartition, edgeToColumn, edgeToRow, hybrid_type, row_nzr, col_nzr)[0]

        torch.cuda.synchronize()
        dur = time.perf_counter() - start
        print("=> Backward aggregation (ms): {:.3f}".format(dur*1e3))

        return d_input, d_weights, None, None, None, None, None, None, None, None
        # return None, d_weights, None, None, None, None, None, None

class TCGNNFunction_GINFinal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X, weights, row_pointers, column_index, blockPartition, edgeToColumn, edgeToRow, hybrid_type, row_nzr, col_nzr):
        
        torch.cuda.synchronize()
        start = time.perf_counter()

        # SpMM: Neighbor AggreAGNNion.
        # X_prime = HYGNN.forward_fixed32(X, row_pointers, column_index, blockPartition, edgeToColumn, edgeToRow, hybrid_type, row_nzr, col_nzr)[0]
        tmp = HYGNN.forward_GIN_final_fused(X, row_pointers, column_index, blockPartition, edgeToColumn, edgeToRow, hybrid_type, row_nzr, col_nzr, weights)
        X_prime_new, X_prime = tmp[0], tmp[1]
        # print(X_prime.shape, weights.shape, X_prime_new.shape)
        ctx.save_for_backward(X_prime, weights, row_pointers, column_index, blockPartition, edgeToColumn, edgeToRow, hybrid_type, row_nzr, col_nzr)
        # GEMM node update
        # X_prime = torch.mm(X_prime, weights)
        
        torch.cuda.synchronize()
        dur = time.perf_counter() - start
        print("=> Forward aggregation (ms): {:.3f}".format(dur*1e3))

        return X_prime_new

    @staticmethod
    def backward(ctx, d_output):
        X_prime, weights, row_pointers, column_index, blockPartition, edgeToColumn, edgeToRow, hybrid_type, row_nzr, col_nzr = ctx.saved_tensors

        torch.cuda.synchronize()
        start = time.perf_counter()

        # GEMM backward propaAGNNion.
        d_X_prime = torch.mm(d_output, weights.transpose(0,1))
        d_weights = torch.mm(X_prime.transpose(0,1), d_output)  

        # SPMM backward propaAGNNion.
        d_input = HYGNN.forward_fixed32(d_X_prime, row_pointers, column_index, blockPartition, edgeToColumn, edgeToRow, hybrid_type, row_nzr, col_nzr)[0]

        torch.cuda.synchronize()
        dur = time.perf_counter() - start
        print("=> Backward aggregation (ms): {:.3f}".format(dur*1e3))

        return d_input, d_weights, None, None, None, None, None, None, None, None
        # return None, d_weights, None, None, None, None, None, None

###################################
# Definition of each conv layers
###################################
class SAG(torch.nn.Module):
    def __init__(self, row_pointers, column_index, \
                    blockPartition, edgeToColumn, edgeToRow, hybrid_type, row_nzr, col_nzr):
        super(SAG, self).__init__()

        self.row_pointers = row_pointers
        self.column_index = column_index
        self.blockPartition = blockPartition
        self.edgeToColumn = edgeToColumn
        self.edgeToRow = edgeToRow
        self.hybrid_type = hybrid_type
        self.row_nzr = row_nzr
        self.col_nzr = col_nzr


    def profile(self, X, num_rounds=200):
        
        torch.cuda.synchronize()
        start = time.perf_counter()

        for _ in tqdm(range(num_rounds)):
            TCGNNFunction_SAG.apply(X, self.row_pointers, self.column_index, \
                                        self.blockPartition, self.edgeToColumn, self.edgeToRow, self.hybrid_type, self.row_nzr, self.col_nzr)
        torch.cuda.synchronize()
        dur = time.perf_counter() - start
        print("=> SAG profiling avg (ms): {:.3f}".format(dur*1e3/num_rounds))
        print()

class GCNConv(torch.nn.Module):
    def __init__(self, input_dim, output_dim, fixed = 0):
        super(GCNConv, self).__init__()
        self.weights = torch.nn.Parameter(torch.randn(input_dim, output_dim))
        # self.reset_parameters()
        self.fixed = fixed

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weights.size(1))
        self.weights.data.uniform_(-stdv, stdv)

    def forward(self, X, row_pointers, column_index, blockPartition, edgeToColumn, edgeToRow, hybrid_type, row_nzr, col_nzr, output):
        '''
        @param:
        X:  the input tensor of the graph node embedding, shape: [n_nodes, n_dim].
        A:  the CSR node pointer of the graph, shape: [node, 1].
        edges: the CSR edge list of the graph, shape: [edge, 1].
        partitioin: for the graph with the part-based optimziation.
        '''
        if self.fixed == 0: # 指中间层
            return TCGNNFunctionFixed32.apply(X, self.weights, row_pointers, column_index, blockPartition, edgeToColumn, edgeToRow, hybrid_type, row_nzr, col_nzr)
        elif self.fixed == 2: # 指最后一层
            return TCGNNFunctionFinal.apply(X, self.weights, row_pointers, column_index, blockPartition, edgeToColumn, edgeToRow, hybrid_type, row_nzr, col_nzr, output)
        else: # 指第一层
            return TCGNNFunctionFirst.apply(X, self.weights, row_pointers, column_index, blockPartition, edgeToColumn, edgeToRow, hybrid_type, row_nzr, col_nzr)


class GINConv(torch.nn.Module):
    def __init__(self, input_dim, output_dim, fixed = 0):
        super(GINConv, self).__init__()
        self.weights = torch.nn.Parameter(torch.randn(input_dim, output_dim))
        # self.reset_parameters()
        self.fixed = fixed

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weights.size(1))
        self.weights.data.uniform_(-stdv, stdv)

    def forward(self, X, row_pointers, column_index, blockPartition, edgeToColumn, edgeToRow, hybrid_type, row_nzr, col_nzr, output):
        '''
        @param:
        X:  the input tensor of the graph node embedding, shape: [n_nodes, n_dim].
        A:  the CSR node pointer of the graph, shape: [node, 1].
        edges: the CSR edge list of the graph, shape: [edge, 1].
        partitioin: for the graph with the part-based optimziation.
        '''
        if self.fixed == 0: # 指中间层
            return TCGNNFunction_GINFixed32.apply(X, self.weights, row_pointers, column_index, blockPartition, edgeToColumn, edgeToRow, hybrid_type, row_nzr, col_nzr)
        elif self.fixed == 2: # 指最后一层
            return TCGNNFunction_GINFinal.apply(X, self.weights, row_pointers, column_index, blockPartition, edgeToColumn, edgeToRow, hybrid_type, row_nzr, col_nzr)
        else: # 指第一层
            return TCGNNFunction_GINFirst.apply(X, self.weights, row_pointers, column_index, blockPartition, edgeToColumn, edgeToRow, hybrid_type, row_nzr, col_nzr)
import torch
import sys
import math
import time 

from tqdm.std import tqdm
import HCSPMM

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


class HCSPMMFunction_SAG(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X, row_pointers, column_index, \
                blockPartition, edgeToColumn, edgeToRow, hybrid_type, row_nzr, col_nzr):

        ctx.save_for_backward(row_pointers, column_index, \
                                blockPartition, edgeToColumn, edgeToRow, hybrid_type, row_nzr, col_nzr)
        real_embedding_dim = rd

        # torch.cuda.synchronize()
        # start = time.perf_counter()

        # Basic Scatter and Gather
        X_out = HCSPMM.forward_fixed32(X, row_pointers, column_index, \
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
        d_input = HCSPMM.forward(d_output, row_pointers, column_index, \
                                blockPartition, edgeToColumn, edgeToRow, hybrid_type, row_nzr, col_nzr, real_embedding_dim)[0]

        return d_input, None, None, None, None, None, None


class HCSPMMFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X, weights, row_pointers, column_index, blockPartition, edgeToColumn, edgeToRow, hybrid_type, row_nzr, col_nzr):
        # X = torch.sparse.mm(edge_coo, X)
        ctx.save_for_backward(X, weights, row_pointers, column_index, blockPartition, edgeToColumn, edgeToRow, hybrid_type, row_nzr, col_nzr)
        real_embedding_dim = rd

        X_prime = torch.mm(X, weights)
        
        X_prime = HCSPMM.forward(X_prime, row_pointers, column_index, blockPartition, edgeToColumn, edgeToRow, hybrid_type, row_nzr, col_nzr)[0]

        return X_prime

    @staticmethod
    def backward(ctx, d_output):
        X, weights, row_pointers, column_index, blockPartition, edgeToColumn, edgeToRow, hybrid_type, row_nzr, col_nzr = ctx.saved_tensors
        real_embedding_dim = rd
        d_input_prime = HCSPMM.forward(d_output, row_pointers, column_index, blockPartition, edgeToColumn, edgeToRow, hybrid_type, row_nzr, col_nzr)[0]
        d_input = torch.mm(d_input_prime, weights.transpose(0,1))
        d_weights = torch.mm(X.transpose(0,1), d_input_prime)
        return d_input, d_weights, None, None, None, None, None, None, None, None

class HCSPMMFunctionFixed32(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X, weights, row_pointers, column_index, blockPartition, edgeToColumn, edgeToRow, hybrid_type, row_nzr, col_nzr):
        ctx.save_for_backward(X, weights, row_pointers, column_index, blockPartition, edgeToColumn, edgeToRow, hybrid_type, row_nzr, col_nzr)
        
        X_prime = torch.mm(X, weights)

        X_prime = HCSPMM.forward_fixed32(X_prime, row_pointers, column_index, blockPartition, edgeToColumn, edgeToRow, hybrid_type, row_nzr, col_nzr)[0]
        
        return X_prime

    @staticmethod
    def backward(ctx, d_output):

        X, weights, row_pointers, column_index, blockPartition, edgeToColumn, edgeToRow, hybrid_type, row_nzr, col_nzr = ctx.saved_tensors

        tmp = HCSPMM.forward_fixed32_fused(d_output, row_pointers, column_index, blockPartition, edgeToColumn, edgeToRow, hybrid_type, row_nzr, col_nzr, weights.transpose(0, 1))
        d_input, d_input_prime = tmp[0], tmp[1]
        
        d_weights = torch.mm(X.transpose(0,1), d_input_prime)

        return d_input, d_weights, None, None, None, None, None, None, None, None

class HCSPMMFunctionFinal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X, weights, row_pointers, column_index, blockPartition, edgeToColumn, edgeToRow, hybrid_type, row_nzr, col_nzr, output):
        ctx.save_for_backward(X, weights, row_pointers, column_index, blockPartition, edgeToColumn, edgeToRow, hybrid_type, row_nzr, col_nzr, output)
        
        X_prime = torch.mm(X, weights)
        X_prime = HCSPMM.forward(X_prime, row_pointers, column_index, blockPartition, edgeToColumn, edgeToRow, hybrid_type, row_nzr, col_nzr)[0]

        return X_prime

    @staticmethod
    def backward(ctx, d_output):

        X, weights, row_pointers, column_index, blockPartition, edgeToColumn, edgeToRow, hybrid_type, row_nzr, col_nzr, output = ctx.saved_tensors
        
        tmp = HCSPMM.forward_final_fused(d_output, row_pointers, column_index, blockPartition, edgeToColumn, edgeToRow, hybrid_type, row_nzr, col_nzr, weights.transpose(0, 1), output)
        # d_input_prime = HCSPMM.forward(d_output, row_pointers, column_index, blockPartition, edgeToColumn, edgeToRow, hybrid_type, row_nzr, col_nzr)[0]
        # d_input = torch.mm(d_input_prime, weights.transpose(0,1))
        d_input, d_input_prime = tmp[0], tmp[1]
        d_weights = torch.mm(X.transpose(0,1), d_input_prime)
        # print(d_input[0])

        return d_input, d_weights, None, None, None, None, None, None, None, None, None

class HCSPMMFunctionFirst(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X, weights, row_pointers, column_index, blockPartition, edgeToColumn, edgeToRow, hybrid_type, row_nzr, col_nzr):
        ctx.save_for_backward(X, weights, row_pointers, column_index, blockPartition, edgeToColumn, edgeToRow, hybrid_type, row_nzr, col_nzr)

        X_prime = torch.mm(X, weights)
        # print(weights.shape, X_prime.shape)
        X_prime = HCSPMM.forward_fixed32(X_prime, row_pointers, column_index, blockPartition, edgeToColumn, edgeToRow, hybrid_type, row_nzr, col_nzr)[0]
        # X_prime = HCSPMM.forward_fixed64(X_prime, row_pointers, column_index, blockPartition, edgeToColumn, edgeToRow, hybrid_type, row_nzr, col_nzr)[0]

        return X_prime

    @staticmethod
    def backward(ctx, d_output):

        # torch.cuda.synchronize()
        # start = time.perf_counter()

        X, weights, row_pointers, column_index, blockPartition, edgeToColumn, edgeToRow, hybrid_type, row_nzr, col_nzr = ctx.saved_tensors
        # start = time.perf_counter()
        # print(d_output.shape)

        d_input_prime = HCSPMM.forward_fixed32(d_output, row_pointers, column_index, blockPartition, edgeToColumn, edgeToRow, hybrid_type, row_nzr, col_nzr)[0]
        # d_input_prime = HCSPMM.forward_fixed64(d_output, row_pointers, column_index, blockPartition, edgeToColumn, edgeToRow, hybrid_type, row_nzr, col_nzr)[0]
        # dur = time.perf_counter() - start
        # print("=> Forward aggregation (ms): {:.3f}".format(dur*1e3))

        # print(d_output.shape, d_input_prime.shape)
        
        d_input = torch.mm(d_input_prime, weights.transpose(0,1))
        
        d_weights = torch.mm(X.transpose(0,1), d_input_prime)
        
        return d_input, d_weights, None, None, None, None, None, None, None, None

#################### GIN #######################

class HCSPMMFunction_GINFixed32(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X, weights, row_pointers, column_index, blockPartition, edgeToColumn, edgeToRow, hybrid_type, row_nzr, col_nzr):
        tmp = HCSPMM.forward_fixed32_fused(X, row_pointers, column_index, blockPartition, edgeToColumn, edgeToRow, hybrid_type, row_nzr, col_nzr, weights)
        X_prime_new, X_prime = tmp[0], tmp[1]
        ctx.save_for_backward(X_prime, weights, row_pointers, column_index, blockPartition, edgeToColumn, edgeToRow, hybrid_type, row_nzr, col_nzr)
        return X_prime_new

    @staticmethod
    def backward(ctx, d_output):
        X_prime, weights, row_pointers, column_index, blockPartition, edgeToColumn, edgeToRow, hybrid_type, row_nzr, col_nzr = ctx.saved_tensors

        d_X_prime = torch.mm(d_output, weights.transpose(0,1))
        d_weights = torch.mm(X_prime.transpose(0,1), d_output)  

        d_input = HCSPMM.forward_fixed32(d_X_prime, row_pointers, column_index, blockPartition, edgeToColumn, edgeToRow, hybrid_type, row_nzr, col_nzr)[0]

        return d_input, d_weights, None, None, None, None, None, None, None, None
        # return None, d_weights, None, None, None, None, None, None

class HCSPMMFunction_GINFirst(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X, weights, row_pointers, column_index, blockPartition, edgeToColumn, edgeToRow, hybrid_type, row_nzr, col_nzr):

        X_prime = HCSPMM.forward(X, row_pointers, column_index, blockPartition, edgeToColumn, edgeToRow, hybrid_type, row_nzr, col_nzr)[0]

        ctx.save_for_backward(X_prime, weights, row_pointers, column_index, blockPartition, edgeToColumn, edgeToRow, hybrid_type, row_nzr, col_nzr)

        X_prime = torch.mm(X_prime, weights)

        return X_prime

    @staticmethod
    def backward(ctx, d_output):
        X_prime, weights, row_pointers, column_index, blockPartition, edgeToColumn, edgeToRow, hybrid_type, row_nzr, col_nzr = ctx.saved_tensors

        d_X_prime = torch.mm(d_output, weights.transpose(0,1))
        d_weights = torch.mm(X_prime.transpose(0,1), d_output)  

        d_input = HCSPMM.forward(d_X_prime, row_pointers, column_index, blockPartition, edgeToColumn, edgeToRow, hybrid_type, row_nzr, col_nzr)[0]

        return d_input, d_weights, None, None, None, None, None, None, None, None
        # return None, d_weights, None, None, None, None, None, None

class HCSPMMFunction_GINFinal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X, weights, row_pointers, column_index, blockPartition, edgeToColumn, edgeToRow, hybrid_type, row_nzr, col_nzr):

        # X_prime = HCSPMM.forward_fixed32(X, row_pointers, column_index, blockPartition, edgeToColumn, edgeToRow, hybrid_type, row_nzr, col_nzr)[0]
        tmp = HCSPMM.forward_GIN_final_fused(X, row_pointers, column_index, blockPartition, edgeToColumn, edgeToRow, hybrid_type, row_nzr, col_nzr, weights)
        X_prime_new, X_prime = tmp[0], tmp[1]
        # print(X_prime.shape, weights.shape, X_prime_new.shape)
        ctx.save_for_backward(X_prime, weights, row_pointers, column_index, blockPartition, edgeToColumn, edgeToRow, hybrid_type, row_nzr, col_nzr)
        # X_prime = torch.mm(X_prime, weights)

        return X_prime_new

    @staticmethod
    def backward(ctx, d_output):
        X_prime, weights, row_pointers, column_index, blockPartition, edgeToColumn, edgeToRow, hybrid_type, row_nzr, col_nzr = ctx.saved_tensors

        d_X_prime = torch.mm(d_output, weights.transpose(0,1))
        d_weights = torch.mm(X_prime.transpose(0,1), d_output)  

        d_input = HCSPMM.forward_fixed32(d_X_prime, row_pointers, column_index, blockPartition, edgeToColumn, edgeToRow, hybrid_type, row_nzr, col_nzr)[0]

        return d_input, d_weights, None, None, None, None, None, None, None, None
        # return None, d_weights, None, None, None, None, None, None


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
            HCSPMMFunction_SAG.apply(X, self.row_pointers, self.column_index, \
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

        if self.fixed == 0:
            return HCSPMMFunctionFixed32.apply(X, self.weights, row_pointers, column_index, blockPartition, edgeToColumn, edgeToRow, hybrid_type, row_nzr, col_nzr)
        elif self.fixed == 2: 
            return HCSPMMFunctionFinal.apply(X, self.weights, row_pointers, column_index, blockPartition, edgeToColumn, edgeToRow, hybrid_type, row_nzr, col_nzr, output)
        else:
            return HCSPMMFunctionFirst.apply(X, self.weights, row_pointers, column_index, blockPartition, edgeToColumn, edgeToRow, hybrid_type, row_nzr, col_nzr)


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
        if self.fixed == 0: 
            return HCSPMMFunction_GINFixed32.apply(X, self.weights, row_pointers, column_index, blockPartition, edgeToColumn, edgeToRow, hybrid_type, row_nzr, col_nzr)
        elif self.fixed == 2: 
            return HCSPMMFunction_GINFinal.apply(X, self.weights, row_pointers, column_index, blockPartition, edgeToColumn, edgeToRow, hybrid_type, row_nzr, col_nzr)
        else: 
            return HCSPMMFunction_GINFirst.apply(X, self.weights, row_pointers, column_index, blockPartition, edgeToColumn, edgeToRow, hybrid_type, row_nzr, col_nzr)

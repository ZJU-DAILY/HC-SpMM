#include <torch/extension.h>
#include <vector>
#include <string.h>
#include <cstdlib>
#include <map>

#include <thrust/sort.h>
#include <thrust/execution_policy.h>


#define min(x, y) (((x) < (y))? (x) : (y))

std::vector<torch::Tensor> spmm_forward_plus(
    torch::Tensor input,
    torch::Tensor nodePointer,
    torch::Tensor edgeList,
    torch::Tensor blockPartition, 
    torch::Tensor edgeToColumn,
    torch::Tensor edgeToRow,
    int num_nodes,
    int num_edges,
    int embedding_dim,
    torch::Tensor hybrid_type,
    torch::Tensor row_nzr,
    torch::Tensor col_nzr
  );

  std::vector<torch::Tensor> spmm_forward_plus_more(
    torch::Tensor input,
    torch::Tensor nodePointer,
    torch::Tensor edgeList,
    torch::Tensor blockPartition, 
    torch::Tensor edgeToColumn,
    torch::Tensor edgeToRow,
    int num_nodes,
    int num_edges,
    int embedding_dim,
    torch::Tensor hybrid_type,
    torch::Tensor row_nzr,
    torch::Tensor col_nzr
  );

std::vector<torch::Tensor> spmm_forward_plus_fixed32(
    torch::Tensor input,
    torch::Tensor nodePointer,
    torch::Tensor edgeList,
    torch::Tensor blockPartition, 
    torch::Tensor edgeToColumn,
    torch::Tensor edgeToRow,
    int num_nodes,
    int num_edges,
    int embedding_dim,
    torch::Tensor hybrid_type,
    torch::Tensor row_nzr,
    torch::Tensor col_nzr
  );

std::vector<torch::Tensor> spmm_forward_plus_fixed64(
    torch::Tensor input,
    torch::Tensor nodePointer,
    torch::Tensor edgeList,
    torch::Tensor blockPartition, 
    torch::Tensor edgeToColumn,
    torch::Tensor edgeToRow,
    int num_nodes,
    int num_edges,
    int embedding_dim,
    torch::Tensor hybrid_type,
    torch::Tensor row_nzr,
    torch::Tensor col_nzr
  );

std::vector<torch::Tensor> spmm_forward_plus_fixed32_fused(
    torch::Tensor input,
    torch::Tensor nodePointer,
    torch::Tensor edgeList,
    torch::Tensor blockPartition,
    torch::Tensor edgeToColumn,
    torch::Tensor edgeToRow,
    int num_nodes,
    int num_edges,
    int embedding_dim,
    int hidden_dim,
    torch::Tensor hybrid_type,
    torch::Tensor row_nzr,
    torch::Tensor col_nzr,
    torch::Tensor weights
);

std::vector<torch::Tensor> spmm_forward_plus_fixed64_fused(
    torch::Tensor input,
    torch::Tensor nodePointer,
    torch::Tensor edgeList,
    torch::Tensor blockPartition,
    torch::Tensor edgeToColumn,
    torch::Tensor edgeToRow,
    int num_nodes,
    int num_edges,
    int embedding_dim,
    int hidden_dim,
    torch::Tensor hybrid_type,
    torch::Tensor row_nzr,
    torch::Tensor col_nzr,
    torch::Tensor weights
);

// std::vector<torch::Tensor> spmm_forward_plus_fixed32_fused_all(
//         torch::Tensor input,
//         torch::Tensor nodePointer,
//         torch::Tensor edgeList,
//         torch::Tensor blockPartition,
//         torch::Tensor edgeToColumn,
//         torch::Tensor edgeToRow,
//         int num_nodes,
//         int num_edges,
//         int embedding_dim,
//         int hidden_dim,
//         torch::Tensor hybrid_type,
//         torch::Tensor row_nzr,
//         torch::Tensor col_nzr,
//         torch::Tensor weights,
//         torch::Tensor X,
//         torch::Tensor output2
// );

std::vector<torch::Tensor> spmm_forward_plus_final_fused(
        torch::Tensor input,
        torch::Tensor nodePointer,
        torch::Tensor edgeList,
        torch::Tensor blockPartition,
        torch::Tensor edgeToColumn,
        torch::Tensor edgeToRow,
        int num_nodes,
        int num_edges,
        int embedding_dim,
        int hidden_dim,
        torch::Tensor hybrid_type,
        torch::Tensor row_nzr,
        torch::Tensor col_nzr,
        torch::Tensor weights,
        torch::Tensor output
);

std::vector<torch::Tensor> spmm_forward_plus_final_fused_64(
        torch::Tensor input,
        torch::Tensor nodePointer,
        torch::Tensor edgeList,
        torch::Tensor blockPartition,
        torch::Tensor edgeToColumn,
        torch::Tensor edgeToRow,
        int num_nodes,
        int num_edges,
        int embedding_dim,
        int hidden_dim,
        torch::Tensor hybrid_type,
        torch::Tensor row_nzr,
        torch::Tensor col_nzr,
        torch::Tensor weights,
        torch::Tensor output
);

std::vector<torch::Tensor> spmm_forward_plus_GIN_final_fused(
        torch::Tensor input,
        torch::Tensor nodePointer,
        torch::Tensor edgeList,
        torch::Tensor blockPartition,
        torch::Tensor edgeToColumn,
        torch::Tensor edgeToRow,
        int num_nodes,
        int num_edges,
        int embedding_dim,
        int hidden_dim,
        torch::Tensor hybrid_type,
        torch::Tensor row_nzr,
        torch::Tensor col_nzr,
        torch::Tensor weights
);

#define CHECK_CUDA(x) TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

//////////////////////////////////////////
//
// SPMM Foward Pass (GCN, GraphSAGE)
//
////////////////////////////////////////////
std::vector<torch::Tensor> spmm_forward(
    torch::Tensor input,
    torch::Tensor nodePointer,
    torch::Tensor edgeList,
    torch::Tensor blockPartition, 
    torch::Tensor edgeToColumn,
    torch::Tensor edgeToRow,
    torch::Tensor hybrid_type,
    torch::Tensor row_nzr,
    torch::Tensor col_nzr
) {
  CHECK_INPUT(input);
  CHECK_INPUT(nodePointer);
  CHECK_INPUT(edgeList);
  CHECK_INPUT(blockPartition);
  CHECK_INPUT(edgeToColumn);
  CHECK_INPUT(edgeToRow);

  int num_nodes = nodePointer.size(0) - 1;
  int num_edges = edgeList.size(0);
  int embedding_dim = input.size(1);

  return spmm_forward_plus(input, nodePointer, edgeList, 
                            blockPartition, edgeToColumn, edgeToRow, 
                            num_nodes, num_edges, embedding_dim,
                            hybrid_type, row_nzr, col_nzr
                            );
}

std::vector<torch::Tensor> spmm_forward_more(
    torch::Tensor input,
    torch::Tensor nodePointer,
    torch::Tensor edgeList,
    torch::Tensor blockPartition, 
    torch::Tensor edgeToColumn,
    torch::Tensor edgeToRow,
    torch::Tensor hybrid_type,
    torch::Tensor row_nzr,
    torch::Tensor col_nzr
) {
  CHECK_INPUT(input);
  CHECK_INPUT(nodePointer);
  CHECK_INPUT(edgeList);
  CHECK_INPUT(blockPartition);
  CHECK_INPUT(edgeToColumn);
  CHECK_INPUT(edgeToRow);

  int num_nodes = nodePointer.size(0) - 1;
  int num_edges = edgeList.size(0);
  int embedding_dim = input.size(1);

  return spmm_forward_plus_more(input, nodePointer, edgeList, 
                            blockPartition, edgeToColumn, edgeToRow, 
                            num_nodes, num_edges, embedding_dim,
                            hybrid_type, row_nzr, col_nzr
                            );
}

std::vector<torch::Tensor> spmm_forward_fixed32(
    torch::Tensor input,
    torch::Tensor nodePointer,
    torch::Tensor edgeList,
    torch::Tensor blockPartition, 
    torch::Tensor edgeToColumn,
    torch::Tensor edgeToRow,
    torch::Tensor hybrid_type,
    torch::Tensor row_nzr,
    torch::Tensor col_nzr
) {
  CHECK_INPUT(input);
  CHECK_INPUT(nodePointer);
  CHECK_INPUT(edgeList);
  CHECK_INPUT(blockPartition);
  CHECK_INPUT(edgeToColumn);
  CHECK_INPUT(edgeToRow);

  int num_nodes = nodePointer.size(0) - 1;
  int num_edges = edgeList.size(0);
  int embedding_dim = input.size(1);

  return spmm_forward_plus_fixed32(input, nodePointer, edgeList, 
                            blockPartition, edgeToColumn, edgeToRow, 
                            num_nodes, num_edges, embedding_dim,
                            hybrid_type, row_nzr, col_nzr
                            );
}

std::vector<torch::Tensor> spmm_forward_fixed64(
    torch::Tensor input,
    torch::Tensor nodePointer,
    torch::Tensor edgeList,
    torch::Tensor blockPartition, 
    torch::Tensor edgeToColumn,
    torch::Tensor edgeToRow,
    torch::Tensor hybrid_type,
    torch::Tensor row_nzr,
    torch::Tensor col_nzr
) {
  CHECK_INPUT(input);
  CHECK_INPUT(nodePointer);
  CHECK_INPUT(edgeList);
  CHECK_INPUT(blockPartition);
  CHECK_INPUT(edgeToColumn);
  CHECK_INPUT(edgeToRow);

  int num_nodes = nodePointer.size(0) - 1;
  int num_edges = edgeList.size(0);
  int embedding_dim = input.size(1);

  return spmm_forward_plus_fixed64(input, nodePointer, edgeList, 
                            blockPartition, edgeToColumn, edgeToRow, 
                            num_nodes, num_edges, embedding_dim,
                            hybrid_type, row_nzr, col_nzr
                            );
}

std::vector<torch::Tensor> spmm_forward_fixed32_fused(
    torch::Tensor input,
    torch::Tensor nodePointer,
    torch::Tensor edgeList,
    torch::Tensor blockPartition, 
    torch::Tensor edgeToColumn,
    torch::Tensor edgeToRow,
    torch::Tensor hybrid_type,
    torch::Tensor row_nzr,
    torch::Tensor col_nzr,
    torch::Tensor weights
) {
  CHECK_INPUT(input);
  CHECK_INPUT(nodePointer);
  CHECK_INPUT(edgeList);
  CHECK_INPUT(blockPartition);
  CHECK_INPUT(edgeToColumn);
  CHECK_INPUT(edgeToRow);

  int num_nodes = nodePointer.size(0) - 1;
  int num_edges = edgeList.size(0);
  int embedding_dim = input.size(1);
  int hidden_dim = weights.size(1);

  return spmm_forward_plus_fixed32_fused(input, nodePointer, edgeList, 
                            blockPartition, edgeToColumn, edgeToRow, 
                            num_nodes, num_edges, embedding_dim, hidden_dim,
                            hybrid_type, row_nzr, col_nzr, weights
                            );
}

std::vector<torch::Tensor> spmm_forward_fixed64_fused(
    torch::Tensor input,
    torch::Tensor nodePointer,
    torch::Tensor edgeList,
    torch::Tensor blockPartition, 
    torch::Tensor edgeToColumn,
    torch::Tensor edgeToRow,
    torch::Tensor hybrid_type,
    torch::Tensor row_nzr,
    torch::Tensor col_nzr,
    torch::Tensor weights
) {
  CHECK_INPUT(input);
  CHECK_INPUT(nodePointer);
  CHECK_INPUT(edgeList);
  CHECK_INPUT(blockPartition);
  CHECK_INPUT(edgeToColumn);
  CHECK_INPUT(edgeToRow);

  int num_nodes = nodePointer.size(0) - 1;
  int num_edges = edgeList.size(0);
  int embedding_dim = input.size(1);
  int hidden_dim = weights.size(1);

  return spmm_forward_plus_fixed64_fused(input, nodePointer, edgeList, 
                            blockPartition, edgeToColumn, edgeToRow, 
                            num_nodes, num_edges, embedding_dim, hidden_dim,
                            hybrid_type, row_nzr, col_nzr, weights
                            );
}

// std::vector<torch::Tensor> spmm_forward_fixed32_fused_all(
//     torch::Tensor input,
//     torch::Tensor nodePointer,
//     torch::Tensor edgeList,
//     torch::Tensor blockPartition, 
//     torch::Tensor edgeToColumn,
//     torch::Tensor edgeToRow,
//     torch::Tensor hybrid_type,
//     torch::Tensor row_nzr,
//     torch::Tensor col_nzr,
//     torch::Tensor weights,
//     torch::Tensor X,
//     torch::Tensor output2
// ) {
//   CHECK_INPUT(input);
//   CHECK_INPUT(nodePointer);
//   CHECK_INPUT(edgeList);
//   CHECK_INPUT(blockPartition);
//   CHECK_INPUT(edgeToColumn);
//   CHECK_INPUT(edgeToRow);

//   int num_nodes = nodePointer.size(0) - 1;
//   int num_edges = edgeList.size(0);
//   int embedding_dim = input.size(1);
//   int hidden_dim = weights.size(1);

//   return spmm_forward_plus_fixed32_fused_all(input, nodePointer, edgeList, 
//                             blockPartition, edgeToColumn, edgeToRow, 
//                             num_nodes, num_edges, embedding_dim, hidden_dim,
//                             hybrid_type, row_nzr, col_nzr, weights, X, output2
//                             );
// }

std::vector<torch::Tensor> spmm_forward_final_fused(
    torch::Tensor input,
    torch::Tensor nodePointer,
    torch::Tensor edgeList,
    torch::Tensor blockPartition, 
    torch::Tensor edgeToColumn,
    torch::Tensor edgeToRow,
    torch::Tensor hybrid_type,
    torch::Tensor row_nzr,
    torch::Tensor col_nzr,
    torch::Tensor weights,
    torch::Tensor output
) {
  CHECK_INPUT(input);
  CHECK_INPUT(nodePointer);
  CHECK_INPUT(edgeList);
  CHECK_INPUT(blockPartition);
  CHECK_INPUT(edgeToColumn);
  CHECK_INPUT(edgeToRow);

  int num_nodes = nodePointer.size(0) - 1;
  int num_edges = edgeList.size(0);
  int embedding_dim = input.size(1);
  int hidden_dim = weights.size(1);

  return spmm_forward_plus_final_fused(input, nodePointer, edgeList, 
                            blockPartition, edgeToColumn, edgeToRow, 
                            num_nodes, num_edges, embedding_dim, hidden_dim,
                            hybrid_type, row_nzr, col_nzr, weights, output
                            );
}

std::vector<torch::Tensor> spmm_forward_final_fused_64(
    torch::Tensor input,
    torch::Tensor nodePointer,
    torch::Tensor edgeList,
    torch::Tensor blockPartition, 
    torch::Tensor edgeToColumn,
    torch::Tensor edgeToRow,
    torch::Tensor hybrid_type,
    torch::Tensor row_nzr,
    torch::Tensor col_nzr,
    torch::Tensor weights,
    torch::Tensor output
) {
  CHECK_INPUT(input);
  CHECK_INPUT(nodePointer);
  CHECK_INPUT(edgeList);
  CHECK_INPUT(blockPartition);
  CHECK_INPUT(edgeToColumn);
  CHECK_INPUT(edgeToRow);

  int num_nodes = nodePointer.size(0) - 1;
  int num_edges = edgeList.size(0);
  int embedding_dim = input.size(1);
  int hidden_dim = weights.size(1);

  return spmm_forward_plus_final_fused_64(input, nodePointer, edgeList, 
                            blockPartition, edgeToColumn, edgeToRow, 
                            num_nodes, num_edges, embedding_dim, hidden_dim,
                            hybrid_type, row_nzr, col_nzr, weights, output
                            );
}

std::vector<torch::Tensor> spmm_forward_GIN_final_fused(
    torch::Tensor input,
    torch::Tensor nodePointer,
    torch::Tensor edgeList,
    torch::Tensor blockPartition, 
    torch::Tensor edgeToColumn,
    torch::Tensor edgeToRow,
    torch::Tensor hybrid_type,
    torch::Tensor row_nzr,
    torch::Tensor col_nzr,
    torch::Tensor weights
) {
  CHECK_INPUT(input);
  CHECK_INPUT(nodePointer);
  CHECK_INPUT(edgeList);
  CHECK_INPUT(blockPartition);
  CHECK_INPUT(edgeToColumn);
  CHECK_INPUT(edgeToRow);

  int num_nodes = nodePointer.size(0) - 1;
  int num_edges = edgeList.size(0);
  int embedding_dim = input.size(1);
  int hidden_dim = weights.size(1);

  return spmm_forward_plus_GIN_final_fused(input, nodePointer, edgeList, 
                            blockPartition, edgeToColumn, edgeToRow, 
                            num_nodes, num_edges, embedding_dim, hidden_dim,
                            hybrid_type, row_nzr, col_nzr, weights
                            );
}

// condense an sorted array with duplication: [1,2,2,3,4,5,5]
// after condense, it becomes: [1,2,3,4,5].
// Also, mapping the origin value to the corresponding new location in the new array.
// 1->[0], 2->[1], 3->[2], 4->[3], 5->[4]. 
std::map<unsigned, unsigned> inplace_deduplication(unsigned* array, unsigned length){
    int loc=0, cur=1;
    std::map<unsigned, unsigned> nb2col;
    nb2col[array[0]] = 0;
    while (cur < length){
        if(array[cur] != array[cur - 1]){
            loc++;
            array[loc] = array[cur];
            nb2col[array[cur]] = loc;       // mapping from eid to TC_block column index.[]
        }
        cur++;
    }
    return nb2col;
}

void preprocess(torch::Tensor edgeList_tensor, 
                torch::Tensor nodePointer_tensor, 
                int num_nodes, 
                int blockSize_h,
                int blockSize_w,
                torch::Tensor blockPartition_tensor, 
                torch::Tensor edgeToColumn_tensor,
                torch::Tensor edgeToRow_tensor,
                torch::Tensor hybrid_type_tensor,
                torch::Tensor row_nzr_tensor,
                torch::Tensor col_nzr_tensor
                ){

    // input tensors.
    auto edgeList = edgeList_tensor.accessor<int, 1>();
    auto nodePointer = nodePointer_tensor.accessor<int, 1>();

    // output tensors.
    auto blockPartition = blockPartition_tensor.accessor<int, 1>();
    auto edgeToColumn = edgeToColumn_tensor.accessor<int, 1>();
    auto edgeToRow = edgeToRow_tensor.accessor<int, 1>();
    auto hybrid_type = hybrid_type_tensor.accessor<int, 1>();
    auto row_nzr = row_nzr_tensor.accessor<int, 1>();
    auto col_nzr = col_nzr_tensor.accessor<int, 1>();

    unsigned block_counter = 0;
    int csr_block_num = 0, tcu_block_num = 0, all_csr_k = 0, all_tcu_k = 0, max_k = 0, max_edges = 0;

    #pragma omp parallel for 
    for (unsigned nid = 0; nid < num_nodes; nid++){
        for (unsigned eid = nodePointer[nid]; eid < nodePointer[nid+1]; eid++)
            edgeToRow[eid] = nid;
    }

    #pragma omp parallel for reduction(+:block_counter)
    for (unsigned iter = 0; iter < num_nodes + 1; iter +=  blockSize_h){
        unsigned windowId = iter / blockSize_h;
        unsigned block_start = nodePointer[iter];
        unsigned block_end = nodePointer[min(iter + blockSize_h, num_nodes)];
        unsigned num_window_edges = block_end - block_start;
        unsigned *neighbor_window = (unsigned *) malloc (num_window_edges * sizeof(unsigned));
        memcpy(neighbor_window, &edgeList[block_start], num_window_edges * sizeof(unsigned));

        // Step-1: Sort the neighbor id array of a row window.
        thrust::sort(neighbor_window, neighbor_window + num_window_edges);

        // Step-2: Deduplication of the edge id array.
        // printf("Before dedupblication: %d\n", num_window_edges);
        std::map<unsigned, unsigned> clean_edges2col = inplace_deduplication(neighbor_window, num_window_edges);

        // generate blockPartition --> number of TC_blcok in each row window.
        blockPartition[windowId] = (clean_edges2col.size() + blockSize_w - 1) /blockSize_w;
        block_counter += blockPartition[windowId];

        // if((float)blockPartition[windowId] * 3.739 - 0.0328 * (float)num_window_edges - 12.61 > 0){
        if(clean_edges2col.size() > 32 || (float)clean_edges2col.size() * 0.19854024 - ((float)num_window_edges / (blockPartition[windowId] * 16 * 8)) * 6.578043 - 3.14922857 > 0){
            all_csr_k += blockPartition[windowId];
            csr_block_num++;
            hybrid_type[windowId] = 0;
            if(num_window_edges > max_edges){
                max_edges = num_window_edges;
                // if(max_edges == 2388) printf("%d\n", windowId);
            }
        }
        else{
            all_tcu_k += blockPartition[windowId];
            tcu_block_num++;
            if(max_k < blockPartition[windowId]){
                max_k = blockPartition[windowId];
            }
            hybrid_type[windowId] = 1;
        }

        for (unsigned e_index = block_start; e_index < block_end; e_index++){
            unsigned eid = edgeList[e_index];
            edgeToColumn[e_index] = clean_edges2col[eid];
        }
    }

    int nzrn = 0;
    for(int i = 0; i < num_nodes; i++){
        if(nodePointer[i + 1] > nodePointer[i]){
            col_nzr[nzrn++] = i;
        }
        if(i % blockSize_h == blockSize_h - 1){
            row_nzr[i / blockSize_h + 1] = nzrn;
        }
    }
    // row_nzr[num_nodes / blockSize_h + 1] = nzrn;
    row_nzr[(num_nodes + blockSize_h - 1) / blockSize_h] = nzrn;

    printf("TC_Blocks:\t%d\nExp_Edges:\t%d\nmax_k:\t%d\ncsr:\t%d\ntcu:\t%d\nmax_edges:\t%d\n", block_counter, block_counter * 8 * 16, max_k, csr_block_num, tcu_block_num, max_edges);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("preprocess", &preprocess, "Preprocess Step (CPU)");

  // forward computation
  m.def("forward", &spmm_forward, "HCSPMM SPMM forward (CUDA)");
  m.def("forward_more", &spmm_forward_more, "HCSPMM SPMM forward more (CUDA)");
  m.def("forward_fixed32", &spmm_forward_fixed32, "HCSPMM SPMM forward fixed32 (CUDA)");
  m.def("forward_fixed32_fused", &spmm_forward_fixed32_fused, "HCSPMM SPMM forward fixed32 fused (CUDA)");
  m.def("forward_final_fused", &spmm_forward_final_fused, "HCSPMM SPMM forward final fused (CUDA)");
  m.def("forward_fixed64", &spmm_forward_fixed64, "HCSPMM SPMM forward fixed64 (CUDA)");
  m.def("forward_fixed64_fused", &spmm_forward_fixed64_fused, "HCSPMM SPMM forward fixed64 fused (CUDA)");
  m.def("forward_final_fused_64", &spmm_forward_final_fused_64, "HCSPMM SPMM forward final fused 64 (CUDA)");
  m.def("forward_GIN_final_fused", &spmm_forward_GIN_final_fused, "HCSPMM SPMM forward for GIN final fused (CUDA)");
//   m.def("forward_fixed32_fused_all", &spmm_forward_fixed32_fused_all, "HCSPMM SPMM forward fixed32 fused all (CUDA)");

  // backward
  m.def("backward", &spmm_forward, "HCSPMM SPMM backward (CUDA)");
  m.def("backward_fixed32", &spmm_forward_fixed32, "HCSPMM SPMM backward fixed32 (CUDA)");
  m.def("backward_fixed32_fused", &spmm_forward_fixed32_fused, "HCSPMM SPMM backward fixed32 fused (CUDA)");
  m.def("backward_final_fused", &spmm_forward_final_fused, "HCSPMM SPMM backward final fused (CUDA)");
  m.def("backward_fixed64", &spmm_forward_fixed64, "HCSPMM SPMM backward fixed 64 (CUDA)");
  m.def("backward_fixed64_fused", &spmm_forward_fixed64_fused, "HCSPMM SPMM backward fixed 64 fused (CUDA)");
  m.def("backward_final_fused_64", &spmm_forward_final_fused_64, "HCSPMM SPMM backward final fused 64 (CUDA)");
  m.def("backward_GIN_final_fused", &spmm_forward_GIN_final_fused, "HCSPMM SPMM backward for GIN final fused (CUDA)");
//   m.def("backward_fixed32_fused_all", &spmm_forward_fixed32_fused_all, "HCSPMM SPMM backward fixed32 fused all (CUDA)");
}

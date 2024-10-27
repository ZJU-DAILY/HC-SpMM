#include <torch/extension.h>
#include <vector>
#include <string.h>
#include <cstdlib>
#include <map>

#include <thrust/sort.h>
#include <thrust/execution_policy.h>


#define min(x, y) (((x) < (y))? (x) : (y))

std::vector<torch::Tensor> preprocess(torch::Tensor edgeList_tensor, 
                torch::Tensor nodePointer_tensor,
                int num_nodes, 
                int edge_num,
                int block_num);

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

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("preprocess", &preprocess, "Preprocess Step (GPU)");

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

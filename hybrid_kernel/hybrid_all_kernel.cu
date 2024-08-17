#include <torch/extension.h>
#include <stdio.h>
#include <vector>
#include <cublas.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include<chrono>

#include <cuda.h>
#include <mma.h>
#include <cuda_runtime.h>

#include "config.h"

#define WPBMore 6
#define WPB 3
#define MAX_BLK 3
// #define BLOCKNUM 196133
// #define TMPSIZE 5
#define S_SIZE 62

using namespace nvcuda;

__global__ void spmm_forward_cuda_kernel_arbi_warps_hybrid_adaptive(
        const int * __restrict__ nodePointer,		// node pointer.
        const int *__restrict__ edgeList,			// edge list.
        const int *__restrict__ blockPartition, 	// number of TC_blocks (16x8) in each row_window.
        const int *__restrict__ edgeToColumn, 		// eid -> col within each row_window.
        const int *__restrict__ edgeToRow, 		    // eid -> col within each row_window.
        const int numNodes,
        const int numEdges,
        const int embedding_dim,				    // embedding dimension.
        const float *__restrict__ input,		    // input feature matrix.
        float *output,							    // aggreAGNNed output feature matrix.
        const int *__restrict__ hybrid_type,
        const int *__restrict__ row_nzr,
        const int *__restrict__ col_nzr
);

__global__ void spmm_forward_cuda_kernel_arbi_warps_hybrid_adaptive_more(
        const int * __restrict__ nodePointer,		// node pointer.
        const int *__restrict__ edgeList,			// edge list.
        const int *__restrict__ blockPartition, 	// number of TC_blocks (16x8) in each row_window.
        const int *__restrict__ edgeToColumn, 		// eid -> col within each row_window.
        const int *__restrict__ edgeToRow, 		    // eid -> col within each row_window.
        const int numNodes,
        const int numEdges,
        const int embedding_dim,				    // embedding dimension.
        const float *__restrict__ input,		    // input feature matrix.
        float *output,							    // aggreAGNNed output feature matrix.
        const int *__restrict__ hybrid_type,
        const int *__restrict__ row_nzr,
        const int *__restrict__ col_nzr
);

__global__ void spmm_forward_cuda_kernel_arbi_warps_hybrid_32(
        const int * __restrict__ nodePointer,		// node pointer.
        const int *__restrict__ edgeList,			// edge list.
        const int *__restrict__ blockPartition, 	// number of TC_blocks (16x8) in each row_window.
        const int *__restrict__ edgeToColumn, 		// eid -> col within each row_window.
        const int *__restrict__ edgeToRow, 		    // eid -> col within each row_window.
        const int numNodes,
        const int numEdges,
        const int embedding_dim,				    // embedding dimension.
        const float *__restrict__ input,		    // input feature matrix.
        float *output,							    // aggreAGNNed output feature matrix.
        const int *__restrict__ hybrid_type,
        const int *__restrict__ row_nzr,
        const int *__restrict__ col_nzr
);

__global__ void spmm_forward_cuda_kernel_arbi_warps_hybrid_64(
        const int * __restrict__ nodePointer,		// node pointer.
        const int *__restrict__ edgeList,			// edge list.
        const int *__restrict__ blockPartition, 	// number of TC_blocks (16x8) in each row_window.
        const int *__restrict__ edgeToColumn, 		// eid -> col within each row_window.
        const int *__restrict__ edgeToRow, 		    // eid -> col within each row_window.
        const int numNodes,
        const int numEdges,
        const int embedding_dim,				    // embedding dimension.
        const float *__restrict__ input,		    // input feature matrix.
        float *output,							    // aggreAGNNed output feature matrix.
        const int *__restrict__ hybrid_type,
        const int *__restrict__ row_nzr,
        const int *__restrict__ col_nzr
);

__global__ void spmm_forward_cuda_kernel_arbi_warps_hybrid_32_fused(
        const int * __restrict__ nodePointer,		// node pointer.
        const int *__restrict__ edgeList,			// edge list.
        const int *__restrict__ blockPartition, 	// number of TC_blocks (16x8) in each row_window.
        const int *__restrict__ edgeToColumn, 		// eid -> col within each row_window.
        const int *__restrict__ edgeToRow, 		    // eid -> col within each row_window.
        const int numNodes,
        const int numEdges,
        const int embedding_dim,				    // embedding dimension.
        const int hidden_dim,
        const float *__restrict__ input,		    // input feature matrix.
        float *output,							    // aggreAGNNed output feature matrix.
        float *output2,
        const int *__restrict__ hybrid_type,
        const int *__restrict__ row_nzr,
        const int *__restrict__ col_nzr,
        const float *__restrict__ weights
);

__global__ void spmm_forward_cuda_kernel_arbi_warps_hybrid_64_fused(
        const int * __restrict__ nodePointer,		// node pointer.
        const int *__restrict__ edgeList,			// edge list.
        const int *__restrict__ blockPartition, 	// number of TC_blocks (16x8) in each row_window.
        const int *__restrict__ edgeToColumn, 		// eid -> col within each row_window.
        const int *__restrict__ edgeToRow, 		    // eid -> col within each row_window.
        const int numNodes,
        const int numEdges,
        const int embedding_dim,				    // embedding dimension.
        const int hidden_dim,
        const float *__restrict__ input,		    // input feature matrix.
        float *output,							    // aggreAGNNed output feature matrix.
        float *output2,
        const int *__restrict__ hybrid_type,
        const int *__restrict__ row_nzr,
        const int *__restrict__ col_nzr,
        const float *__restrict__ weights
);

// __global__ void spmm_forward_cuda_kernel_arbi_warps_hybrid_32_fused_all(
//         const int * __restrict__ nodePointer,		// node pointer.
//         const int *__restrict__ edgeList,			// edge list.
//         const int *__restrict__ blockPartition, 	// number of TC_blocks (16x8) in each row_window.
//         const int *__restrict__ edgeToColumn, 		// eid -> col within each row_window.
//         const int *__restrict__ edgeToRow, 		    // eid -> col within each row_window.
//         const int numNodes,
//         const int numEdges,
//         const int embedding_dim,				    // embedding dimension.
//         const int hidden_dim,
//         const float *__restrict__ input,		    // input feature matrix.
//         float *output,
//         float *output2,							    // aggreAGNNed output feature matrix.
//         const int *__restrict__ hybrid_type,
//         const int *__restrict__ row_nzr,
//         const int *__restrict__ col_nzr,
//         const float *__restrict__ weights,
//         const float *__restrict__ X
// );

__global__ void spmm_forward_cuda_kernel_arbi_warps_hybrid_final_fused(
        const int * __restrict__ nodePointer,		// node pointer.
        const int *__restrict__ edgeList,			// edge list.
        const int *__restrict__ blockPartition, 	// number of TC_blocks (16x8) in each row_window.
        const int *__restrict__ edgeToColumn, 		// eid -> col within each row_window.
        const int *__restrict__ edgeToRow, 		    // eid -> col within each row_window.
        const int numNodes,
        const int numEdges,
        const int embedding_dim,				    // embedding dimension.
        const int hidden_dim,
        const float *__restrict__ input,		    // input feature matrix.
        float *output,
        float *output2,							    // aggreAGNNed output feature matrix.
        const int *__restrict__ hybrid_type,
        const int *__restrict__ row_nzr,
        const int *__restrict__ col_nzr,
        const float *__restrict__ weights
);

__global__ void spmm_forward_cuda_kernel_arbi_warps_hybrid_final_fused_64(
        const int * __restrict__ nodePointer,		// node pointer.
        const int *__restrict__ edgeList,			// edge list.
        const int *__restrict__ blockPartition, 	// number of TC_blocks (16x8) in each row_window.
        const int *__restrict__ edgeToColumn, 		// eid -> col within each row_window.
        const int *__restrict__ edgeToRow, 		    // eid -> col within each row_window.
        const int numNodes,
        const int numEdges,
        const int embedding_dim,				    // embedding dimension.
        const int hidden_dim,
        const float *__restrict__ input,		    // input feature matrix.
        float *output,
        float *output2,							    // aggreAGNNed output feature matrix.
        const int *__restrict__ hybrid_type,
        const int *__restrict__ row_nzr,
        const int *__restrict__ col_nzr,
        const float *__restrict__ weights
);

__global__ void spmm_forward_cuda_kernel_arbi_warps_hybrid_GIN_final_fused(
        const int * __restrict__ nodePointer,		// node pointer.
        const int *__restrict__ edgeList,			// edge list.
        const int *__restrict__ blockPartition, 	// number of TC_blocks (16x8) in each row_window.
        const int *__restrict__ edgeToColumn, 		// eid -> col within each row_window.
        const int *__restrict__ edgeToRow, 		    // eid -> col within each row_window.
        const int numNodes,
        const int numEdges,
        const int embedding_dim,				    // embedding dimension.
        const int hidden_dim,
        const float *__restrict__ input,		    // input feature matrix.
        float *output,
        float *output2,							    // aggreAGNNed output feature matrix.
        const int *__restrict__ hybrid_type,
        const int *__restrict__ row_nzr,
        const int *__restrict__ col_nzr,
        const float *__restrict__ weights
);
// const int node_num = 334925, edge_num = 1686092, embedding_dim = 112, real_embedding_dim = 100;
// const int node_num = 410236, edge_num = 4878875, embedding_dim = 32;
// const int node_num = 1710902, edge_num = 3636546, embedding_dim = 32;
// const int node_num = 19717, edge_num = 88676, embedding_dim = 32;

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
) {
    const int WARPperBlock = WPB;

    const int dimTileNum = (embedding_dim + BLK_H - 1) / BLK_H;
    const int dynamic_shared_size = dimTileNum * BLK_W * BLK_H * sizeof(float); // dynamic shared memory.

    // auto output = torch::zeros_like(input);

    auto devid = input.device().index();
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA, devid);
    auto output = torch::empty({num_nodes, embedding_dim}, options);

    dim3 grid(blockPartition.size(0), 1, 1);
    dim3 block(WARP_SIZE, WARPperBlock, 1);

    spmm_forward_cuda_kernel_arbi_warps_hybrid_adaptive<<<grid, block, dynamic_shared_size>>>(
            nodePointer.data<int>(),
            edgeList.data<int>(),
            blockPartition.data<int>(),
            edgeToColumn.data<int>(),
            edgeToRow.data<int>(),
            num_nodes,
            num_edges,
            embedding_dim,
            input.data<float>(),
            output.data<float>(),
            hybrid_type.data<int>(),
            row_nzr.data<int>(),
            col_nzr.data<int>()
    );
    
    return {output};
}

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
) {
    const int WARPperBlock = WPBMore;

    const int dimTileNum = (embedding_dim + BLK_H - 1) / BLK_H;
    const int dynamic_shared_size = dimTileNum * BLK_W * BLK_H * sizeof(float); // dynamic shared memory.

    auto output = torch::zeros_like(input);

    dim3 grid(blockPartition.size(0), 1, 1);
    dim3 block(WARP_SIZE, WARPperBlock, 1);

    spmm_forward_cuda_kernel_arbi_warps_hybrid_adaptive_more<<<grid, block, dynamic_shared_size>>>(
            nodePointer.data<int>(),
            edgeList.data<int>(),
            blockPartition.data<int>(),
            edgeToColumn.data<int>(),
            edgeToRow.data<int>(),
            num_nodes,
            num_edges,
            embedding_dim,
            input.data<float>(),
            output.data<float>(),
            hybrid_type.data<int>(),
            row_nzr.data<int>(),
            col_nzr.data<int>()
    );
    
    return {output};
}

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
) {
    // const int WARPperBlock = WPB;

    const int dimTileNum = (embedding_dim + BLK_H - 1) / BLK_H;
    const int dynamic_shared_size = dimTileNum * BLK_W * BLK_H * sizeof(float); // dynamic shared memory.

    // auto output = torch::zeros_like(input);
    auto devid = input.device().index();
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA, devid);
    auto output = torch::empty({num_nodes, embedding_dim}, options);

    dim3 grid(blockPartition.size(0), 1, 1);
    dim3 block(WARP_SIZE, WPB, 1);

    // auto start = std::chrono::high_resolution_clock::now();

    spmm_forward_cuda_kernel_arbi_warps_hybrid_32<<<grid, block, dynamic_shared_size>>>(
            nodePointer.data<int>(),
            edgeList.data<int>(),
            blockPartition.data<int>(),
            edgeToColumn.data<int>(),
            edgeToRow.data<int>(),
            num_nodes,
            num_edges,
            embedding_dim,
            input.data<float>(),
            output.data<float>(),
            hybrid_type.data<int>(),
            row_nzr.data<int>(),
            col_nzr.data<int>()
    );

    // cudaDeviceSynchronize();
    // auto end = std::chrono::high_resolution_clock::now();
    // std::chrono::duration<float> duration = end - start;
    // std::cout << "All time: " << duration.count() << "s" << std::endl;
    
    return {output};
}

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
) {
    // const int WARPperBlock = WPB;

    const int dimTileNum = (embedding_dim + BLK_H - 1) / BLK_H;
    const int dynamic_shared_size = dimTileNum * BLK_W * BLK_H * sizeof(float); // dynamic shared memory.

    auto output = torch::zeros_like(input);

    dim3 grid(blockPartition.size(0), 1, 1);
    dim3 block(WARP_SIZE, WPB, 1);

    spmm_forward_cuda_kernel_arbi_warps_hybrid_64<<<grid, block, dynamic_shared_size>>>(
            nodePointer.data<int>(),
            edgeList.data<int>(),
            blockPartition.data<int>(),
            edgeToColumn.data<int>(),
            edgeToRow.data<int>(),
            num_nodes,
            num_edges,
            embedding_dim,
            input.data<float>(),
            output.data<float>(),
            hybrid_type.data<int>(),
            row_nzr.data<int>(),
            col_nzr.data<int>()
    );
    
    return {output};
}

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
) {
    // const int WARPperBlock = WPB;

    const int dimTileNum = (embedding_dim + BLK_H - 1) / BLK_H;
    const int dynamic_shared_size = dimTileNum * BLK_W * BLK_H * sizeof(float); // dynamic shared memory.

    // auto output2 = torch::zeros_like(input);
    // auto output = torch::zeros_like(input);
    const auto k = input.size(1);
    auto devid = input.device().index();
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA, devid);
    auto output = torch::empty({num_nodes, k}, options);
    auto output2 = torch::empty({num_nodes, k}, options);

    // printf("dim: %d, %d\n", num_nodes, hidden_dim);
    dim3 grid(blockPartition.size(0), 1, 1);
    dim3 block(WARP_SIZE, WPB, 1);
    // printf("hhhh %.1f\n", output.data<float>());
    spmm_forward_cuda_kernel_arbi_warps_hybrid_32_fused<<<grid, block, dynamic_shared_size * 2>>>(
            nodePointer.data<int>(),
            edgeList.data<int>(),
            blockPartition.data<int>(),
            edgeToColumn.data<int>(),
            edgeToRow.data<int>(),
            num_nodes,
            num_edges,
            embedding_dim,
            hidden_dim,
            input.data<float>(),
            output.data<float>(),
            output2.data<float>(),
            hybrid_type.data<int>(),
            row_nzr.data<int>(),
            col_nzr.data<int>(),
            weights.data<float>()
    );
    
    return {output, output2};
}

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
) {
    // const int WARPperBlock = WPB;

    const int dimTileNum = (embedding_dim + BLK_H - 1) / BLK_H;
    const int dynamic_shared_size = dimTileNum * BLK_W * BLK_H * sizeof(float); // dynamic shared memory.

    auto output2 = torch::zeros_like(input);
    // auto output = torch::zeros({num_nodes, hidden_dim});
    auto output = torch::zeros_like(input);
    // printf("dim: %d, %d\n", num_nodes, hidden_dim);
    dim3 grid(blockPartition.size(0), 1, 1);
    dim3 block(WARP_SIZE, WPB, 1);
    // printf("hhhh %.1f\n", output.data<float>());
    spmm_forward_cuda_kernel_arbi_warps_hybrid_64_fused<<<grid, block, dynamic_shared_size * 2>>>(
            nodePointer.data<int>(),
            edgeList.data<int>(),
            blockPartition.data<int>(),
            edgeToColumn.data<int>(),
            edgeToRow.data<int>(),
            num_nodes,
            num_edges,
            embedding_dim,
            hidden_dim,
            input.data<float>(),
            output.data<float>(),
            output2.data<float>(),
            hybrid_type.data<int>(),
            row_nzr.data<int>(),
            col_nzr.data<int>(),
            weights.data<float>()
    );
    
    return {output, output2};
}

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
) {
    // const int WARPperBlock = WPB;

    const int dimTileNum = (embedding_dim + BLK_H - 1) / BLK_H;
    const int dynamic_shared_size = dimTileNum * BLK_W * BLK_H * sizeof(float); // dynamic shared memory.

    // auto output2 = torch::zeros_like(input);
    const auto k = input.size(1);

    auto devid = input.device().index();
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA, devid);
    // auto output = torch::empty({num_nodes, k}, options);
    auto output2 = torch::empty({num_nodes, k}, options);

    // auto output = torch::zeros({num_nodes, hidden_dim});
    // auto output = torch::zeros_like(input);
    // printf("dim: %d, %d\n", num_nodes, hidden_dim);
    dim3 grid(blockPartition.size(0), 1, 1);
    dim3 block(WARP_SIZE, WPB, 1);
    // printf("hhhh %.1f\n", output.data<float>());
    spmm_forward_cuda_kernel_arbi_warps_hybrid_final_fused<<<grid, block, dynamic_shared_size * 2>>>(
            nodePointer.data<int>(),
            edgeList.data<int>(),
            blockPartition.data<int>(),
            edgeToColumn.data<int>(),
            edgeToRow.data<int>(),
            num_nodes,
            num_edges,
            embedding_dim,
            hidden_dim,
            input.data<float>(),
            output.data<float>(),
            output2.data<float>(),
            hybrid_type.data<int>(),
            row_nzr.data<int>(),
            col_nzr.data<int>(),
            weights.data<float>()
    );
    
    return {output, output2};
}

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
) {
    // const int WARPperBlock = WPB;

    const int dimTileNum = (embedding_dim + BLK_H - 1) / BLK_H;
    const int dynamic_shared_size = dimTileNum * BLK_W * BLK_H * sizeof(float); // dynamic shared memory.

    auto output2 = torch::zeros_like(input);
    // auto output = torch::zeros({num_nodes, hidden_dim});
    // auto output = torch::zeros_like(input);
    // printf("dim: %d, %d\n", num_nodes, hidden_dim);
    dim3 grid(blockPartition.size(0), 1, 1);
    dim3 block(WARP_SIZE, WPB, 1);
    // printf("hhhh %.1f\n", output.data<float>());
    spmm_forward_cuda_kernel_arbi_warps_hybrid_final_fused_64<<<grid, block, dynamic_shared_size * 2>>>(
            nodePointer.data<int>(),
            edgeList.data<int>(),
            blockPartition.data<int>(),
            edgeToColumn.data<int>(),
            edgeToRow.data<int>(),
            num_nodes,
            num_edges,
            embedding_dim,
            hidden_dim,
            input.data<float>(),
            output.data<float>(),
            output2.data<float>(),
            hybrid_type.data<int>(),
            row_nzr.data<int>(),
            col_nzr.data<int>(),
            weights.data<float>()
    );
    
    return {output, output2};
}

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
) {
    // const int WARPperBlock = WPB;

    const int dimTileNum = (embedding_dim + BLK_H - 1) / BLK_H;
    const int dynamic_shared_size = dimTileNum * BLK_W * BLK_H * sizeof(float); // dynamic shared memory.

    // auto output2 = torch::zeros_like(input);
    // auto output = torch::zeros_like(input);
    const auto k = input.size(1);
    auto devid = input.device().index();
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA, devid);
    auto output = torch::empty({num_nodes, hidden_dim}, options);
    auto output2 = torch::empty({num_nodes, k}, options);

    // printf("dim: %d, %d\n", num_nodes, hidden_dim);
    dim3 grid(blockPartition.size(0), 1, 1);
    dim3 block(WARP_SIZE, WPB, 1);
    // printf("hhhh %.1f\n", output.data<float>());
    spmm_forward_cuda_kernel_arbi_warps_hybrid_GIN_final_fused<<<grid, block, dynamic_shared_size * 2>>>(
            nodePointer.data<int>(),
            edgeList.data<int>(),
            blockPartition.data<int>(),
            edgeToColumn.data<int>(),
            edgeToRow.data<int>(),
            num_nodes,
            num_edges,
            embedding_dim,
            hidden_dim,
            input.data<float>(),
            output.data<float>(),
            output2.data<float>(),
            hybrid_type.data<int>(),
            row_nzr.data<int>(),
            col_nzr.data<int>(),
            weights.data<float>()
    );
    
    return {output, output2};
}
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
// ) {
//     // const int WARPperBlock = WPB;

//     const int dimTileNum = (embedding_dim + BLK_H - 1) / BLK_H;
//     const int dynamic_shared_size = dimTileNum * BLK_W * BLK_H * sizeof(float); // dynamic shared memory.

//     // auto output2 = torch::cat({torch::zeros_like(input), torch::zeros_like(input)}, 0);
//     // auto output2 = torch::zeros({20933, 1024}).to(input.device());
//     // auto output = torch::zeros({num_nodes, hidden_dim}).to(torch::float).to();
//     auto output = torch::zeros_like(input);
//     // printf("dim: %d, %d\n", num_nodes, hidden_dim);
//     dim3 grid(blockPartition.size(0), 1, 1);
//     dim3 block(WARP_SIZE, 4, 1);
//     // printf("hhhh %.1f\n", output.data<float>());
//     spmm_forward_cuda_kernel_arbi_warps_hybrid_32_fused_all<<<grid, block, dynamic_shared_size * 2>>>(
//             nodePointer.data<int>(),
//             edgeList.data<int>(),
//             blockPartition.data<int>(),
//             edgeToColumn.data<int>(),
//             edgeToRow.data<int>(),
//             num_nodes,
//             num_edges,
//             embedding_dim,
//             hidden_dim,
//             input.data<float>(),
//             output.data<float>(),
//             output2.data<float>(),
//             hybrid_type.data<int>(),
//             row_nzr.data<int>(),
//             col_nzr.data<int>(),
//             weights.data<float>(),
//             X.data<float>()
//     );
    
//     return {output, output2};
// }


__global__ void spmm_forward_cuda_kernel_arbi_warps_hybrid_adaptive(
        const int * __restrict__ nodePointer,		// node pointer.
        const int *__restrict__ edgeList,			// edge list.
        const int *__restrict__ blockPartition, 	// number of TC_blocks (16x8) in each row_window.
        const int *__restrict__ edgeToColumn, 		// eid -> col within each row_window.
        const int *__restrict__ edgeToRow, 		    // eid -> col within each row_window.
        const int numNodes,
        const int numEdges,
        const int embedding_dim,				    // embedding dimension.
        const float *__restrict__ input,		    // input feature matrix.
        float *output,							    // aggreAGNNed output feature matrix.
        const int *__restrict__ hybrid_type,
        const int *__restrict__ row_nzr,
        const int *__restrict__ col_nzr
) {
    unsigned bid = blockIdx.x;								// block_index == row_window_index
    unsigned wid = threadIdx.y;								// warp_index handling multi-dimension > 16.
    const unsigned laneid = threadIdx.x;							// lanid of each warp.
    const unsigned tid = threadIdx.y * blockDim.x + laneid;			// threadid of each block.
    // const unsigned warpSize = blockDim.x;							// number of threads per warp.
    const unsigned threadPerBlock = blockDim.x * blockDim.y;		// number of threads per block.

    const unsigned dimTileNum = embedding_dim / BLK_H;              // number of tiles along the dimension
    const unsigned shared_size = (embedding_dim + 31) / 32;
    unsigned nIdx_start = bid * BLK_H;					    // starting nodeIdx of current row_window.
    unsigned nIdx_end = min((bid + 1) * BLK_H, numNodes);		// ending nodeIdx of current row_window.

    unsigned eIdx_start = nodePointer[nIdx_start];			// starting edgeIdx of current row_window.
    unsigned eIdx_end = nodePointer[nIdx_end];				// ending edgeIdx of the current row_window.
    unsigned num_TC_blocks = blockPartition[bid]; 			// number of TC_blocks of the current row_window.
    const unsigned dense_bound = numNodes * embedding_dim;

    unsigned csr_full = embedding_dim / 32, csr_reserve = embedding_dim % 32;

    __shared__ float sparse_A[BLK_H * BLK_W * MAX_BLK];					// row-major sparse matrix shared memory store.
    __shared__ int sparse_AToX_index[BLK_W * MAX_BLK];					// TC_block col to dense_tile row.

    extern __shared__ float dense_X[];
    __shared__ int tmp_A[S_SIZE];
    // __shared__ float tmp_acc[WPB * TMPSIZE * 32];

    if(hybrid_type[bid] == 0){
        // int end_nzr = row_nzr[bid + 1];
        int end_nzr = (bid + 1) * BLK_H > numNodes ? numNodes : (bid + 1) * BLK_H;
        
        unsigned begin_edge = nodePointer[bid * 16], end_edge = nodePointer[min((bid + 1) * 16, numNodes)];
        for(int i = begin_edge + tid; i < end_edge; i += threadPerBlock){
            tmp_A[i - begin_edge] = edgeList[i];
        }
        __syncthreads();

        // 32
        if(csr_full > 0){
            // for (int z = row_nzr[bid] + wid; z < end_nzr; z += WPB) {
            for (int z = bid * BLK_H + wid; z < end_nzr; z += WPB) {
                // int row = col_nzr[z];
                int row = z;
                int target_id = row * embedding_dim + laneid;
                int begin_col_id = nodePointer[row], end_col_id = nodePointer[row + 1];
                for(int i = 0; i < csr_full; i++){
                    dense_X[wid * shared_size * 32 + i * 32 + laneid] = 0.0;
                }
                __syncthreads();
                for (int j = begin_col_id; j < end_col_id; j++) {
                    // int cur_edge = edgeList[j];
                    int cur_edge = tmp_A[j - begin_edge];
                    for(int i = 0; i < csr_full; i++){
                        dense_X[wid * shared_size * 32 + i * 32 + laneid] += input[laneid + cur_edge * embedding_dim + i * 32];
                    }
                }
                for(int i = 0; i < csr_full; i++){
                    output[target_id + i * 32] = dense_X[wid * shared_size * 32 + i * 32 + laneid];
                }
            }
        }
        // 32
        if(csr_reserve > 0){
            if(csr_reserve > 16){
                // int end_nzr = row_nzr[bid + 1];
                // for (int z = row_nzr[bid] + wid; z < end_nzr; z += WPB) {
                for (int z = bid * BLK_H + wid; z < end_nzr; z += WPB) {
                    // int row = col_nzr[z];
                    int row = z;
                    int target_id = row * embedding_dim + laneid + csr_full * 32;
                    if(laneid + csr_full * 32 < embedding_dim){
                        int begin_col_id = nodePointer[row], end_col_id = nodePointer[row + 1];
                        float acc = 0.0;
                        for (int j = begin_col_id; j < end_col_id; j++) {
                            // acc += input[laneid + edgeList[j] * embedding_dim + csr_full * 32];
                            acc += input[laneid + tmp_A[j - begin_edge] * embedding_dim + csr_full * 32];
                        }
                        output[target_id] = acc;
                    }
                }
            
            }
            // 16
            else{
                int off = laneid < 16 ? 0 : 1, col_offset = laneid < 16 ? laneid : laneid - 16;
                // for (int z = row_nzr[bid] + wid * 2; z < end_nzr; z += 2 * WPB) {
                for (int z = bid * BLK_H + wid * 2; z < end_nzr; z += 2 * WPB) {
                    if(z + off < end_nzr) {
                        // int row = col_nzr[z + off];
                        int row = z + off;
                        int target_id = row * embedding_dim + col_offset + csr_full * 32;
                        if(col_offset + csr_full * 32 < embedding_dim){
                            int begin_col_id = nodePointer[row], end_col_id = nodePointer[row + 1];
                            float acc = 0.0;
                            for (int j = begin_col_id; j < end_col_id; j++) {
                                // acc += input[col_offset + edgeList[j] * embedding_dim + csr_full * 32];
                                acc += input[col_offset + tmp_A[j - begin_edge] * embedding_dim + csr_full * 32];
                            }
                            output[target_id] = acc;
                        }
                    }
                }
            }
        }
    }

    else{
        wmma::fragment<wmma::matrix_a, BLK_H, BLK_H, BLK_W, wmma::precision::tf32, wmma::row_major> a_frag;
        wmma::fragment<wmma::matrix_b, BLK_H, BLK_H, BLK_W, wmma::precision::tf32, wmma::col_major> b_frag;
        wmma::fragment<wmma::accumulator, BLK_H, BLK_H, BLK_W, float> acc_frag;
        wmma::fill_fragment(acc_frag, 0.0f);

        nIdx_start = bid * BLK_H;					    // starting nodeIdx of current row_window.
        nIdx_end = min((bid + 1) * BLK_H, numNodes);		// ending nodeIdx of current row_window.

        eIdx_start = nodePointer[nIdx_start];			// starting edgeIdx of current row_window.
        eIdx_end = nodePointer[nIdx_end];				// ending edgeIdx of the current row_window.
        num_TC_blocks = blockPartition[bid];

        // Init A_colToX_row with dummy values.
        if (tid < BLK_W * MAX_BLK) {
            sparse_AToX_index[tid] = numNodes + 1;
        }

        __syncthreads();
        
        // Init sparse_A with zero values.
#pragma unroll
        for (unsigned idx = tid; idx < BLK_W * BLK_H * MAX_BLK; idx += threadPerBlock) {
            sparse_A[idx] = 0;
        }

        #pragma unroll
        // Init dense_X with zero values.
        for (unsigned idx = tid; idx < dimTileNum * BLK_W * BLK_H; idx += threadPerBlock) {
            dense_X[idx] = 0;
        }

#pragma unroll
        for (unsigned eIdx = eIdx_start + tid; eIdx < eIdx_end; eIdx += threadPerBlock) {
            unsigned col = edgeToColumn[eIdx];
            unsigned row_local = edgeToRow[eIdx] % BLK_H;
            unsigned blk_id = col / 8;
            unsigned col_local = col % 8;
            sparse_A[row_local * BLK_W + col_local + blk_id * BLK_H * BLK_W] = 1;        // set the edge of the sparse_A.
            sparse_AToX_index[col] = edgeList[eIdx];        // record the mapping from sparse_A colId to rowId of dense_X.
        }

        __syncthreads();

        for (unsigned i = 0; i < num_TC_blocks; i++) {
            
            int off = laneid < 16 ? 0 : 1, col_offset = laneid < 16 ? laneid : laneid - 16;
            #pragma unroll
            for (unsigned idx = wid; idx < 4 * dimTileNum; idx += WPB) {
                unsigned dense_rowIdx = sparse_AToX_index[(idx % 4) * 2 + i * BLK_W + off];                        // TC_block_col to dense_tile_row.
                unsigned source_idx = dense_rowIdx * embedding_dim + col_offset + (idx / 4) * BLK_H;
                unsigned target_idx = (col_offset + (idx / 4) * BLK_H) * BLK_W + (idx % 4) * 2 + off;
                // boundary test.
                if (col_offset + (idx / 4) * BLK_H < embedding_dim)
                    dense_X[target_idx] = source_idx < dense_bound ? __ldca(&input[source_idx]) : 0;
            }

            __syncthreads();

            if (wid < dimTileNum) {
                wmma::load_matrix_sync(a_frag, sparse_A + i * BLK_W * BLK_H, BLK_W);
                wmma::load_matrix_sync(b_frag, dense_X + wid * BLK_W * BLK_H, BLK_W);
#pragma unroll
                for (unsigned t = 0; t < a_frag.num_elements; t++) {
                    a_frag.x[t] = wmma::__float_to_tf32(a_frag.x[t]);
                }

#pragma unroll
                for (unsigned t = 0; t < b_frag.num_elements; t++) {
                    b_frag.x[t] = wmma::__float_to_tf32(b_frag.x[t]);
                }
                // Perform the matrix multiplication.
                wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
            }

        }
        if (wid < dimTileNum)
            // Store the matrix to output matrix.
            // * Note * embeeding dimension should be padded divisible by BLK_H for output correctness.
            wmma::store_matrix_sync(output + bid * BLK_H * embedding_dim + wid * BLK_H, acc_frag, embedding_dim,
                                    wmma::mem_row_major);

    }

}

__global__ void spmm_forward_cuda_kernel_arbi_warps_hybrid_adaptive_more(
        const int * __restrict__ nodePointer,		// node pointer.
        const int *__restrict__ edgeList,			// edge list.
        const int *__restrict__ blockPartition, 	// number of TC_blocks (16x8) in each row_window.
        const int *__restrict__ edgeToColumn, 		// eid -> col within each row_window.
        const int *__restrict__ edgeToRow, 		    // eid -> col within each row_window.
        const int numNodes,
        const int numEdges,
        const int embedding_dim,				    // embedding dimension.
        const float *__restrict__ input,		    // input feature matrix.
        float *output,							    // aggreAGNNed output feature matrix.
        const int *__restrict__ hybrid_type,
        const int *__restrict__ row_nzr,
        const int *__restrict__ col_nzr
) {
    unsigned bid = blockIdx.x;								// block_index == row_window_index
    unsigned wid = threadIdx.y;								// warp_index handling multi-dimension > 16.
    const unsigned laneid = threadIdx.x;							// lanid of each warp.
    const unsigned tid = threadIdx.y * blockDim.x + laneid;			// threadid of each block.
    // const unsigned warpSize = blockDim.x;							// number of threads per warp.
    const unsigned threadPerBlock = blockDim.x * blockDim.y;		// number of threads per block.

    const unsigned dimTileNum = embedding_dim / BLK_H;              // number of tiles along the dimension
    const unsigned shared_size = embedding_dim / 32;
    unsigned nIdx_start = bid * BLK_H;					    // starting nodeIdx of current row_window.
    unsigned nIdx_end = min((bid + 1) * BLK_H, numNodes);		// ending nodeIdx of current row_window.

    unsigned eIdx_start = nodePointer[nIdx_start];			// starting edgeIdx of current row_window.
    unsigned eIdx_end = nodePointer[nIdx_end];				// ending edgeIdx of the current row_window.
    unsigned num_TC_blocks = blockPartition[bid]; 			// number of TC_blocks of the current row_window.
    const unsigned dense_bound = numNodes * embedding_dim;

    unsigned csr_full = embedding_dim / 32, csr_reserve = embedding_dim % 32;

    __shared__ float sparse_A[BLK_H * BLK_W * MAX_BLK];					// row-major sparse matrix shared memory store.
    __shared__ int sparse_AToX_index[BLK_W * MAX_BLK];					// TC_block col to dense_tile row.

    extern __shared__ float dense_X[];
    __shared__ int tmp_A[S_SIZE];
    // __shared__ float tmp_acc[WPB * TMPSIZE * 32];

    if(hybrid_type[bid] == 0){
        int end_nzr = row_nzr[bid + 1];
        unsigned begin_edge = nodePointer[bid * 16], end_edge = nodePointer[min((bid + 1) * 16, numNodes)];
        for(int i = begin_edge + tid; i < end_edge; i += threadPerBlock){
            tmp_A[i - begin_edge] = edgeList[i];
        }
        __syncthreads();
        // 32
        if(csr_full > 0){
            for (int z = row_nzr[bid] + wid; z < end_nzr; z += WPBMore) {
                int row = col_nzr[z];
                int target_id = row * embedding_dim + laneid;
                int begin_col_id = nodePointer[row], end_col_id = nodePointer[row + 1];
                // for(int i = 0; i < csr_full; i++){
                //     dense_X[wid * TMPSIZE * 32 + i * 32 + laneid] = 0.0;
                // }
                for (int j = begin_col_id; j < end_col_id; j++) {
                    // int cur_edge = edgeList[j];
                    int cur_edge = tmp_A[j - begin_edge];
                    for(int i = 0; i < csr_full; i++){
                        dense_X[wid * shared_size * 32 + i * 32 + laneid] += input[laneid + cur_edge * embedding_dim + i * 32];
                    }
                }
                for(int i = 0; i < csr_full; i++){
                    output[target_id + i * 32] = dense_X[wid * shared_size * 32 + i * 32 + laneid];
                }
            }
        }
        // 32
        if(csr_reserve > 0){
            if(csr_reserve > 16){
                int end_nzr = row_nzr[bid + 1];
                for (int z = row_nzr[bid] + wid; z < end_nzr; z += WPBMore) {
                    int row = col_nzr[z];
                    int target_id = row * embedding_dim + laneid + csr_full * 32;
                    if(laneid + csr_full * 32 < embedding_dim){
                        int begin_col_id = nodePointer[row], end_col_id = nodePointer[row + 1];
                        float acc = 0.0;
                        for (int j = begin_col_id; j < end_col_id; j++) {
                            // acc += input[laneid + edgeList[j] * embedding_dim + csr_full * 32];
                            acc += input[laneid + tmp_A[j - begin_edge] * embedding_dim + csr_full * 32];
                        }
                        output[target_id] = acc;
                    }
                }
            
            }
            // 16
            else{
                int off = laneid < 16 ? 0 : 1, col_offset = laneid < 16 ? laneid : laneid - 16;
                for (int z = row_nzr[bid] + wid * 2; z < end_nzr; z += 2 * WPBMore) {
                    if(z + off < end_nzr) {
                        int row = col_nzr[z + off];
                        int target_id = row * embedding_dim + col_offset + csr_full * 32;
                        if(col_offset + csr_full * 32 < embedding_dim){
                            int begin_col_id = nodePointer[row], end_col_id = nodePointer[row + 1];
                            float acc = 0.0;
                            for (int j = begin_col_id; j < end_col_id; j++) {
                                // acc += input[col_offset + edgeList[j] * embedding_dim + csr_full * 32];
                                acc += input[col_offset + tmp_A[j - begin_edge] * embedding_dim + csr_full * 32];
                            }
                            output[target_id] = acc;
                        }
                    }
                }
            }
        }
    }

    else{
        wmma::fragment<wmma::matrix_a, BLK_H, BLK_H, BLK_W, wmma::precision::tf32, wmma::row_major> a_frag;
        wmma::fragment<wmma::matrix_b, BLK_H, BLK_H, BLK_W, wmma::precision::tf32, wmma::col_major> b_frag;
        wmma::fragment<wmma::accumulator, BLK_H, BLK_H, BLK_W, float> acc_frag;
        wmma::fill_fragment(acc_frag, 0.0f);

        nIdx_start = bid * BLK_H;					    // starting nodeIdx of current row_window.
        nIdx_end = min((bid + 1) * BLK_H, numNodes);		// ending nodeIdx of current row_window.

        eIdx_start = nodePointer[nIdx_start];			// starting edgeIdx of current row_window.
        eIdx_end = nodePointer[nIdx_end];				// ending edgeIdx of the current row_window.
        num_TC_blocks = blockPartition[bid];

        // Init A_colToX_row with dummy values.
        if (tid < BLK_W * MAX_BLK) {
            sparse_AToX_index[tid] = numNodes + 1;
        }

        __syncthreads();
        
        // Init sparse_A with zero values.
#pragma unroll
        for (unsigned idx = tid; idx < BLK_W * BLK_H * MAX_BLK; idx += threadPerBlock) {
            sparse_A[idx] = 0;
        }

        #pragma unroll
        // Init dense_X with zero values.
        for (unsigned idx = tid; idx < dimTileNum * BLK_W * BLK_H; idx += threadPerBlock) {
            dense_X[idx] = 0;
        }

#pragma unroll
        for (unsigned eIdx = eIdx_start + tid; eIdx < eIdx_end; eIdx += threadPerBlock) {
            unsigned col = edgeToColumn[eIdx];
            unsigned row_local = edgeToRow[eIdx] % BLK_H;
            unsigned blk_id = col / 8;
            unsigned col_local = col % 8;
            sparse_A[row_local * BLK_W + col_local + blk_id * BLK_H * BLK_W] = 1;        // set the edge of the sparse_A.
            sparse_AToX_index[col] = edgeList[eIdx];        // record the mapping from sparse_A colId to rowId of dense_X.
        }

        __syncthreads();

        for (unsigned i = 0; i < num_TC_blocks; i++) {
            
            int off = laneid < 16 ? 0 : 1, col_offset = laneid < 16 ? laneid : laneid - 16;
            #pragma unroll
            for (unsigned idx = wid; idx < 4 * dimTileNum; idx += WPBMore) {
                unsigned dense_rowIdx = sparse_AToX_index[(idx % 4) * 2 + i * BLK_W + off];                        // TC_block_col to dense_tile_row.
                unsigned source_idx = dense_rowIdx * embedding_dim + col_offset + (idx / 4) * BLK_H;
                unsigned target_idx = (col_offset + (idx / 4) * BLK_H) * BLK_W + (idx % 4) * 2 + off;
                // boundary test.
                if (col_offset + (idx / 4) * BLK_H < embedding_dim)
                    dense_X[target_idx] = source_idx < dense_bound ? __ldca(&input[source_idx]) : 0;
            }

            __syncthreads();

            if (wid < dimTileNum) {
                wmma::load_matrix_sync(a_frag, sparse_A + i * BLK_W * BLK_H, BLK_W);
                wmma::load_matrix_sync(b_frag, dense_X + wid * BLK_W * BLK_H, BLK_W);
#pragma unroll
                for (unsigned t = 0; t < a_frag.num_elements; t++) {
                    a_frag.x[t] = wmma::__float_to_tf32(a_frag.x[t]);
                }

#pragma unroll
                for (unsigned t = 0; t < b_frag.num_elements; t++) {
                    b_frag.x[t] = wmma::__float_to_tf32(b_frag.x[t]);
                }
                // Perform the matrix multiplication.
                wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
            }

        }
        if (wid < dimTileNum)
            // Store the matrix to output matrix.
            // * Note * embeeding dimension should be padded divisible by BLK_H for output correctness.
            wmma::store_matrix_sync(output + bid * BLK_H * embedding_dim + wid * BLK_H, acc_frag, embedding_dim,
                                    wmma::mem_row_major);

    }

}

__global__ void spmm_forward_cuda_kernel_arbi_warps_hybrid_32(
        const int * __restrict__ nodePointer,		// node pointer.
        const int *__restrict__ edgeList,			// edge list.
        const int *__restrict__ blockPartition, 	// number of TC_blocks (16x8) in each row_window.
        const int *__restrict__ edgeToColumn, 		// eid -> col within each row_window.
        const int *__restrict__ edgeToRow, 		    // eid -> col within each row_window.
        const int numNodes,
        const int numEdges,
        const int embedding_dim,				    // embedding dimension.
        const float *__restrict__ input,		    // input feature matrix.
        float *output,							    // aggreAGNNed output feature matrix.
        const int *__restrict__ hybrid_type,
        const int *__restrict__ row_nzr,
        const int *__restrict__ col_nzr
) {
    unsigned bid = blockIdx.x;								// block_index == row_window_index
    unsigned wid = threadIdx.y;								// warp_index handling multi-dimension > 16.
    const unsigned laneid = threadIdx.x;							// lanid of each warp.
    const unsigned tid = threadIdx.y * blockDim.x + laneid;			// threadid of each block.
    // const unsigned warpSize = blockDim.x;							// number of threads per warp.
    const unsigned threadPerBlock = blockDim.x * blockDim.y;		// number of threads per block.

    const unsigned dimTileNum = embedding_dim / BLK_H;              // number of tiles along the dimension
    unsigned nIdx_start = bid * BLK_H;					    // starting nodeIdx of current row_window.
    unsigned nIdx_end = min((bid + 1) * BLK_H, numNodes);		// ending nodeIdx of current row_window.

    unsigned eIdx_start = nodePointer[nIdx_start];			// starting edgeIdx of current row_window.
    unsigned eIdx_end = nodePointer[nIdx_end];				// ending edgeIdx of the current row_window.
    unsigned num_TC_blocks = blockPartition[bid]; 			// number of TC_blocks of the current row_window.
    const unsigned dense_bound = numNodes * embedding_dim;

    __shared__ float sparse_A[BLK_H * BLK_W * MAX_BLK];					// row-major sparse matrix shared memory store.
    __shared__ int sparse_AToX_index[BLK_W * MAX_BLK];					// TC_block col to dense_tile row.

    extern __shared__ float dense_X[];

    __shared__ int tmp_A[S_SIZE];

    if(hybrid_type[bid] == 0){

        // int end_nzr = row_nzr[bid + 1];
        int end_nzr = (bid + 1) * BLK_H > numNodes ? numNodes : (bid + 1) * BLK_H;

        unsigned begin_edge = nodePointer[bid * 16], end_edge = nodePointer[min((bid + 1) * 16, numNodes)];
        for(int i = begin_edge + tid; i < end_edge; i += threadPerBlock){
            tmp_A[i - begin_edge] = edgeList[i];
        }
        __syncthreads();

        // for (int z = row_nzr[bid] + wid; z < end_nzr; z += WPB) {
        for (int z = bid * BLK_H + wid; z < end_nzr; z += WPB) {
            // int row = col_nzr[z];
            int row = z;
            int target_id = row * embedding_dim + laneid;
            int begin_col_id = nodePointer[row], end_col_id = nodePointer[row + 1];
            float acc = 0.0;
            for (int j = begin_col_id; j < end_col_id; j++) {
                // acc += input[laneid + edgeList[j] * embedding_dim];
                acc += input[laneid + tmp_A[j - begin_edge] * embedding_dim];
            }
            output[target_id] = acc;
        }
    }

    else{
        wmma::fragment<wmma::matrix_a, BLK_H, BLK_H, BLK_W, wmma::precision::tf32, wmma::row_major> a_frag;
        wmma::fragment<wmma::matrix_b, BLK_H, BLK_H, BLK_W, wmma::precision::tf32, wmma::col_major> b_frag;
        wmma::fragment<wmma::accumulator, BLK_H, BLK_H, BLK_W, float> acc_frag;
        wmma::fill_fragment(acc_frag, 0.0f);

        // nIdx_start = bid * BLK_H;					    // starting nodeIdx of current row_window.
        // nIdx_end = min((bid + 1) * BLK_H, numNodes);		// ending nodeIdx of current row_window.

        // eIdx_start = nodePointer[nIdx_start];			// starting edgeIdx of current row_window.
        // eIdx_end = nodePointer[nIdx_end];				// ending edgeIdx of the current row_window.
        // num_TC_blocks = blockPartition[bid];

        // Init A_colToX_row with dummy values.
        if (tid < BLK_W * MAX_BLK) {
            sparse_AToX_index[tid] = numNodes + 1;
        }

        __syncthreads();
        
        // Init sparse_A with zero values.
#pragma unroll
        for (unsigned idx = tid; idx < BLK_W * BLK_H * MAX_BLK; idx += threadPerBlock) {
            sparse_A[idx] = 0;
        }

        // #pragma unroll
        // // Init dense_X with zero values.
        // for (unsigned idx = tid; idx < dimTileNum * BLK_W * BLK_H; idx += threadPerBlock) {
        //     dense_X[idx] = 0;
        // }

#pragma unroll
        for (unsigned eIdx = eIdx_start + tid; eIdx < eIdx_end; eIdx += threadPerBlock) {
            unsigned col = edgeToColumn[eIdx];
            unsigned row_local = edgeToRow[eIdx] % BLK_H;
            unsigned blk_id = col / 8;
            unsigned col_local = col % 8;
            sparse_A[row_local * BLK_W + col_local + blk_id * BLK_H * BLK_W] = 1;        // set the edge of the sparse_A.
            sparse_AToX_index[col] = edgeList[eIdx];        // record the mapping from sparse_A colId to rowId of dense_X.
        }

        __syncthreads();

        for (unsigned i = 0; i < num_TC_blocks; i++) {
//             if (wid < dimTileNum)
// #pragma unroll
//                 for (unsigned idx = laneid; idx < BLK_W * BLK_H; idx += warpSize) {
//                     unsigned dense_rowIdx = sparse_AToX_index[idx % BLK_W + i * BLK_W];                        // TC_block_col to dense_tile_row.
//                     unsigned dense_dimIdx = idx / BLK_W;                                        // dimIndex of the dense tile.
//                     unsigned source_idx = dense_rowIdx * *embedding_dim + wid * BLK_H + dense_dimIdx;
//                     unsigned target_idx = wid * BLK_W * BLK_H + idx;
//                     // boundary test.
//                     dense_X[target_idx] = source_idx < dense_bound ? input[source_idx] : 0;
//                 }
#pragma unroll
                for (unsigned idx = wid; idx < BLK_W; idx += WPB) {
                    unsigned dense_rowIdx = sparse_AToX_index[idx % BLK_W + i * BLK_W];                        // TC_block_col to dense_tile_row.
                    unsigned source_idx = dense_rowIdx * embedding_dim + laneid;
                    unsigned target_idx = laneid * BLK_W + idx;
                    // boundary test.
                    dense_X[target_idx] = source_idx < dense_bound ? __ldca(&input[source_idx]) : 0;
                }

            __syncthreads();

            if (wid < dimTileNum) {
                wmma::load_matrix_sync(a_frag, sparse_A + i * BLK_W * BLK_H, BLK_W);
                wmma::load_matrix_sync(b_frag, dense_X + wid * BLK_W * BLK_H, BLK_W);
#pragma unroll
                for (unsigned t = 0; t < a_frag.num_elements; t++) {
                    a_frag.x[t] = wmma::__float_to_tf32(a_frag.x[t]);
                }

#pragma unroll
                for (unsigned t = 0; t < b_frag.num_elements; t++) {
                    b_frag.x[t] = wmma::__float_to_tf32(b_frag.x[t]);
                }
                // Perform the matrix multiplication.
                wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
            }

        }
        if (wid < dimTileNum)
            // Store the matrix to output matrix.
            // * Note * embeeding dimension should be padded divisible by BLK_H for output correctness.
            wmma::store_matrix_sync(output + bid * BLK_H * embedding_dim + wid * BLK_H, acc_frag, embedding_dim,
                                    wmma::mem_row_major);

    }

}

__global__ void spmm_forward_cuda_kernel_arbi_warps_hybrid_64(
        const int * __restrict__ nodePointer,		// node pointer.
        const int *__restrict__ edgeList,			// edge list.
        const int *__restrict__ blockPartition, 	// number of TC_blocks (16x8) in each row_window.
        const int *__restrict__ edgeToColumn, 		// eid -> col within each row_window.
        const int *__restrict__ edgeToRow, 		    // eid -> col within each row_window.
        const int numNodes,
        const int numEdges,
        const int embedding_dim,				    // embedding dimension.
        const float *__restrict__ input,		    // input feature matrix.
        float *output,							    // aggreAGNNed output feature matrix.
        const int *__restrict__ hybrid_type,
        const int *__restrict__ row_nzr,
        const int *__restrict__ col_nzr
) {
    unsigned bid = blockIdx.x;								// block_index == row_window_index
    unsigned wid = threadIdx.y;								// warp_index handling multi-dimension > 16.
    const unsigned laneid = threadIdx.x;							// lanid of each warp.
    const unsigned tid = threadIdx.y * blockDim.x + laneid;			// threadid of each block.
    // const unsigned warpSize = blockDim.x;							// number of threads per warp.
    const unsigned threadPerBlock = blockDim.x * blockDim.y;		// number of threads per block.

    const unsigned dimTileNum = embedding_dim / BLK_H;              // number of tiles along the dimension
    unsigned nIdx_start = bid * BLK_H;					    // starting nodeIdx of current row_window.
    unsigned nIdx_end = min((bid + 1) * BLK_H, numNodes);		// ending nodeIdx of current row_window.

    unsigned eIdx_start = nodePointer[nIdx_start];			// starting edgeIdx of current row_window.
    unsigned eIdx_end = nodePointer[nIdx_end];				// ending edgeIdx of the current row_window.
    unsigned num_TC_blocks = blockPartition[bid]; 			// number of TC_blocks of the current row_window.
    const unsigned dense_bound = numNodes * embedding_dim;

    __shared__ float sparse_A[BLK_H * BLK_W * MAX_BLK];					// row-major sparse matrix shared memory store.
    __shared__ int sparse_AToX_index[BLK_W * MAX_BLK];					// TC_block col to dense_tile row.

    extern __shared__ float dense_X[];

    __shared__ int tmp_A[S_SIZE];

    if(hybrid_type[bid] == 0){

        // int end_nzr = row_nzr[bid + 1];
        int end_nzr = (bid + 1) * BLK_H > numNodes ? numNodes : (bid + 1) * BLK_H;

        unsigned begin_edge = nodePointer[bid * 16], end_edge = nodePointer[min((bid + 1) * 16, numNodes)];
        for(int i = begin_edge + tid; i < end_edge; i += threadPerBlock){
            tmp_A[i - begin_edge] = edgeList[i];
        }
        __syncthreads();

        // for (int z = row_nzr[bid] + wid; z < end_nzr; z += WPB) {
        for (int z = bid * BLK_H + wid; z < end_nzr; z += WPB) {
            // int row = col_nzr[z];
            int row = z;
            int begin_col_id = nodePointer[row], end_col_id = nodePointer[row + 1];
            for(int h = 0; h < 2; h++){
                int target_id = row * embedding_dim + laneid + h * 32;
                float acc = 0.0;
                for (int j = begin_col_id; j < end_col_id; j++) {
                    // acc += input[laneid + edgeList[j] * embedding_dim];
                    acc += input[laneid + h * 32 + tmp_A[j - begin_edge] * embedding_dim];
                }
                output[target_id] = acc;
            }
        }
    }

    else{
        wmma::fragment<wmma::matrix_a, BLK_H, BLK_H, BLK_W, wmma::precision::tf32, wmma::row_major> a_frag;
        wmma::fragment<wmma::matrix_b, BLK_H, BLK_H, BLK_W, wmma::precision::tf32, wmma::col_major> b_frag;
        wmma::fragment<wmma::accumulator, BLK_H, BLK_H, BLK_W, float> acc_frag;
        wmma::fill_fragment(acc_frag, 0.0f);

        // nIdx_start = bid * BLK_H;					    // starting nodeIdx of current row_window.
        // nIdx_end = min((bid + 1) * BLK_H, numNodes);		// ending nodeIdx of current row_window.

        // eIdx_start = nodePointer[nIdx_start];			// starting edgeIdx of current row_window.
        // eIdx_end = nodePointer[nIdx_end];				// ending edgeIdx of the current row_window.
        // num_TC_blocks = blockPartition[bid];

        // Init A_colToX_row with dummy values.
        if (tid < BLK_W * MAX_BLK) {
            sparse_AToX_index[tid] = numNodes + 1;
        }

        __syncthreads();
        
        // Init sparse_A with zero values.
#pragma unroll
        for (unsigned idx = tid; idx < BLK_W * BLK_H * MAX_BLK; idx += threadPerBlock) {
            sparse_A[idx] = 0;
        }

        // #pragma unroll
        // // Init dense_X with zero values.
        // for (unsigned idx = tid; idx < dimTileNum * BLK_W * BLK_H; idx += threadPerBlock) {
        //     dense_X[idx] = 0;
        // }

#pragma unroll
        for (unsigned eIdx = eIdx_start + tid; eIdx < eIdx_end; eIdx += threadPerBlock) {
            unsigned col = edgeToColumn[eIdx];
            unsigned row_local = edgeToRow[eIdx] % BLK_H;
            unsigned blk_id = col / 8;
            unsigned col_local = col % 8;
            sparse_A[row_local * BLK_W + col_local + blk_id * BLK_H * BLK_W] = 1;        // set the edge of the sparse_A.
            sparse_AToX_index[col] = edgeList[eIdx];        // record the mapping from sparse_A colId to rowId of dense_X.
        }

        __syncthreads();

        for (unsigned i = 0; i < num_TC_blocks; i++) {
//             if (wid < dimTileNum)
// #pragma unroll
//                 for (unsigned idx = laneid; idx < BLK_W * BLK_H; idx += warpSize) {
//                     unsigned dense_rowIdx = sparse_AToX_index[idx % BLK_W + i * BLK_W];                        // TC_block_col to dense_tile_row.
//                     unsigned dense_dimIdx = idx / BLK_W;                                        // dimIndex of the dense tile.
//                     unsigned source_idx = dense_rowIdx * *embedding_dim + wid * BLK_H + dense_dimIdx;
//                     unsigned target_idx = wid * BLK_W * BLK_H + idx;
//                     // boundary test.
//                     dense_X[target_idx] = source_idx < dense_bound ? input[source_idx] : 0;
//                 }
#pragma unroll
                for (unsigned idx = wid; idx < BLK_W; idx += WPB) {
                    unsigned dense_rowIdx = sparse_AToX_index[idx % BLK_W + i * BLK_W];                        // TC_block_col to dense_tile_row.
                    for(int h = 0; h < 2; h++){
                        unsigned source_idx = dense_rowIdx * embedding_dim + laneid + h * 32;
                        unsigned target_idx = (laneid + h * 32) * BLK_W + idx;
                        // boundary test.
                        dense_X[target_idx] = source_idx < dense_bound ? __ldca(&input[source_idx]) : 0;
                    }
                }

            __syncthreads();

            if (wid < dimTileNum) {
                wmma::load_matrix_sync(a_frag, sparse_A + i * BLK_W * BLK_H, BLK_W);
                wmma::load_matrix_sync(b_frag, dense_X + wid * BLK_W * BLK_H, BLK_W);
#pragma unroll
                for (unsigned t = 0; t < a_frag.num_elements; t++) {
                    a_frag.x[t] = wmma::__float_to_tf32(a_frag.x[t]);
                }

#pragma unroll
                for (unsigned t = 0; t < b_frag.num_elements; t++) {
                    b_frag.x[t] = wmma::__float_to_tf32(b_frag.x[t]);
                }
                // Perform the matrix multiplication.
                wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
            }

        }
        if (wid < dimTileNum)
            // Store the matrix to output matrix.
            // * Note * embeeding dimension should be padded divisible by BLK_H for output correctness.
            wmma::store_matrix_sync(output + bid * BLK_H * embedding_dim + wid * BLK_H, acc_frag, embedding_dim,
                                    wmma::mem_row_major);

    }

}

__global__ void spmm_forward_cuda_kernel_arbi_warps_hybrid_32_fused(
        const int * __restrict__ nodePointer,		// node pointer.
        const int *__restrict__ edgeList,			// edge list.
        const int *__restrict__ blockPartition, 	// number of TC_blocks (16x8) in each row_window.
        const int *__restrict__ edgeToColumn, 		// eid -> col within each row_window.
        const int *__restrict__ edgeToRow, 		    // eid -> col within each row_window.
        const int numNodes,
        const int numEdges,
        const int embedding_dim,				    // embedding dimension.
        const int hidden_dim,
        const float *__restrict__ input,		    // input feature matrix.
        float *output,
        float *output2,							    // aggreAGNNed output feature matrix.
        const int *__restrict__ hybrid_type,
        const int *__restrict__ row_nzr,
        const int *__restrict__ col_nzr,
        const float *__restrict__ weights
) {
    unsigned bid = blockIdx.x;								// block_index == row_window_index
    unsigned wid = threadIdx.y;								// warp_index handling multi-dimension > 16.
    const unsigned laneid = threadIdx.x;							// lanid of each warp.
    const unsigned tid = threadIdx.y * blockDim.x + laneid;			// threadid of each block.
    // const unsigned warpSize = blockDim.x;							// number of threads per warp.
    const unsigned threadPerBlock = blockDim.x * blockDim.y;		// number of threads per block.

    const unsigned dimTileNum = embedding_dim / BLK_H;              // number of tiles along the dimension
    const unsigned windows_size = embedding_dim / BLK_W;              // number of tiles along the dimension
    unsigned nIdx_start = bid * BLK_H;					    // starting nodeIdx of current row_window.
    unsigned nIdx_end = min((bid + 1) * BLK_H, numNodes);		// ending nodeIdx of current row_window.

    unsigned eIdx_start = nodePointer[nIdx_start];			// starting edgeIdx of current row_window.
    unsigned eIdx_end = nodePointer[nIdx_end];				// ending edgeIdx of the current row_window.
    unsigned num_TC_blocks = blockPartition[bid]; 			// number of TC_blocks of the current row_window.
    const unsigned dense_bound = numNodes * embedding_dim;

    __shared__ float sparse_A[BLK_H * BLK_W * MAX_BLK];					// row-major sparse matrix shared memory store.
    __shared__ int sparse_AToX_index[BLK_W * MAX_BLK];					// TC_block col to dense_tile row.

    extern __shared__ float dense_X[];

    __shared__ int tmp_A[S_SIZE];
    // printf("%.1f\n", output[0]);
    wmma::fragment<wmma::matrix_a, BLK_H, BLK_H, BLK_W, wmma::precision::tf32, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, BLK_H, BLK_H, BLK_W, wmma::precision::tf32, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, BLK_H, BLK_H, BLK_W, float> acc_frag;
    // if(threadIdx.y == 0 && threadIdx.x == 0) printf("%d\n", blockIdx.x);
    if(hybrid_type[bid] == 0){
        // int end_nzr = row_nzr[bid + 1];
        int end_nzr = (bid + 1) * BLK_H > numNodes ? numNodes : (bid + 1) * BLK_H;
        
        unsigned begin_edge = nodePointer[bid * 16], end_edge = nodePointer[min((bid + 1) * 16, numNodes)];
        for(int i = begin_edge + tid; i < end_edge; i += threadPerBlock){
            tmp_A[i - begin_edge] = edgeList[i];
        }
        __syncthreads();

        // for (int z = row_nzr[bid] + wid; z < end_nzr; z += WPB) {
        for(int z = bid * BLK_H + wid; z < end_nzr; z += WPB) {
            // int row = col_nzr[z];
            int row = z;
            int target_id = (laneid / BLK_W) * BLK_H * BLK_W + (row % BLK_H) * BLK_W + laneid % BLK_W;
            int target_id2 = row * embedding_dim + laneid;
            int begin_col_id = nodePointer[row], end_col_id = nodePointer[row + 1];
            float acc = 0.0;
            for (int j = begin_col_id; j < end_col_id; j++) {
                // acc += input[laneid + edgeList[j] * embedding_dim];
                acc += input[laneid + tmp_A[j - begin_edge] * embedding_dim];
            }
            dense_X[target_id] = acc;
            output2[target_id2] = acc;
        }
        // if(blockIdx.x == 0 && threadIdx.y == 0 && threadIdx.x == 0) printf("%.1f %.1f %.1f %.1f\n", input[0], input[1],input[2],input[3]);
    }

    else{
        wmma::fill_fragment(acc_frag, 0.0f);

        nIdx_start = bid * BLK_H;					    // starting nodeIdx of current row_window.
        nIdx_end = min((bid + 1) * BLK_H, numNodes);		// ending nodeIdx of current row_window.

        eIdx_start = nodePointer[nIdx_start];			// starting edgeIdx of current row_window.
        eIdx_end = nodePointer[nIdx_end];				// ending edgeIdx of the current row_window.
        num_TC_blocks = blockPartition[bid];

        // Init A_colToX_row with dummy values.
        if (tid < BLK_W * MAX_BLK) {
            sparse_AToX_index[tid] = numNodes + 1;
        }

        __syncthreads();
        
        // Init sparse_A with zero values.
#pragma unroll
        for (unsigned idx = tid; idx < BLK_W * BLK_H * MAX_BLK; idx += threadPerBlock) {
            sparse_A[idx] = 0;
        }

        #pragma unroll
        // Init dense_X with zero values.
        for (unsigned idx = tid; idx < dimTileNum * BLK_W * BLK_H; idx += threadPerBlock) {
            dense_X[idx] = 0;
        }

#pragma unroll
        for (unsigned eIdx = eIdx_start + tid; eIdx < eIdx_end; eIdx += threadPerBlock) {
            unsigned col = edgeToColumn[eIdx];
            unsigned row_local = edgeToRow[eIdx] % BLK_H;
            unsigned blk_id = col / 8;
            unsigned col_local = col % 8;
            sparse_A[row_local * BLK_W + col_local + blk_id * BLK_H * BLK_W] = 1;        // set the edge of the sparse_A.
            sparse_AToX_index[col] = edgeList[eIdx];        // record the mapping from sparse_A colId to rowId of dense_X.
        }

        __syncthreads();

        for (unsigned i = 0; i < num_TC_blocks; i++) {
//             if (wid < dimTileNum)
// #pragma unroll
//                 for (unsigned idx = laneid; idx < BLK_W * BLK_H; idx += warpSize) {
//                     unsigned dense_rowIdx = sparse_AToX_index[idx % BLK_W + i * BLK_W];                        // TC_block_col to dense_tile_row.
//                     unsigned dense_dimIdx = idx / BLK_W;                                        // dimIndex of the dense tile.
//                     unsigned source_idx = dense_rowIdx * *embedding_dim + wid * BLK_H + dense_dimIdx;
//                     unsigned target_idx = wid * BLK_W * BLK_H + idx;
//                     // boundary test.
//                     dense_X[target_idx] = source_idx < dense_bound ? input[source_idx] : 0;
//                 }
#pragma unroll
                for (unsigned idx = wid; idx < BLK_W; idx += WPB) {
                    unsigned dense_rowIdx = sparse_AToX_index[idx % BLK_W + i * BLK_W];                        // TC_block_col to dense_tile_row.
                    unsigned source_idx = dense_rowIdx * embedding_dim + laneid;
                    unsigned target_idx = laneid * BLK_W + idx;
                    // boundary test.
                    dense_X[target_idx] = source_idx < dense_bound ? __ldca(&input[source_idx]) : 0;
                }

            __syncthreads();

            if (wid < dimTileNum) {
                wmma::load_matrix_sync(a_frag, sparse_A + i * BLK_W * BLK_H, BLK_W);
                wmma::load_matrix_sync(b_frag, dense_X + wid * BLK_W * BLK_H, BLK_W);
#pragma unroll
                for (unsigned t = 0; t < a_frag.num_elements; t++) {
                    a_frag.x[t] = wmma::__float_to_tf32(a_frag.x[t]);
                }

#pragma unroll
                for (unsigned t = 0; t < b_frag.num_elements; t++) {
                    b_frag.x[t] = wmma::__float_to_tf32(b_frag.x[t]);
                }
                // Perform the matrix multiplication.
                wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
            }

        }
        if (wid < dimTileNum){
            // Store the matrix to output matrix.
            // * Note * embeeding dimension should be padded divisible by BLK_H for output correctness.
            wmma::store_matrix_sync(output2 + bid * BLK_H * embedding_dim + wid * BLK_H, acc_frag, embedding_dim, wmma::mem_row_major);
            dense_X[wid * BLK_H * BLK_W + laneid % 4 + (laneid / 4) * BLK_W] = acc_frag.x[0];
            dense_X[wid * BLK_H * BLK_W + 4 + laneid % 4 + (laneid / 4) * BLK_W] = acc_frag.x[1];
            dense_X[wid * BLK_H * BLK_W + laneid % 4 + (laneid / 4 + 8) * BLK_W] = acc_frag.x[2];
            dense_X[wid * BLK_H * BLK_W + 4 + laneid % 4 + (laneid / 4 + 8) * BLK_W] = acc_frag.x[3];
            dense_X[wid * BLK_H * BLK_W + 8 + laneid % 4 + (laneid / 4) * BLK_W] = acc_frag.x[4];
            dense_X[wid * BLK_H * BLK_W + 12 + laneid % 4 + (laneid / 4) * BLK_W] = acc_frag.x[5];
            dense_X[wid * BLK_H * BLK_W + 8 + laneid % 4 + (laneid / 4 + 8) * BLK_W] = acc_frag.x[6];
            dense_X[wid * BLK_H * BLK_W + 12 + laneid % 4 + (laneid / 4 + 8) * BLK_W] = acc_frag.x[7];
        }
    }
    wmma::fill_fragment(acc_frag, 0.0f);
    
    for(int i = 0; i < windows_size; i++){

        for(int j = wid; j < BLK_W; j += WPB){
            unsigned source_idx = (i * BLK_W + j) * hidden_dim + laneid;
            unsigned target_idx = j + laneid * BLK_W;
            
            // boundary test.
            sparse_A[target_idx] = weights[source_idx];
        }
        __syncthreads();

        if (wid < dimTileNum) {
            wmma::load_matrix_sync(a_frag, dense_X + i * BLK_W * BLK_H, BLK_W);
            wmma::load_matrix_sync(b_frag, sparse_A + wid * BLK_W * BLK_H, BLK_W);
            #pragma unroll
            for (unsigned t = 0; t < a_frag.num_elements; t++) {
                a_frag.x[t] = wmma::__float_to_tf32(a_frag.x[t]);
            }
            #pragma unroll
            for (unsigned t = 0; t < b_frag.num_elements; t++) {
                b_frag.x[t] = wmma::__float_to_tf32(b_frag.x[t]);
            }
            // Perform the matrix multiplication.
            wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
        }
    }
    if (wid < dimTileNum)
        // if(threadIdx.x == 0) printf("%.1f\n", output[0]);
        wmma::store_matrix_sync(output + bid * BLK_H * hidden_dim + wid * BLK_H, acc_frag, hidden_dim, wmma::mem_row_major);
        // wmma::store_matrix_sync(output + bid * BLK_H * hidden_dim + wid * BLK_H, acc_frag, hidden_dim, wmma::mem_row_major);
    // if(threadIdx.y == 0 && threadIdx.x == 0) printf("%.1f\n", b_frag.x[0]);
    // if(blockIdx.x == 0 && threadIdx.y == 0 && threadIdx.x == 0) printf("%.1f %.1f %.1f %.1f\n", acc_frag.x[0],acc_frag.x[1],acc_frag.x[2],acc_frag.x[3]);
    // if(blockIdx.x == 0 && threadIdx.y ==0 && threadIdx.x == 0){
    //     printf("AAAA\n");
    //     for(int i = 0; i < 32; i++){
    //         printf("%.1f, ", output[i]);
    //     }
    //     printf("\n");
    // }
}

__global__ void spmm_forward_cuda_kernel_arbi_warps_hybrid_64_fused(
        const int * __restrict__ nodePointer,		// node pointer.
        const int *__restrict__ edgeList,			// edge list.
        const int *__restrict__ blockPartition, 	// number of TC_blocks (16x8) in each row_window.
        const int *__restrict__ edgeToColumn, 		// eid -> col within each row_window.
        const int *__restrict__ edgeToRow, 		    // eid -> col within each row_window.
        const int numNodes,
        const int numEdges,
        const int embedding_dim,				    // embedding dimension.
        const int hidden_dim,
        const float *__restrict__ input,		    // input feature matrix.
        float *output,
        float *output2,							    // aggreAGNNed output feature matrix.
        const int *__restrict__ hybrid_type,
        const int *__restrict__ row_nzr,
        const int *__restrict__ col_nzr,
        const float *__restrict__ weights
) {
    unsigned bid = blockIdx.x;								// block_index == row_window_index
    unsigned wid = threadIdx.y;								// warp_index handling multi-dimension > 16.
    const unsigned laneid = threadIdx.x;							// lanid of each warp.
    const unsigned tid = threadIdx.y * blockDim.x + laneid;			// threadid of each block.
    // const unsigned warpSize = blockDim.x;							// number of threads per warp.
    const unsigned threadPerBlock = blockDim.x * blockDim.y;		// number of threads per block.

    const unsigned dimTileNum = embedding_dim / BLK_H;              // number of tiles along the dimension
    const unsigned windows_size = embedding_dim / BLK_W;              // number of tiles along the dimension
    unsigned nIdx_start = bid * BLK_H;					    // starting nodeIdx of current row_window.
    unsigned nIdx_end = min((bid + 1) * BLK_H, numNodes);		// ending nodeIdx of current row_window.

    unsigned eIdx_start = nodePointer[nIdx_start];			// starting edgeIdx of current row_window.
    unsigned eIdx_end = nodePointer[nIdx_end];				// ending edgeIdx of the current row_window.
    unsigned num_TC_blocks = blockPartition[bid]; 			// number of TC_blocks of the current row_window.
    const unsigned dense_bound = numNodes * embedding_dim;

    __shared__ float sparse_A[BLK_H * BLK_W * MAX_BLK];					// row-major sparse matrix shared memory store.
    __shared__ int sparse_AToX_index[BLK_W * MAX_BLK];					// TC_block col to dense_tile row.

    extern __shared__ float dense_X[];

    __shared__ int tmp_A[S_SIZE];
    // printf("%.1f\n", output[0]);
    wmma::fragment<wmma::matrix_a, BLK_H, BLK_H, BLK_W, wmma::precision::tf32, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, BLK_H, BLK_H, BLK_W, wmma::precision::tf32, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, BLK_H, BLK_H, BLK_W, float> acc_frag;
    // if(threadIdx.y == 0 && threadIdx.x == 0) printf("%d\n", blockIdx.x);
    if(hybrid_type[bid] == 0){
        // int end_nzr = row_nzr[bid + 1];
        int end_nzr = (bid + 1) * BLK_H > numNodes ? numNodes : (bid + 1) * BLK_H;
        
        unsigned begin_edge = nodePointer[bid * 16], end_edge = nodePointer[min((bid + 1) * 16, numNodes)];
        for(int i = begin_edge + tid; i < end_edge; i += threadPerBlock){
            tmp_A[i - begin_edge] = edgeList[i];
        }
        __syncthreads();

        // for (int z = row_nzr[bid] + wid; z < end_nzr; z += WPB) {
        for(int z = bid * BLK_H + wid; z < end_nzr; z += WPB) {
            // int row = col_nzr[z];
            int row = z;
            int begin_col_id = nodePointer[row], end_col_id = nodePointer[row + 1];
            for(int h = 0; h < 2; h++){
                int target_id = ((laneid + h * 32) / BLK_W) * BLK_H * BLK_W + (row % BLK_H) * BLK_W + (laneid + h * 32) % BLK_W;
                int target_id2 = row * embedding_dim + laneid + h * 32;
                float acc = 0.0;
                for (int j = begin_col_id; j < end_col_id; j++) {
                    // acc += input[laneid + edgeList[j] * embedding_dim];
                    acc += input[laneid + h * 32 + tmp_A[j - begin_edge] * embedding_dim];
                }
                dense_X[target_id] = acc;
                output2[target_id2] = acc;
            }
        }
        // if(blockIdx.x == 0 && threadIdx.y == 0 && threadIdx.x == 0) printf("%.1f %.1f %.1f %.1f\n", input[0], input[1],input[2],input[3]);
    }

    else{
        wmma::fill_fragment(acc_frag, 0.0f);

        nIdx_start = bid * BLK_H;					    // starting nodeIdx of current row_window.
        nIdx_end = min((bid + 1) * BLK_H, numNodes);		// ending nodeIdx of current row_window.

        eIdx_start = nodePointer[nIdx_start];			// starting edgeIdx of current row_window.
        eIdx_end = nodePointer[nIdx_end];				// ending edgeIdx of the current row_window.
        num_TC_blocks = blockPartition[bid];

        // Init A_colToX_row with dummy values.
        if (tid < BLK_W * MAX_BLK) {
            sparse_AToX_index[tid] = numNodes + 1;
        }

        __syncthreads();
        
        // Init sparse_A with zero values.
#pragma unroll
        for (unsigned idx = tid; idx < BLK_W * BLK_H * MAX_BLK; idx += threadPerBlock) {
            sparse_A[idx] = 0;
        }

        #pragma unroll
        // Init dense_X with zero values.
        for (unsigned idx = tid; idx < dimTileNum * BLK_W * BLK_H; idx += threadPerBlock) {
            dense_X[idx] = 0;
        }

#pragma unroll
        for (unsigned eIdx = eIdx_start + tid; eIdx < eIdx_end; eIdx += threadPerBlock) {
            unsigned col = edgeToColumn[eIdx];
            unsigned row_local = edgeToRow[eIdx] % BLK_H;
            unsigned blk_id = col / 8;
            unsigned col_local = col % 8;
            sparse_A[row_local * BLK_W + col_local + blk_id * BLK_H * BLK_W] = 1;        // set the edge of the sparse_A.
            sparse_AToX_index[col] = edgeList[eIdx];        // record the mapping from sparse_A colId to rowId of dense_X.
        }

        __syncthreads();

        for (unsigned i = 0; i < num_TC_blocks; i++) {
//             if (wid < dimTileNum)
// #pragma unroll
//                 for (unsigned idx = laneid; idx < BLK_W * BLK_H; idx += warpSize) {
//                     unsigned dense_rowIdx = sparse_AToX_index[idx % BLK_W + i * BLK_W];                        // TC_block_col to dense_tile_row.
//                     unsigned dense_dimIdx = idx / BLK_W;                                        // dimIndex of the dense tile.
//                     unsigned source_idx = dense_rowIdx * *embedding_dim + wid * BLK_H + dense_dimIdx;
//                     unsigned target_idx = wid * BLK_W * BLK_H + idx;
//                     // boundary test.
//                     dense_X[target_idx] = source_idx < dense_bound ? input[source_idx] : 0;
//                 }
#pragma unroll
                for (unsigned idx = wid; idx < BLK_W; idx += WPB) {
                    for(int h = 0; h < 2; h++){
                        unsigned dense_rowIdx = sparse_AToX_index[idx % BLK_W + i * BLK_W];                        // TC_block_col to dense_tile_row.
                        unsigned source_idx = dense_rowIdx * embedding_dim + laneid + h * 32;
                        unsigned target_idx = (laneid + h * 32) * BLK_W + idx;
                        // boundary test.
                        dense_X[target_idx] = source_idx < dense_bound ? __ldca(&input[source_idx]) : 0;
                    }
                }

            __syncthreads();

            if (wid < dimTileNum) {
                wmma::load_matrix_sync(a_frag, sparse_A + i * BLK_W * BLK_H, BLK_W);
                wmma::load_matrix_sync(b_frag, dense_X + wid * BLK_W * BLK_H, BLK_W);
#pragma unroll
                for (unsigned t = 0; t < a_frag.num_elements; t++) {
                    a_frag.x[t] = wmma::__float_to_tf32(a_frag.x[t]);
                }

#pragma unroll
                for (unsigned t = 0; t < b_frag.num_elements; t++) {
                    b_frag.x[t] = wmma::__float_to_tf32(b_frag.x[t]);
                }
                // Perform the matrix multiplication.
                wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
            }

        }
        if (wid < dimTileNum){
            // Store the matrix to output matrix.
            // * Note * embeeding dimension should be padded divisible by BLK_H for output correctness.
            wmma::store_matrix_sync(output2 + bid * BLK_H * embedding_dim + wid * BLK_H, acc_frag, embedding_dim, wmma::mem_row_major);
            dense_X[wid * BLK_H * BLK_W + laneid % 4 + (laneid / 4) * BLK_W] = acc_frag.x[0];
            dense_X[wid * BLK_H * BLK_W + 4 + laneid % 4 + (laneid / 4) * BLK_W] = acc_frag.x[1];
            dense_X[wid * BLK_H * BLK_W + laneid % 4 + (laneid / 4 + 8) * BLK_W] = acc_frag.x[2];
            dense_X[wid * BLK_H * BLK_W + 4 + laneid % 4 + (laneid / 4 + 8) * BLK_W] = acc_frag.x[3];
            dense_X[wid * BLK_H * BLK_W + 8 + laneid % 4 + (laneid / 4) * BLK_W] = acc_frag.x[4];
            dense_X[wid * BLK_H * BLK_W + 12 + laneid % 4 + (laneid / 4) * BLK_W] = acc_frag.x[5];
            dense_X[wid * BLK_H * BLK_W + 8 + laneid % 4 + (laneid / 4 + 8) * BLK_W] = acc_frag.x[6];
            dense_X[wid * BLK_H * BLK_W + 12 + laneid % 4 + (laneid / 4 + 8) * BLK_W] = acc_frag.x[7];
        }
    }
    wmma::fill_fragment(acc_frag, 0.0f);
    
    for(int i = 0; i < windows_size; i++){
        // 加载weights到sparse_A中：8*hidden_embedding_dim
        for(int j = wid; j < BLK_W; j += WPB){
            for(int h = 0; h < 2; h++){
                unsigned source_idx = (i * BLK_W + j) * hidden_dim + laneid + h * 32;
                unsigned target_idx = j + (laneid + h * 32) * BLK_W;
                
                // boundary test.
                sparse_A[target_idx] = weights[source_idx];
            }
        }
        __syncthreads();

        if (wid < dimTileNum) {
            wmma::load_matrix_sync(a_frag, dense_X + i * BLK_W * BLK_H, BLK_W);
            wmma::load_matrix_sync(b_frag, sparse_A + wid * BLK_W * BLK_H, BLK_W);
            #pragma unroll
            for (unsigned t = 0; t < a_frag.num_elements; t++) {
                a_frag.x[t] = wmma::__float_to_tf32(a_frag.x[t]);
            }
            #pragma unroll
            for (unsigned t = 0; t < b_frag.num_elements; t++) {
                b_frag.x[t] = wmma::__float_to_tf32(b_frag.x[t]);
            }
            // Perform the matrix multiplication.
            wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
        }
    }
    if (wid < dimTileNum)
        // if(threadIdx.x == 0) printf("%.1f\n", output[0]);
        wmma::store_matrix_sync(output + bid * BLK_H * hidden_dim + wid * BLK_H, acc_frag, hidden_dim, wmma::mem_row_major);
        // wmma::store_matrix_sync(output + bid * BLK_H * hidden_dim + wid * BLK_H, acc_frag, hidden_dim, wmma::mem_row_major);
    // if(threadIdx.y == 0 && threadIdx.x == 0) printf("%.1f\n", b_frag.x[0]);
    // if(blockIdx.x == 0 && threadIdx.y == 0 && threadIdx.x == 0) printf("%.1f %.1f %.1f %.1f\n", acc_frag.x[0],acc_frag.x[1],acc_frag.x[2],acc_frag.x[3]);
    // if(blockIdx.x == 0 && threadIdx.y ==0 && threadIdx.x == 0){
    //     printf("AAAA\n");
    //     for(int i = 0; i < 32; i++){
    //         printf("%.1f, ", output[i]);
    //     }
    //     printf("\n");
    // }
}

__global__ void spmm_forward_cuda_kernel_arbi_warps_hybrid_final_fused(
        const int * __restrict__ nodePointer,		// node pointer.
        const int *__restrict__ edgeList,			// edge list.
        const int *__restrict__ blockPartition, 	// number of TC_blocks (16x8) in each row_window.
        const int *__restrict__ edgeToColumn, 		// eid -> col within each row_window.
        const int *__restrict__ edgeToRow, 		    // eid -> col within each row_window.
        const int numNodes,
        const int numEdges,
        const int embedding_dim,				    // embedding dimension.
        const int hidden_dim,
        const float *__restrict__ input,		    // input feature matrix.
        float *output,
        float *output2,							    // aggreAGNNed output feature matrix.
        const int *__restrict__ hybrid_type,
        const int *__restrict__ row_nzr,
        const int *__restrict__ col_nzr,
        const float *__restrict__ weights
) {
    unsigned bid = blockIdx.x;								// block_index == row_window_index
    unsigned wid = threadIdx.y;								// warp_index handling multi-dimension > 16.
    const unsigned laneid = threadIdx.x;							// lanid of each warp.
    const unsigned tid = threadIdx.y * blockDim.x + laneid;			// threadid of each block.
    // const unsigned warpSize = blockDim.x;							// number of threads per warp.
    const unsigned threadPerBlock = blockDim.x * blockDim.y;		// number of threads per block.

    const unsigned dimTileNum = (embedding_dim + BLK_H - 1) / BLK_H;              // number of tiles along the dimension
    const unsigned windows_size = (embedding_dim + BLK_W - 1) / BLK_W;              // number of tiles along the dimension
    unsigned nIdx_start = bid * BLK_H;					    // starting nodeIdx of current row_window.
    unsigned nIdx_end = min((bid + 1) * BLK_H, numNodes);		// ending nodeIdx of current row_window.
    const unsigned shared_size = (embedding_dim + 31) / 32;
    unsigned eIdx_start = nodePointer[nIdx_start];			// starting edgeIdx of current row_window.
    unsigned eIdx_end = nodePointer[nIdx_end];				// ending edgeIdx of the current row_window.
    unsigned num_TC_blocks = blockPartition[bid]; 			// number of TC_blocks of the current row_window.
    const unsigned dense_bound = numNodes * embedding_dim;

    __shared__ float sparse_A[BLK_H * BLK_W * MAX_BLK];					// row-major sparse matrix shared memory store.
    __shared__ int sparse_AToX_index[BLK_W * MAX_BLK];					// TC_block col to dense_tile row.
    unsigned csr_full = embedding_dim / 32, csr_reserve = embedding_dim % 32;
    extern __shared__ float dense_X[];
    __shared__ int tmp_A[S_SIZE];
    // printf("%.1f\n", output[0]);
    wmma::fragment<wmma::matrix_a, BLK_H, BLK_H, BLK_W, wmma::precision::tf32, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, BLK_H, BLK_H, BLK_W, wmma::precision::tf32, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, BLK_H, BLK_H, BLK_W, float> acc_frag;
    // if(threadIdx.y == 0 && threadIdx.x == 0) printf("%d\n", blockIdx.x);
    if(hybrid_type[bid] == 0){
        // int end_nzr = row_nzr[bid + 1];
        int end_nzr = (bid + 1) * BLK_H > numNodes ? numNodes : (bid + 1) * BLK_H;

        unsigned begin_edge = nodePointer[bid * 16], end_edge = nodePointer[min((bid + 1) * 16, numNodes)];
        for(int i = begin_edge + tid; i < end_edge; i += threadPerBlock){
            tmp_A[i - begin_edge] = edgeList[i];
        }
        __syncthreads();

        // 32
        if(csr_full > 0){
            for(int i = tid; i < dimTileNum * BLK_H * BLK_W; i += gridDim.x * blockDim.x){
                dense_X[i] = 0;
            }
            __syncthreads();
            // for (int z = row_nzr[bid] + wid; z < end_nzr; z += WPB) {
            for (int z = bid * BLK_H + wid; z < end_nzr; z += WPB) {
                // int row = col_nzr[z];
                int row = z;
                int target_id = row * embedding_dim + laneid;
                // int target_id = (laneid / BLK_W) * BLK_H * BLK_W + (row % BLK_H) * BLK_W + laneid % BLK_W;
                int begin_col_id = nodePointer[row], end_col_id = nodePointer[row + 1];
                for(int i = 0; i < csr_full; i++){
                    dense_X[wid * shared_size * 32 + i * 32 + laneid] = 0.0;
                }
                __syncthreads();
                for (int j = begin_col_id; j < end_col_id; j++) {
                    // int cur_edge = edgeList[j];
                    int cur_edge = tmp_A[j - begin_edge];
                    for(int i = 0; i < csr_full; i++){
                        // tmp_acc[wid * TMPSIZE * 32 + i * 32 + laneid] += input[laneid + cur_edge * embedding_dim + i * 32];
                        dense_X[(row % 16) * BLK_W + laneid % 8 + i * 4 * BLK_W * BLK_H + (laneid / 8) * BLK_W * BLK_H] += input[laneid + cur_edge * embedding_dim + i * 32];
                    }
                }
                for(int i = 0; i < csr_full; i++){
                    output2[target_id + i * 32] = dense_X[(row % 16) * BLK_W + laneid % 8 + i * 4 * BLK_W * BLK_H + (laneid / 8) * BLK_H * BLK_W];
                }
            }
        }
        // 32
        if(csr_reserve > 0){
            if(csr_reserve > 16){
                // for (int z = row_nzr[bid] + wid; z < end_nzr; z += WPB) {
                for (int z = bid * BLK_H + wid; z < end_nzr; z += WPB) {
                    // int row = col_nzr[z];
                    int row = z;
                    int target_id2 = row * embedding_dim + laneid + csr_full * 32;
                    int target_id = csr_full * BLK_H * BLK_W * 4 + (laneid / BLK_W) * BLK_H * BLK_W + (row % BLK_H) * BLK_W + laneid % BLK_W;
                    if(laneid + csr_full * 32 < embedding_dim){
                        int begin_col_id = nodePointer[row], end_col_id = nodePointer[row + 1];
                        float acc = 0.0;
                        for (int j = begin_col_id; j < end_col_id; j++) {
                            // acc += input[laneid + edgeList[j] * embedding_dim + csr_full * 32];
                            acc += input[laneid + tmp_A[j - begin_edge] * embedding_dim + csr_full * 32];
                        }
                        output2[target_id2] = acc;
                        dense_X[target_id] = acc;
                    }
                }
            }
            // 16
            else{
                int off = laneid < 16 ? 0 : 1, col_offset = laneid < 16 ? laneid : laneid - 16;
                // for (int z = row_nzr[bid] + wid * 2; z < end_nzr; z += 2 * WPB) {
                for (int z = bid * BLK_H + wid * 2; z < end_nzr; z += 2 * WPB) {
                    if(z + off < end_nzr) {
                        // int row = col_nzr[z + off];
                        int row = z + off;
                        int target_id2 = row * embedding_dim + col_offset + csr_full * 32;
                        int target_id = csr_full * BLK_H * BLK_W * 4 + (col_offset / BLK_W) * BLK_H * BLK_W + (row % BLK_H) * BLK_W + col_offset % BLK_W;
                        if(col_offset + csr_full * 32 < embedding_dim){
                            int begin_col_id = nodePointer[row], end_col_id = nodePointer[row + 1];
                            float acc = 0.0;
                            for (int j = begin_col_id; j < end_col_id; j++) {
                                // acc += input[col_offset + edgeList[j] * embedding_dim + csr_full * 32];
                                acc += input[col_offset + tmp_A[j - begin_edge] * embedding_dim + csr_full * 32];
                            }
                            output2[target_id2] = acc;
                            dense_X[target_id] = acc;
                        }
                    }
                }
            }
        }
    }

    else{
        wmma::fill_fragment(acc_frag, 0.0f);

        nIdx_start = bid * BLK_H;					    // starting nodeIdx of current row_window.
        nIdx_end = min((bid + 1) * BLK_H, numNodes);		// ending nodeIdx of current row_window.

        eIdx_start = nodePointer[nIdx_start];			// starting edgeIdx of current row_window.
        eIdx_end = nodePointer[nIdx_end];				// ending edgeIdx of the current row_window.
        num_TC_blocks = blockPartition[bid];

        // Init A_colToX_row with dummy values.
        if (tid < BLK_W * MAX_BLK) {
            sparse_AToX_index[tid] = numNodes + 1;
        }

        __syncthreads();
        
        // Init sparse_A with zero values.
#pragma unroll
        for (unsigned idx = tid; idx < BLK_W * BLK_H * MAX_BLK; idx += threadPerBlock) {
            sparse_A[idx] = 0;
        }

        #pragma unroll
        // Init dense_X with zero values.
        for (unsigned idx = tid; idx < dimTileNum * BLK_W * BLK_H; idx += threadPerBlock) {
            dense_X[idx] = 0;
        }

#pragma unroll
        for (unsigned eIdx = eIdx_start + tid; eIdx < eIdx_end; eIdx += threadPerBlock) {
            unsigned col = edgeToColumn[eIdx];
            unsigned row_local = edgeToRow[eIdx] % BLK_H;
            unsigned blk_id = col / 8;
            unsigned col_local = col % 8;
            sparse_A[row_local * BLK_W + col_local + blk_id * BLK_H * BLK_W] = 1;        // set the edge of the sparse_A.
            sparse_AToX_index[col] = edgeList[eIdx];        // record the mapping from sparse_A colId to rowId of dense_X.
        }

        __syncthreads();

        for (unsigned i = 0; i < num_TC_blocks; i++) {
            int off = laneid < 16 ? 0 : 1, col_offset = laneid < 16 ? laneid : laneid - 16;
            #pragma unroll
            for (unsigned idx = wid; idx < 4 * dimTileNum; idx += WPB) {
                unsigned dense_rowIdx = sparse_AToX_index[(idx % 4) * 2 + i * BLK_W + off];                        // TC_block_col to dense_tile_row.
                unsigned source_idx = dense_rowIdx * embedding_dim + col_offset + (idx / 4) * BLK_H;
                unsigned target_idx = (col_offset + (idx / 4) * BLK_H) * BLK_W + (idx % 4) * 2 + off;
                // boundary test.
                if (col_offset + (idx / 4) * BLK_H < embedding_dim)
                    dense_X[target_idx] = source_idx < dense_bound ? __ldca(&input[source_idx]) : 0;
            }
            __syncthreads();

            if (wid < dimTileNum) {
                wmma::load_matrix_sync(a_frag, sparse_A + i * BLK_W * BLK_H, BLK_W);
                wmma::load_matrix_sync(b_frag, dense_X + wid * BLK_W * BLK_H, BLK_W);
#pragma unroll
                for (unsigned t = 0; t < a_frag.num_elements; t++) {
                    a_frag.x[t] = wmma::__float_to_tf32(a_frag.x[t]);
                }

#pragma unroll
                for (unsigned t = 0; t < b_frag.num_elements; t++) {
                    b_frag.x[t] = wmma::__float_to_tf32(b_frag.x[t]);
                }
                // Perform the matrix multiplication.
                wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
            }

        }
        if (wid < dimTileNum){
            // Store the matrix to output matrix.
            // * Note * embeeding dimension should be padded divisible by BLK_H for output correctness.
            wmma::store_matrix_sync(output2 + bid * BLK_H * embedding_dim + wid * BLK_H, acc_frag, embedding_dim, wmma::mem_row_major);
            dense_X[wid * BLK_H * BLK_W + laneid % 4 + (laneid / 4) * BLK_W] = acc_frag.x[0];
            dense_X[wid * BLK_H * BLK_W + 4 + laneid % 4 + (laneid / 4) * BLK_W] = acc_frag.x[1];
            dense_X[wid * BLK_H * BLK_W + laneid % 4 + (laneid / 4 + 8) * BLK_W] = acc_frag.x[2];
            dense_X[wid * BLK_H * BLK_W + 4 + laneid % 4 + (laneid / 4 + 8) * BLK_W] = acc_frag.x[3];
            dense_X[wid * BLK_H * BLK_W + 8 + laneid % 4 + (laneid / 4) * BLK_W] = acc_frag.x[4];
            dense_X[wid * BLK_H * BLK_W + 12 + laneid % 4 + (laneid / 4) * BLK_W] = acc_frag.x[5];
            dense_X[wid * BLK_H * BLK_W + 8 + laneid % 4 + (laneid / 4 + 8) * BLK_W] = acc_frag.x[6];
            dense_X[wid * BLK_H * BLK_W + 12 + laneid % 4 + (laneid / 4 + 8) * BLK_W] = acc_frag.x[7];
        }
    }
    wmma::fill_fragment(acc_frag, 0.0f);

    // int row_end = ((embedding_dim + 7) / 8) * 8;
    // __syncthreads();
    for(int i = 0; i < windows_size; i++){
        for(int j = wid; j < BLK_W; j += WPB){
            unsigned source_idx = (j + i * BLK_W) * hidden_dim + laneid;
            unsigned target_idx = j + (laneid % 16) * BLK_W + BLK_W * BLK_H * (laneid / 16);
            // boundary test.
            if (j + i * BLK_W < embedding_dim)
                sparse_A[target_idx] = weights[source_idx];
            else
                sparse_A[target_idx] = 0;
        }

        __syncthreads();
        if (wid < dimTileNum) {
            wmma::load_matrix_sync(a_frag, dense_X + i * BLK_W * BLK_H, BLK_W);
            wmma::load_matrix_sync(b_frag, sparse_A + wid * BLK_W * BLK_H, BLK_W);
            #pragma unroll
            for (unsigned t = 0; t < a_frag.num_elements; t++) {
                a_frag.x[t] = wmma::__float_to_tf32(a_frag.x[t]);
            }
            #pragma unroll
            for (unsigned t = 0; t < b_frag.num_elements; t++) {
                b_frag.x[t] = wmma::__float_to_tf32(b_frag.x[t]);
            }
            // Perform the matrix multiplication.
            wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
        }
    }
    if (wid < dimTileNum)
        wmma::store_matrix_sync(output + bid * BLK_H * hidden_dim + wid * BLK_H, acc_frag, hidden_dim, wmma::mem_row_major);
}

__global__ void spmm_forward_cuda_kernel_arbi_warps_hybrid_final_fused_64(
        const int * __restrict__ nodePointer,		// node pointer.
        const int *__restrict__ edgeList,			// edge list.
        const int *__restrict__ blockPartition, 	// number of TC_blocks (16x8) in each row_window.
        const int *__restrict__ edgeToColumn, 		// eid -> col within each row_window.
        const int *__restrict__ edgeToRow, 		    // eid -> col within each row_window.
        const int numNodes,
        const int numEdges,
        const int embedding_dim,				    // embedding dimension.
        const int hidden_dim,
        const float *__restrict__ input,		    // input feature matrix.
        float *output,
        float *output2,							    // aggreAGNNed output feature matrix.
        const int *__restrict__ hybrid_type,
        const int *__restrict__ row_nzr,
        const int *__restrict__ col_nzr,
        const float *__restrict__ weights
) {
    unsigned bid = blockIdx.x;								// block_index == row_window_index
    unsigned wid = threadIdx.y;								// warp_index handling multi-dimension > 16.
    const unsigned laneid = threadIdx.x;							// lanid of each warp.
    const unsigned tid = threadIdx.y * blockDim.x + laneid;			// threadid of each block.
    // const unsigned warpSize = blockDim.x;							// number of threads per warp.
    const unsigned threadPerBlock = blockDim.x * blockDim.y;		// number of threads per block.

    const unsigned dimTileNum = (embedding_dim + BLK_H - 1) / BLK_H;              // number of tiles along the dimension
    const unsigned windows_size = (embedding_dim + BLK_W - 1) / BLK_W;              // number of tiles along the dimension
    unsigned nIdx_start = bid * BLK_H;					    // starting nodeIdx of current row_window.
    unsigned nIdx_end = min((bid + 1) * BLK_H, numNodes);		// ending nodeIdx of current row_window.

    unsigned eIdx_start = nodePointer[nIdx_start];			// starting edgeIdx of current row_window.
    unsigned eIdx_end = nodePointer[nIdx_end];				// ending edgeIdx of the current row_window.
    unsigned num_TC_blocks = blockPartition[bid]; 			// number of TC_blocks of the current row_window.
    const unsigned dense_bound = numNodes * embedding_dim;

    __shared__ float sparse_A[BLK_H * BLK_W * MAX_BLK];					// row-major sparse matrix shared memory store.
    __shared__ int sparse_AToX_index[BLK_W * MAX_BLK];					// TC_block col to dense_tile row.
    unsigned csr_full = embedding_dim / 32, csr_reserve = embedding_dim % 32;
    extern __shared__ float dense_X[];
    __shared__ int tmp_A[S_SIZE];
    // printf("%.1f\n", output[0]);
    wmma::fragment<wmma::matrix_a, BLK_H, BLK_H, BLK_W, wmma::precision::tf32, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, BLK_H, BLK_H, BLK_W, wmma::precision::tf32, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, BLK_H, BLK_H, BLK_W, float> acc_frag;
    // if(threadIdx.y == 0 && threadIdx.x == 0) printf("%d\n", blockIdx.x);
    if(hybrid_type[bid] == 0){
        // int end_nzr = row_nzr[bid + 1];
        int end_nzr = (bid + 1) * BLK_H > numNodes ? numNodes : (bid + 1) * BLK_H;

        unsigned begin_edge = nodePointer[bid * 16], end_edge = nodePointer[min((bid + 1) * 16, numNodes)];
        for(int i = begin_edge + tid; i < end_edge; i += threadPerBlock){
            tmp_A[i - begin_edge] = edgeList[i];
        }
        __syncthreads();

        // 32
        if(csr_full > 0){
            for(int i = tid; i < dimTileNum * BLK_H * BLK_W; i += gridDim.x * blockDim.x){
                dense_X[i] = 0;
            }
            __syncthreads();
            // for (int z = row_nzr[bid] + wid; z < end_nzr; z += WPB) {
            for (int z = bid * BLK_H + wid; z < end_nzr; z += WPB) {
                // int row = col_nzr[z];
                int row = z;
                int target_id = row * embedding_dim + laneid;
                // int target_id = (laneid / BLK_W) * BLK_H * BLK_W + (row % BLK_H) * BLK_W + laneid % BLK_W;
                int begin_col_id = nodePointer[row], end_col_id = nodePointer[row + 1];
                // for(int i = 0; i < csr_full; i++){
                //     dense_X[wid * TMPSIZE * 32 + i * 32 + laneid] = 0.0;
                // }
                for (int j = begin_col_id; j < end_col_id; j++) {
                    // int cur_edge = edgeList[j];
                    int cur_edge = tmp_A[j - begin_edge];
                    for(int i = 0; i < csr_full; i++){
                        // tmp_acc[wid * TMPSIZE * 32 + i * 32 + laneid] += input[laneid + cur_edge * embedding_dim + i * 32];
                        dense_X[(row % 16) * BLK_W + laneid % 8 + i * 4 * BLK_W * BLK_H + (laneid / 8) * BLK_W * BLK_H] += input[laneid + cur_edge * embedding_dim + i * 32];
                    }
                }
                for(int i = 0; i < csr_full; i++){
                    output2[target_id + i * 32] = dense_X[(row % 16) * BLK_W + laneid % 8 + i * 4 * BLK_W * BLK_H + (laneid / 8) * BLK_H * BLK_W];
                }
            }
        }
        // 32
        if(csr_reserve > 0){
            if(csr_reserve > 16){
                // for (int z = row_nzr[bid] + wid; z < end_nzr; z += WPB) {
                for (int z = bid * BLK_H + wid; z < end_nzr; z += WPB) {
                    // int row = col_nzr[z];
                    int row = z;
                    int target_id2 = row * embedding_dim + laneid + csr_full * 32;
                    int target_id = csr_full * BLK_H * BLK_W * 4 + (laneid / BLK_W) * BLK_H * BLK_W + (row % BLK_H) * BLK_W + laneid % BLK_W;
                    if(laneid + csr_full * 32 < embedding_dim){
                        int begin_col_id = nodePointer[row], end_col_id = nodePointer[row + 1];
                        float acc = 0.0;
                        for (int j = begin_col_id; j < end_col_id; j++) {
                            // acc += input[laneid + edgeList[j] * embedding_dim + csr_full * 32];
                            acc += input[laneid + tmp_A[j - begin_edge] * embedding_dim + csr_full * 32];
                        }
                        output2[target_id2] = acc;
                        dense_X[target_id] = acc;
                    }
                }
            }
            // 16
            else{
                int off = laneid < 16 ? 0 : 1, col_offset = laneid < 16 ? laneid : laneid - 16;
                // for (int z = row_nzr[bid] + wid * 2; z < end_nzr; z += 2 * WPB) {
                for (int z = bid * BLK_H + wid * 2; z < end_nzr; z += 2 * WPB) {
                    if(z + off < end_nzr) {
                        // int row = col_nzr[z + off];
                        int row = z + off;
                        int target_id2 = row * embedding_dim + col_offset + csr_full * 32;
                        int target_id = csr_full * BLK_H * BLK_W * 4 + (col_offset / BLK_W) * BLK_H * BLK_W + (row % BLK_H) * BLK_W + col_offset % BLK_W;
                        if(col_offset + csr_full * 32 < embedding_dim){
                            int begin_col_id = nodePointer[row], end_col_id = nodePointer[row + 1];
                            float acc = 0.0;
                            for (int j = begin_col_id; j < end_col_id; j++) {
                                // acc += input[col_offset + edgeList[j] * embedding_dim + csr_full * 32];
                                acc += input[col_offset + tmp_A[j - begin_edge] * embedding_dim + csr_full * 32];
                            }
                            output2[target_id2] = acc;
                            dense_X[target_id] = acc;
                        }
                    }
                }
            }
        }
    }

    else{
        wmma::fill_fragment(acc_frag, 0.0f);

        nIdx_start = bid * BLK_H;					    // starting nodeIdx of current row_window.
        nIdx_end = min((bid + 1) * BLK_H, numNodes);		// ending nodeIdx of current row_window.

        eIdx_start = nodePointer[nIdx_start];			// starting edgeIdx of current row_window.
        eIdx_end = nodePointer[nIdx_end];				// ending edgeIdx of the current row_window.
        num_TC_blocks = blockPartition[bid];

        // Init A_colToX_row with dummy values.
        if (tid < BLK_W * MAX_BLK) {
            sparse_AToX_index[tid] = numNodes + 1;
        }

        __syncthreads();
        
        // Init sparse_A with zero values.
#pragma unroll
        for (unsigned idx = tid; idx < BLK_W * BLK_H * MAX_BLK; idx += threadPerBlock) {
            sparse_A[idx] = 0;
        }

        #pragma unroll
        // Init dense_X with zero values.
        for (unsigned idx = tid; idx < dimTileNum * BLK_W * BLK_H; idx += threadPerBlock) {
            dense_X[idx] = 0;
        }

#pragma unroll
        for (unsigned eIdx = eIdx_start + tid; eIdx < eIdx_end; eIdx += threadPerBlock) {
            unsigned col = edgeToColumn[eIdx];
            unsigned row_local = edgeToRow[eIdx] % BLK_H;
            unsigned blk_id = col / 8;
            unsigned col_local = col % 8;
            sparse_A[row_local * BLK_W + col_local + blk_id * BLK_H * BLK_W] = 1;        // set the edge of the sparse_A.
            sparse_AToX_index[col] = edgeList[eIdx];        // record the mapping from sparse_A colId to rowId of dense_X.
        }

        __syncthreads();

        for (unsigned i = 0; i < num_TC_blocks; i++) {
            int off = laneid < 16 ? 0 : 1, col_offset = laneid < 16 ? laneid : laneid - 16;
            #pragma unroll
            for (unsigned idx = wid; idx < 4 * dimTileNum; idx += WPB) {
                unsigned dense_rowIdx = sparse_AToX_index[(idx % 4) * 2 + i * BLK_W + off];                        // TC_block_col to dense_tile_row.
                unsigned source_idx = dense_rowIdx * embedding_dim + col_offset + (idx / 4) * BLK_H;
                unsigned target_idx = (col_offset + (idx / 4) * BLK_H) * BLK_W + (idx % 4) * 2 + off;
                // boundary test.
                if (col_offset + (idx / 4) * BLK_H < embedding_dim)
                    dense_X[target_idx] = source_idx < dense_bound ? __ldca(&input[source_idx]) : 0;
            }
            __syncthreads();

            if (wid < dimTileNum) {
                wmma::load_matrix_sync(a_frag, sparse_A + i * BLK_W * BLK_H, BLK_W);
                wmma::load_matrix_sync(b_frag, dense_X + wid * BLK_W * BLK_H, BLK_W);
#pragma unroll
                for (unsigned t = 0; t < a_frag.num_elements; t++) {
                    a_frag.x[t] = wmma::__float_to_tf32(a_frag.x[t]);
                }

#pragma unroll
                for (unsigned t = 0; t < b_frag.num_elements; t++) {
                    b_frag.x[t] = wmma::__float_to_tf32(b_frag.x[t]);
                }
                // Perform the matrix multiplication.
                wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
            }

        }
        if (wid < dimTileNum){
            // Store the matrix to output matrix.
            // * Note * embeeding dimension should be padded divisible by BLK_H for output correctness.
            wmma::store_matrix_sync(output2 + bid * BLK_H * embedding_dim + wid * BLK_H, acc_frag, embedding_dim, wmma::mem_row_major);
            dense_X[wid * BLK_H * BLK_W + laneid % 4 + (laneid / 4) * BLK_W] = acc_frag.x[0];
            dense_X[wid * BLK_H * BLK_W + 4 + laneid % 4 + (laneid / 4) * BLK_W] = acc_frag.x[1];
            dense_X[wid * BLK_H * BLK_W + laneid % 4 + (laneid / 4 + 8) * BLK_W] = acc_frag.x[2];
            dense_X[wid * BLK_H * BLK_W + 4 + laneid % 4 + (laneid / 4 + 8) * BLK_W] = acc_frag.x[3];
            dense_X[wid * BLK_H * BLK_W + 8 + laneid % 4 + (laneid / 4) * BLK_W] = acc_frag.x[4];
            dense_X[wid * BLK_H * BLK_W + 12 + laneid % 4 + (laneid / 4) * BLK_W] = acc_frag.x[5];
            dense_X[wid * BLK_H * BLK_W + 8 + laneid % 4 + (laneid / 4 + 8) * BLK_W] = acc_frag.x[6];
            dense_X[wid * BLK_H * BLK_W + 12 + laneid % 4 + (laneid / 4 + 8) * BLK_W] = acc_frag.x[7];
        }
    }
    wmma::fill_fragment(acc_frag, 0.0f);

    // int row_end = ((embedding_dim + 7) / 8) * 8;
    // __syncthreads();
    for(int i = 0; i < windows_size; i++){
        for(int j = wid; j < BLK_W; j += WPB){
            for(int h = 0; h < 2; h++){
                unsigned source_idx = (j + i * BLK_W) * hidden_dim + laneid + h * 32;
                unsigned target_idx = j + ((laneid + h * 32) % 16) * BLK_W + BLK_W * BLK_H * ((laneid + h * 32) / 16);
                // boundary test.
                if (j + i * BLK_W < embedding_dim)
                    sparse_A[target_idx] = weights[source_idx];
                else
                    sparse_A[target_idx] = 0;
            }
        }

        __syncthreads();
        if (wid < dimTileNum) {
            wmma::load_matrix_sync(a_frag, dense_X + i * BLK_W * BLK_H, BLK_W);
            wmma::load_matrix_sync(b_frag, sparse_A + wid * BLK_W * BLK_H, BLK_W);
            #pragma unroll
            for (unsigned t = 0; t < a_frag.num_elements; t++) {
                a_frag.x[t] = wmma::__float_to_tf32(a_frag.x[t]);
            }
            #pragma unroll
            for (unsigned t = 0; t < b_frag.num_elements; t++) {
                b_frag.x[t] = wmma::__float_to_tf32(b_frag.x[t]);
            }
            // Perform the matrix multiplication.
            wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
        }
    }
    if (wid < dimTileNum)
        wmma::store_matrix_sync(output + bid * BLK_H * hidden_dim + wid * BLK_H, acc_frag, hidden_dim, wmma::mem_row_major);
}

__global__ void spmm_forward_cuda_kernel_arbi_warps_hybrid_GIN_final_fused(
        const int * __restrict__ nodePointer,		// node pointer.
        const int *__restrict__ edgeList,			// edge list.
        const int *__restrict__ blockPartition, 	// number of TC_blocks (16x8) in each row_window.
        const int *__restrict__ edgeToColumn, 		// eid -> col within each row_window.
        const int *__restrict__ edgeToRow, 		    // eid -> col within each row_window.
        const int numNodes,
        const int numEdges,
        const int embedding_dim,				    // embedding dimension.
        const int hidden_dim,
        const float *__restrict__ input,		    // input feature matrix.
        float *output,
        float *output2,							    // aggreAGNNed output feature matrix.
        const int *__restrict__ hybrid_type,
        const int *__restrict__ row_nzr,
        const int *__restrict__ col_nzr,
        const float *__restrict__ weights
) {
    unsigned bid = blockIdx.x;								// block_index == row_window_index
    unsigned wid = threadIdx.y;								// warp_index handling multi-dimension > 16.
    const unsigned laneid = threadIdx.x;							// lanid of each warp.
    const unsigned tid = threadIdx.y * blockDim.x + laneid;			// threadid of each block.
    // const unsigned warpSize = blockDim.x;							// number of threads per warp.
    const unsigned threadPerBlock = blockDim.x * blockDim.y;		// number of threads per block.

    const unsigned dimTileNum = embedding_dim / BLK_H;              // number of tiles along the dimension
    const unsigned windows_size = embedding_dim / BLK_W;              // number of tiles along the dimension
    unsigned nIdx_start = bid * BLK_H;					    // starting nodeIdx of current row_window.
    unsigned nIdx_end = min((bid + 1) * BLK_H, numNodes);		// ending nodeIdx of current row_window.

    unsigned eIdx_start = nodePointer[nIdx_start];			// starting edgeIdx of current row_window.
    unsigned eIdx_end = nodePointer[nIdx_end];				// ending edgeIdx of the current row_window.
    unsigned num_TC_blocks = blockPartition[bid]; 			// number of TC_blocks of the current row_window.
    const unsigned dense_bound = numNodes * embedding_dim;

    __shared__ float sparse_A[BLK_H * BLK_W * MAX_BLK];					// row-major sparse matrix shared memory store.
    __shared__ int sparse_AToX_index[BLK_W * MAX_BLK];					// TC_block col to dense_tile row.

    extern __shared__ float dense_X[];

    __shared__ int tmp_A[S_SIZE];
    // printf("%.1f\n", output[0]);
    wmma::fragment<wmma::matrix_a, BLK_H, BLK_H, BLK_W, wmma::precision::tf32, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, BLK_H, BLK_H, BLK_W, wmma::precision::tf32, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, BLK_H, BLK_H, BLK_W, float> acc_frag;
    // if(threadIdx.y == 0 && threadIdx.x == 0) printf("%d\n", blockIdx.x);
    if(hybrid_type[bid] == 0){
        // int end_nzr = row_nzr[bid + 1];
        int end_nzr = (bid + 1) * BLK_H > numNodes ? numNodes : (bid + 1) * BLK_H;
        
        unsigned begin_edge = nodePointer[bid * 16], end_edge = nodePointer[min((bid + 1) * 16, numNodes)];
        for(int i = begin_edge + tid; i < end_edge; i += threadPerBlock){
            tmp_A[i - begin_edge] = edgeList[i];
        }
        __syncthreads();

        // for (int z = row_nzr[bid] + wid; z < end_nzr; z += WPB) {
        for(int z = bid * BLK_H + wid; z < end_nzr; z += WPB) {
            // int row = col_nzr[z];
            int row = z;
            int target_id = (laneid / BLK_W) * BLK_H * BLK_W + (row % BLK_H) * BLK_W + laneid % BLK_W;
            int target_id2 = row * embedding_dim + laneid;
            int begin_col_id = nodePointer[row], end_col_id = nodePointer[row + 1];
            float acc = 0.0;
            for (int j = begin_col_id; j < end_col_id; j++) {
                // acc += input[laneid + edgeList[j] * embedding_dim];
                acc += input[laneid + tmp_A[j - begin_edge] * embedding_dim];
            }
            dense_X[target_id] = acc;
            output2[target_id2] = acc;
        }
        // if(blockIdx.x == 0 && threadIdx.y == 0 && threadIdx.x == 0) printf("%.1f %.1f %.1f %.1f\n", input[0], input[1],input[2],input[3]);
    }

    else{
        wmma::fill_fragment(acc_frag, 0.0f);

        nIdx_start = bid * BLK_H;					    // starting nodeIdx of current row_window.
        nIdx_end = min((bid + 1) * BLK_H, numNodes);		// ending nodeIdx of current row_window.

        eIdx_start = nodePointer[nIdx_start];			// starting edgeIdx of current row_window.
        eIdx_end = nodePointer[nIdx_end];				// ending edgeIdx of the current row_window.
        num_TC_blocks = blockPartition[bid];

        // Init A_colToX_row with dummy values.
        if (tid < BLK_W * MAX_BLK) {
            sparse_AToX_index[tid] = numNodes + 1;
        }

        __syncthreads();
        
        // Init sparse_A with zero values.
#pragma unroll
        for (unsigned idx = tid; idx < BLK_W * BLK_H * MAX_BLK; idx += threadPerBlock) {
            sparse_A[idx] = 0;
        }

        #pragma unroll
        // Init dense_X with zero values.
        for (unsigned idx = tid; idx < dimTileNum * BLK_W * BLK_H; idx += threadPerBlock) {
            dense_X[idx] = 0;
        }

#pragma unroll
        for (unsigned eIdx = eIdx_start + tid; eIdx < eIdx_end; eIdx += threadPerBlock) {
            unsigned col = edgeToColumn[eIdx];
            unsigned row_local = edgeToRow[eIdx] % BLK_H;
            unsigned blk_id = col / 8;
            unsigned col_local = col % 8;
            sparse_A[row_local * BLK_W + col_local + blk_id * BLK_H * BLK_W] = 1;        // set the edge of the sparse_A.
            sparse_AToX_index[col] = edgeList[eIdx];        // record the mapping from sparse_A colId to rowId of dense_X.
        }

        __syncthreads();

        for (unsigned i = 0; i < num_TC_blocks; i++) {
//             if (wid < dimTileNum)
// #pragma unroll
//                 for (unsigned idx = laneid; idx < BLK_W * BLK_H; idx += warpSize) {
//                     unsigned dense_rowIdx = sparse_AToX_index[idx % BLK_W + i * BLK_W];                        // TC_block_col to dense_tile_row.
//                     unsigned dense_dimIdx = idx / BLK_W;                                        // dimIndex of the dense tile.
//                     unsigned source_idx = dense_rowIdx * *embedding_dim + wid * BLK_H + dense_dimIdx;
//                     unsigned target_idx = wid * BLK_W * BLK_H + idx;
//                     // boundary test.
//                     dense_X[target_idx] = source_idx < dense_bound ? input[source_idx] : 0;
//                 }
#pragma unroll
                for (unsigned idx = wid; idx < BLK_W; idx += WPB) {
                    unsigned dense_rowIdx = sparse_AToX_index[idx % BLK_W + i * BLK_W];                        // TC_block_col to dense_tile_row.
                    unsigned source_idx = dense_rowIdx * embedding_dim + laneid;
                    unsigned target_idx = laneid * BLK_W + idx;
                    // boundary test.
                    dense_X[target_idx] = source_idx < dense_bound ? __ldca(&input[source_idx]) : 0;
                }

            __syncthreads();

            if (wid < dimTileNum) {
                wmma::load_matrix_sync(a_frag, sparse_A + i * BLK_W * BLK_H, BLK_W);
                wmma::load_matrix_sync(b_frag, dense_X + wid * BLK_W * BLK_H, BLK_W);
#pragma unroll
                for (unsigned t = 0; t < a_frag.num_elements; t++) {
                    a_frag.x[t] = wmma::__float_to_tf32(a_frag.x[t]);
                }

#pragma unroll
                for (unsigned t = 0; t < b_frag.num_elements; t++) {
                    b_frag.x[t] = wmma::__float_to_tf32(b_frag.x[t]);
                }
                // Perform the matrix multiplication.
                wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
            }

        }
        if (wid < dimTileNum){
            // Store the matrix to output matrix.
            // * Note * embeeding dimension should be padded divisible by BLK_H for output correctness.
            wmma::store_matrix_sync(output2 + bid * BLK_H * embedding_dim + wid * BLK_H, acc_frag, embedding_dim, wmma::mem_row_major);
            dense_X[wid * BLK_H * BLK_W + laneid % 4 + (laneid / 4) * BLK_W] = acc_frag.x[0];
            dense_X[wid * BLK_H * BLK_W + 4 + laneid % 4 + (laneid / 4) * BLK_W] = acc_frag.x[1];
            dense_X[wid * BLK_H * BLK_W + laneid % 4 + (laneid / 4 + 8) * BLK_W] = acc_frag.x[2];
            dense_X[wid * BLK_H * BLK_W + 4 + laneid % 4 + (laneid / 4 + 8) * BLK_W] = acc_frag.x[3];
            dense_X[wid * BLK_H * BLK_W + 8 + laneid % 4 + (laneid / 4) * BLK_W] = acc_frag.x[4];
            dense_X[wid * BLK_H * BLK_W + 12 + laneid % 4 + (laneid / 4) * BLK_W] = acc_frag.x[5];
            dense_X[wid * BLK_H * BLK_W + 8 + laneid % 4 + (laneid / 4 + 8) * BLK_W] = acc_frag.x[6];
            dense_X[wid * BLK_H * BLK_W + 12 + laneid % 4 + (laneid / 4 + 8) * BLK_W] = acc_frag.x[7];
        }
    }
    wmma::fill_fragment(acc_frag, 0.0f);
    
    for(int i = 0; i < windows_size; i++){
        for(int j = wid; j < BLK_W; j += WPB){
            unsigned source_idx = (i * BLK_W + j) * hidden_dim + laneid;
            unsigned target_idx = j + laneid * BLK_W;
            
            // boundary test.
            sparse_A[target_idx] = (laneid < hidden_dim ? weights[source_idx] : 0.0);
        }
        __syncthreads();

        if (wid < dimTileNum) {
            wmma::load_matrix_sync(a_frag, dense_X + i * BLK_W * BLK_H, BLK_W);
            wmma::load_matrix_sync(b_frag, sparse_A + wid * BLK_W * BLK_H, BLK_W);
            #pragma unroll
            for (unsigned t = 0; t < a_frag.num_elements; t++) {
                a_frag.x[t] = wmma::__float_to_tf32(a_frag.x[t]);
            }
            #pragma unroll
            for (unsigned t = 0; t < b_frag.num_elements; t++) {
                b_frag.x[t] = wmma::__float_to_tf32(b_frag.x[t]);
            }
            // Perform the matrix multiplication.
            wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
        }
    }
    if (wid < dimTileNum)
        // if(threadIdx.x == 0) printf("%.1f\n", output[0]);
        wmma::store_matrix_sync(output + bid * BLK_H * hidden_dim + wid * BLK_H, acc_frag, hidden_dim, wmma::mem_row_major);
}

// __global__ void spmm_forward_cuda_kernel_arbi_warps_hybrid_32_fused_all(
//         const int * __restrict__ nodePointer,		// node pointer.
//         const int *__restrict__ edgeList,			// edge list.
//         const int *__restrict__ blockPartition, 	// number of TC_blocks (16x8) in each row_window.
//         const int *__restrict__ edgeToColumn, 		// eid -> col within each row_window.
//         const int *__restrict__ edgeToRow, 		    // eid -> col within each row_window.
//         const int numNodes,
//         const int numEdges,
//         const int embedding_dim,				    // embedding dimension.
//         const int hidden_dim,
//         const float *__restrict__ input,		    // input feature matrix.
//         float *output,
//         float *output2,							    // aggreAGNNed output feature matrix.
//         const int *__restrict__ hybrid_type,
//         const int *__restrict__ row_nzr,
//         const int *__restrict__ col_nzr,
//         const float *__restrict__ weights,
//         const float *__restrict__ X
// ) {
//     unsigned bid = blockIdx.x;								// block_index == row_window_index
//     unsigned wid = threadIdx.y;								// warp_index handling multi-dimension > 16.
//     const unsigned laneid = threadIdx.x;							// lanid of each warp.
//     const unsigned tid = threadIdx.y * blockDim.x + laneid;			// threadid of each block.
//     // const unsigned warpSize = blockDim.x;							// number of threads per warp.
//     const unsigned threadPerBlock = blockDim.x * blockDim.y;		// number of threads per block.

//     const unsigned dimTileNum = embedding_dim / BLK_H;              // number of tiles along the dimension
//     const unsigned windows_size = embedding_dim / BLK_W;              // number of tiles along the dimension
//     unsigned nIdx_start = bid * BLK_H;					    // starting nodeIdx of current row_window.
//     unsigned nIdx_end = min((bid + 1) * BLK_H, numNodes);		// ending nodeIdx of current row_window.

//     unsigned eIdx_start = nodePointer[nIdx_start];			// starting edgeIdx of current row_window.
//     unsigned eIdx_end = nodePointer[nIdx_end];				// ending edgeIdx of the current row_window.
//     unsigned num_TC_blocks = blockPartition[bid]; 			// number of TC_blocks of the current row_window.
//     const unsigned dense_bound = numNodes * embedding_dim;

//     __shared__ float sparse_A[BLK_H * BLK_W * MAX_BLK];					// row-major sparse matrix shared memory store.
//     __shared__ int sparse_AToX_index[BLK_W * MAX_BLK];					// TC_block col to dense_tile row.

//     extern __shared__ float dense_X[];
//     // printf("%.1f\n", output[0]);
//     wmma::fragment<wmma::matrix_a, BLK_H, BLK_H, BLK_W, wmma::precision::tf32, wmma::row_major> a_frag;
//     wmma::fragment<wmma::matrix_b, BLK_H, BLK_H, BLK_W, wmma::precision::tf32, wmma::col_major> b_frag;
//     wmma::fragment<wmma::accumulator, BLK_H, BLK_H, BLK_W, float> acc_frag;
//     // if(threadIdx.y == 0 && threadIdx.x == 0) printf("%d\n", blockIdx.x);
//     if(hybrid_type[bid] == 0){
//         int end_nzr = row_nzr[bid + 1];
//         for (int z = row_nzr[bid] + wid; z < end_nzr; z += WPB) {
//             int row = col_nzr[z];
//             int target_id = (laneid / BLK_W) * BLK_H * BLK_W + (row % BLK_H) * BLK_W + laneid % BLK_W;
//             // int target_id2 = row * embedding_dim + laneid;
//             int begin_col_id = nodePointer[row], end_col_id = nodePointer[row + 1];
//             float acc = 0.0;
//             for (int j = begin_col_id; j < end_col_id; j++) {
//                 acc += input[laneid + edgeList[j] * embedding_dim];
//             }
//             dense_X[target_id] = acc;
//             // output2[target_id2] = acc;
//         }
//         // if(blockIdx.x == 0 && threadIdx.y == 0 && threadIdx.x == 0) printf("%.1f %.1f %.1f %.1f\n", input[0], input[1],input[2],input[3]);
//     }

//     else{
//         wmma::fill_fragment(acc_frag, 0.0f);

//         nIdx_start = bid * BLK_H;					    // starting nodeIdx of current row_window.
//         nIdx_end = min((bid + 1) * BLK_H, numNodes);		// ending nodeIdx of current row_window.

//         eIdx_start = nodePointer[nIdx_start];			// starting edgeIdx of current row_window.
//         eIdx_end = nodePointer[nIdx_end];				// ending edgeIdx of the current row_window.
//         num_TC_blocks = blockPartition[bid];

//         // Init A_colToX_row with dummy values.
//         if (tid < BLK_W * MAX_BLK) {
//             sparse_AToX_index[tid] = numNodes + 1;
//         }

//         __syncthreads();
        
//         // Init sparse_A with zero values.
// #pragma unroll
//         for (unsigned idx = tid; idx < BLK_W * BLK_H * MAX_BLK; idx += threadPerBlock) {
//             sparse_A[idx] = 0;
//         }

//         #pragma unroll
//         // Init dense_X with zero values.
//         for (unsigned idx = tid; idx < dimTileNum * BLK_W * BLK_H; idx += threadPerBlock) {
//             dense_X[idx] = 0;
//         }

// #pragma unroll
//         for (unsigned eIdx = eIdx_start + tid; eIdx < eIdx_end; eIdx += threadPerBlock) {
//             unsigned col = edgeToColumn[eIdx];
//             unsigned row_local = edgeToRow[eIdx] % BLK_H;
//             unsigned blk_id = col / 8;
//             unsigned col_local = col % 8;
//             sparse_A[row_local * BLK_W + col_local + blk_id * BLK_H * BLK_W] = 1;        // set the edge of the sparse_A.
//             sparse_AToX_index[col] = edgeList[eIdx];        // record the mapping from sparse_A colId to rowId of dense_X.
//         }

//         __syncthreads();

//         for (unsigned i = 0; i < num_TC_blocks; i++) {
// //             if (wid < dimTileNum)
// // #pragma unroll
// //                 for (unsigned idx = laneid; idx < BLK_W * BLK_H; idx += warpSize) {
// //                     unsigned dense_rowIdx = sparse_AToX_index[idx % BLK_W + i * BLK_W];                        // TC_block_col to dense_tile_row.
// //                     unsigned dense_dimIdx = idx / BLK_W;                                        // dimIndex of the dense tile.
// //                     unsigned source_idx = dense_rowIdx * embedding_dim + wid * BLK_H + dense_dimIdx;
// //                     unsigned target_idx = wid * BLK_W * BLK_H + idx;
// //                     // boundary test.
// //                     dense_X[target_idx] = source_idx < dense_bound ? input[source_idx] : 0;
// //                 }
// #pragma unroll
//                 for (unsigned idx = wid; idx < BLK_W; idx += WPB) {
//                     unsigned dense_rowIdx = sparse_AToX_index[idx % BLK_W + i * BLK_W];                        // TC_block_col to dense_tile_row.
//                     unsigned source_idx = dense_rowIdx * embedding_dim + laneid;
//                     unsigned target_idx = laneid * BLK_W + idx;
//                     // boundary test.
//                     dense_X[target_idx] = source_idx < dense_bound ? __ldca(&input[source_idx]) : 0;
//                 }

//             __syncthreads();

//             if (wid < dimTileNum) {
//                 wmma::load_matrix_sync(a_frag, sparse_A + i * BLK_W * BLK_H, BLK_W);
//                 wmma::load_matrix_sync(b_frag, dense_X + wid * BLK_W * BLK_H, BLK_W);
// #pragma unroll
//                 for (unsigned t = 0; t < a_frag.num_elements; t++) {
//                     a_frag.x[t] = wmma::__float_to_tf32(a_frag.x[t]);
//                 }

// #pragma unroll
//                 for (unsigned t = 0; t < b_frag.num_elements; t++) {
//                     b_frag.x[t] = wmma::__float_to_tf32(b_frag.x[t]);
//                 }
//                 // Perform the matrix multiplication.
//                 wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
//             }

//         }
//         if (wid < dimTileNum){
//             // Store the matrix to output matrix.
//             // * Note * embeeding dimension should be padded divisible by BLK_H for output correctness.
//             // wmma::store_matrix_sync(output2 + bid * BLK_H * embedding_dim + wid * BLK_H, acc_frag, embedding_dim, wmma::mem_row_major);
//             dense_X[wid * BLK_H * BLK_W + laneid % 4 + (laneid / 4) * BLK_W] = acc_frag.x[0];
//             dense_X[wid * BLK_H * BLK_W + 4 + laneid % 4 + (laneid / 4) * BLK_W] = acc_frag.x[1];
//             dense_X[wid * BLK_H * BLK_W + laneid % 4 + (laneid / 4 + 8) * BLK_W] = acc_frag.x[2];
//             dense_X[wid * BLK_H * BLK_W + 4 + laneid % 4 + (laneid / 4 + 8) * BLK_W] = acc_frag.x[3];
//             dense_X[wid * BLK_H * BLK_W + 8 + laneid % 4 + (laneid / 4) * BLK_W] = acc_frag.x[4];
//             dense_X[wid * BLK_H * BLK_W + 12 + laneid % 4 + (laneid / 4) * BLK_W] = acc_frag.x[5];
//             dense_X[wid * BLK_H * BLK_W + 8 + laneid % 4 + (laneid / 4 + 8) * BLK_W] = acc_frag.x[6];
//             dense_X[wid * BLK_H * BLK_W + 12 + laneid % 4 + (laneid / 4 + 8) * BLK_W] = acc_frag.x[7];
//         }
//     }
//     wmma::fill_fragment(acc_frag, 0.0f);

//     for(int i = wid; i < BLK_W; i += WPB){
//         unsigned source_idx = i * hidden_dim + laneid;
//         unsigned target_idx = i + laneid * BLK_W;
//         // boundary test.
//         sparse_A[target_idx] = weights[source_idx];
//     }
//     __syncthreads();
//     for(int i = 0; i < windows_size; i++){
//         if (wid < dimTileNum) {
//             wmma::load_matrix_sync(a_frag, dense_X + i * BLK_W * BLK_H, BLK_W);
//             wmma::load_matrix_sync(b_frag, sparse_A + wid * BLK_W * BLK_H, BLK_W);
//             #pragma unroll
//             for (unsigned t = 0; t < a_frag.num_elements; t++) {
//                 a_frag.x[t] = wmma::__float_to_tf32(a_frag.x[t]);
//             }
//             #pragma unroll
//             for (unsigned t = 0; t < b_frag.num_elements; t++) {
//                 b_frag.x[t] = wmma::__float_to_tf32(b_frag.x[t]);
//             }
//             // Perform the matrix multiplication.
//             wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
//         }
//     }
//     if (wid < dimTileNum)
//         wmma::store_matrix_sync(output + bid * BLK_H * hidden_dim + wid * BLK_H, acc_frag, hidden_dim, wmma::mem_row_major);
    
//     __syncthreads();
//     int order[] = {0, 1, 4, 5, 2, 3, 6, 7, 8, 9, 12, 13, 10, 11, 14, 15};

//     for(int i = wid; i < BLK_H; i += WPB){
//         unsigned target_id = (i % 2) * 4 + laneid / 8 + (laneid % 8 + 8 * ((i % 4) / 2)) * BLK_W + BLK_H * BLK_W * (i / 4);
//         sparse_A[target_id] = dense_X[order[i] * embedding_dim + laneid];
//     }

//     for(int i = wid; i < BLK_H; i += WPB){
//         unsigned source_id = (bid * BLK_H + i) * embedding_dim + laneid;
//         unsigned target_id = (i / 8 + (laneid / 16) * 2) * BLK_H * BLK_W + (laneid % BLK_H) * BLK_W + i % BLK_W;
//         dense_X[target_id] = X[source_id];
//     }
//     __syncthreads();
//     wmma::fill_fragment(acc_frag, 0.0f);
//     for(int i = 0; i < 2; i++){
//         for(int j = 0; j < 2; j++){
//             if (wid < dimTileNum) {
//                 wmma::load_matrix_sync(a_frag, dense_X + (i * 2 + j) * BLK_W * BLK_H, BLK_W);
//                 wmma::load_matrix_sync(b_frag, sparse_A + (wid + j) * BLK_W * BLK_H, BLK_W);
//                 #pragma unroll
//                 for (unsigned t = 0; t < a_frag.num_elements; t++) {
//                     a_frag.x[t] = wmma::__float_to_tf32(a_frag.x[t]);
//                 }
//                 #pragma unroll
//                 for (unsigned t = 0; t < b_frag.num_elements; t++) {
//                     b_frag.x[t] = wmma::__float_to_tf32(b_frag.x[t]);
//                 }
//                 // Perform the matrix multiplication.
//                 wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
//             }
//         }
        
//         // if (wid < dimTileNum)
//         //     wmma::store_matrix_sync(output2 + (bid * 2 + i)  * BLK_H * embedding_dim + wid * BLK_H, acc_frag, embedding_dim, wmma::mem_row_major);
//     }   
// }

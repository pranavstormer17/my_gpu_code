#include <cuda_runtime.h>
#include <iostream>

const int TILE = 16;

__global__ void matMulNaive(float* A, float* B, float* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < N && col < N) {
        float sum = 0;
        for (int k = 0; k < N; ++k)
            sum += A[row * N + k] * B[k * N + col];
        C[row * N + col] = sum;
    }
}

__global__ void matMulTiled(float* A, float* B, float* C, int N) {
    __shared__ float sA[TILE][TILE];
    __shared__ float sB[TILE][TILE];
    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;
    float sum = 0;

    for (int t = 0; t < N; t += TILE) {
        if (row < N && t+threadIdx.x < N)
            sA[threadIdx.y][threadIdx.x] = A[row * N + t + threadIdx.x];
        else sA[threadIdx.y][threadIdx.x] = 0;
        if (col < N && t+threadIdx.y < N)
            sB[threadIdx.y][threadIdx.x] = B[(t + threadIdx.y) * N + col];
        else sB[threadIdx.y][threadIdx.x] = 0;

        __syncthreads();

        for (int k = 0; k < TILE; ++k)
            sum += sA[threadIdx.y][k] * sB[k][threadIdx.x];
        __syncthreads();
    }

    if (row < N && col < N)
        C[row * N + col] = sum;
}

int main() {
    int N = 512;  // N x N matrices
    size_t bytes = N * N * sizeof(float);

    // Allocate & initialize
    float *h_A = new float[N*N], *h_B = new float[N*N], *h_C = new float[N*N];
    for (int i = 0; i < N*N; ++i) h_A[i]=h_B[i]=1.0f;

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, bytes); cudaMalloc(&d_B, bytes); cudaMalloc(&d_C, bytes);
    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);

    dim3 threads(TILE, TILE);
    dim3 blocks((N+TILE-1)/TILE, (N+TILE-1)/TILE);

    cudaEvent_t s, e; cudaEventCreate(&s); cudaEventCreate(&e);
    cudaEventRecord(s);
    matMulNaive<<<blocks, threads>>>(d_A, d_B, d_C, N);
    cudaEventRecord(e); cudaEventSynchronize(e);
    float tNaive; cudaEventElapsedTime(&tNaive, s, e);

    cudaEventRecord(s);
    matMulTiled<<<blocks, threads>>>(d_A, d_B, d_C, N);
    cudaEventRecord(e); cudaEventSynchronize(e);
    float tTiled; cudaEventElapsedTime(&tTiled, s, e);

    std::cout << "Naive: " << tNaive << " ms, Tiled: " << tTiled << " ms\n";

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    delete[] h_A; delete[] h_B; delete[] h_C;
    return 0;
}

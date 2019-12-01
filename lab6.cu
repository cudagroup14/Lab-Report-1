#include <iostream>
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <cublas_v2.h>
#include <sys/time.h>

double cpuTimer()
{
	struct timeval clock;
	gettimeofday(&clock, NULL);
	return( (double)clock.tv_sec + (double)clock.tv_usec * 1.e-6);
}

void CheckResults (float *A, float *B, int Width)
{
	for (int i=0; i<(Width*Width); i++)
	{
		if (A[i] != B[i])
		{
			printf("Results don't match!\n");
			return;
		}

	}
	printf("Results match!\n");
}

__global__ void matrixMulTiled (float *d_A,float *d_B, float *d_C, int Width)
{
	__shared__ int A[32*32];
	__shared__ int B[32*32];

	int tmpsum=0;
	int tile_sz=blockDim.x;

	int row = blockIdx.y * tile_sz + threadIdx.y;
	int col = blockIdx.x * tile_sz + threadIdx.x;

	for (int i=0; i< (Width/tile_sz); i++)
	{
		A[threadIdx.y * tile_sz + threadIdx.x] = d_A[(row * Width) + ( i*tile_sz + threadIdx.x)];
		B[threadIdx.y * tile_sz + threadIdx.x] = d_B[(i * tile_sz * Width + threadIdx.y * Width) + col];

		__syncthreads();

		for (int j=0; j<tile_sz; j++)
		{
			tmpsum += A[(threadIdx.y * tile_sz) + j] * B[(j * tile_sz) + threadIdx.x];
		}

		__syncthreads();
	}
	d_C[row * Width + col] = tmpsum;




}

__global__ void MatrixMulKernel(float* d_A, float* d_B, float* d_C, int Width) {

	int Row = blockIdx.y*blockDim.y+threadIdx.y;

	int Col = blockIdx.x*blockDim.x+threadIdx.x;
	if ((Row < Width) && (Col < Width)) {
			float tempsum = 0;


			for (int k = 0; k < Width; ++k) {
					tempsum += d_A[Row*Width+k]*d_B[k*Width+Col];

			}
				d_C[Row*Width+Col] = tempsum;
	}

}


void MatrixMultiplication(float *h_A, float *h_B, float *h_C, float *h_ST, int Width, int choice) {
    int size = Width*Width*sizeof(float);
    float *d_A = NULL;
    float *d_B = NULL;
    float *d_C = NULL;
    float *d_ST = NULL;

    cudaError_t err = cudaSuccess;

    err = cudaMalloc((void **)&d_A, size);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector A (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaMalloc((void **)&d_B, size);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector B (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaMalloc((void **)&d_C, size);

    if (err != cudaSuccess)
    {
         fprintf(stderr, "Failed to allocate device vector C (error code %s)!\n", cudaGetErrorString(err));
         exit(EXIT_FAILURE);
    }

    err = cudaMalloc((void **)&d_ST, size);

        if (err != cudaSuccess)
        {
            fprintf(stderr, "Failed to allocate device vector ST (error code %s)!\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }

    err = cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);

        if (err != cudaSuccess)
        {
            fprintf(stderr, "Failed to copy vector A from host to device (error code %s)!\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }
    err = cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector B from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }


    int threads = 32;


    int blocks = (Width + threads - 1) / threads;
    dim3 dimblocks(blocks, blocks);
    dim3 dimthreads(threads, threads);

    dim3 dimblocks2(Width, Width);

    if (choice ==1)
    {
		float ti = cpuTimer();
		MatrixMulKernel<<<dimblocks, dimthreads>>>(d_A, d_B, d_C, Width);
		cudaDeviceSynchronize();
		float elapsed = cpuTimer() - ti;
		printf("Time elapsed for the m-threaded function: %f ms\n", elapsed);

		ti = cpuTimer();
		MatrixMulKernel<<<dimblocks2, 1>>>(d_A, d_B, d_ST, Width);
		cudaDeviceSynchronize();
		elapsed = cpuTimer() - ti;
		printf("Time elapsed for the s-threaded function: %f ms\n", elapsed);
    }

    else if ( choice ==2 )
    {
		float ti = cpuTimer();
		matrixMulTiled<<<dimblocks, dimthreads>>>(d_A,d_B,d_C,Width);
		cudaDeviceSynchronize();
		float elapsed = cpuTimer() - ti;
		printf("Time elapsed for the m-threaded function: %f ms\n", elapsed);

		ti = cpuTimer();
		matrixMulTiled<<<dimblocks2, 1>>>(d_A, d_B, d_ST, Width);
		cudaDeviceSynchronize();
		elapsed = cpuTimer() - ti;
		printf("Time elapsed for the s-threaded function: %f ms\n", elapsed);
    }

    else if (choice == 3)
    {
    	cublasHandle_t handle;
    	cublasCreate(&handle);

    	float alpha = 1.0f;
    	float beta = 0.0f;

    	float ti = cpuTimer();
    	cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, Width, Width, Width, &alpha, d_A, Width, d_B, Width, &beta, d_C, Width);
    	cudaDeviceSynchronize();
    	float elapsed = cpuTimer() - ti;
    	printf("Time elapsed for the function: %f ms", elapsed);

    }


    cudaMemcpy(h_C,d_C,size,cudaMemcpyDeviceToHost);

    cudaMemcpy(h_ST,d_ST,size,cudaMemcpyDeviceToHost);

    //Free device matrices
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}
int main(int argc, char **argv) {
    int Width;
    printf("Please specify the N variable of the NxN sized matrix: ");
    scanf("%d", &Width);
    size_t size = Width*Width;

    int choice = 10;
    while (choice<1 || choice>3){
    printf("1 - Basic (naive) implementation\n"
    	   "2 - Shared memory implementation\n"
    	   "3 - Implementation using CuBLAS libraries\n"
    	   "Please choose your version: ");
    scanf("%d", &choice);
    }

    float *h_A, *h_B, *h_C, *h_ST;

    h_A = (float*)malloc(sizeof(h_A)*size);
    h_B = (float*)malloc(sizeof(h_B)*size);
    h_C = (float*)malloc(sizeof(h_C)*size);
    h_ST = (float*)malloc(sizeof(h_ST)*size);

    if (h_A == NULL || h_B == NULL || h_C == NULL || h_ST == NULL)
        {
            fprintf(stderr, "Failed to allocate host vectors!\n");
            exit(EXIT_FAILURE);
        }

    for(int i = 0; i < (Width*Width) ; i++) {
        h_A[i] = 2;
        h_B[i] = 2;
        h_C[i] = 0;
        h_ST[i] = 0;
    }
    MatrixMultiplication(h_A, h_B, h_C, h_ST, Width, choice);

    if (choice == 1 || choice == 2)
    CheckResults(h_C, h_ST, Width);

    for(int i = 0; i < (Width*Width) ; i++)
    {
    	if ((i%(Width)) == 0)
    	printf("\n");
        printf("%f ", h_C[i]);
    }

    return 0;
}




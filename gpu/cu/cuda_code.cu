#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <cuda_runtime.h>

#include "mats.h"

__global__ void add(int *a, int *b, int *c)
{
	*c = *a + *b;
}

void calc()
{
	int a = 2, b = 7, c;
	int *da, *db, *dc;
	int size = sizeof(a);

	cudaMalloc((void **)&da, size);
	cudaMalloc((void **)&db, size);
	cudaMalloc((void **)&dc, size);

	cudaMemcpy(da, &a, size, cudaMemcpyHostToDevice);
	cudaMemcpy(db, &b, size, cudaMemcpyHostToDevice);

	add<<<1, 1>>>(da, db, dc);

	cudaMemcpy(&c, dc, size, cudaMemcpyDeviceToHost);

	cudaFree(da);
	cudaFree(db);
	cudaFree(dc);

	printf("c = %d\n", c);
}

extern "C"
cudaError_t cuda_main()
{
	// generate 16M random numbers on the host
	thrust::host_vector<int> h_vec(1 << 24);
	thrust::generate(h_vec.begin(), h_vec.end(), rand);

	// transfer data to the device
	thrust::device_vector<int> d_vec = h_vec;

	// sort data on the device (805 Mkeys/sec on GeForce GTX 480)
	thrust::sort(d_vec.begin(), d_vec.end());

	// transfer data back to host
	thrust::copy(d_vec.begin(), d_vec.end(), h_vec.begin());

	calc();

	return cudaGetLastError();
}

__global__ void matmul(float* DA, float *DB, float *DC, int ARows, int BRows, int BCols)
{
	int row = threadIdx.y + blockIdx.y * blockDim.y;
	int col = threadIdx.x + blockIdx.x * blockDim.x;

	float sC = 0;

	if(row < ARows && col < BCols){
		for(int i = 0; i < BRows; i++){
			sC += DA[row * BRows + i] * DB[i * BCols + col];
		}
		DC[row * BCols + col] = sC;
	}
}

extern "C"
cudaError_t cuda_mult(mats::Mat<float> *a, mats::Mat<float> *b, mats::Mat<float> *c)
{
#define BLOCKSIZE	16

	if(a->cols != b->rows)
		return cudaGetLastError();

	int x1 = round((double)a->rows / BLOCKSIZE + 0.5);
	int x2 = round((double)a->cols / BLOCKSIZE + 0.5);

	if(!x1) x1 = 1;
	if(!x2) x2 = 1;

	//printf("------ dimGrid(%d, %d)-------", x1, x2);

	dim3 dimGrid(x1, x2), dimBlock(BLOCKSIZE, BLOCKSIZE);

	float *dA, *dB, *dC;

	cudaMalloc((void **)&dA, a->size());
	cudaMalloc((void **)&dB, b->size());
	cudaMalloc((void **)&dC, c->size());

	cudaMemcpy(dA, a->data, a->size(), cudaMemcpyHostToDevice);
	cudaMemcpy(dB, b->data, b->size(), cudaMemcpyHostToDevice);
	cudaMemcpy(dC, c->data, c->size(), cudaMemcpyHostToDevice);

	matmul<<<dimGrid, dimBlock>>>(dA, dB, dC, a->rows, a->cols, b->cols);

//	std::cout << "time_pass=" << mats::getTick() - tick << std::endl;

	cudaMemcpy(c->data, dC, c->size(), cudaMemcpyDeviceToHost);

	cudaFree(dA);
	cudaFree(dB);
	cudaFree(dC);

	return cudaGetLastError();
}

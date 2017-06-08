#ifndef COMMON_DEVICES_H
#define COMMON_DEVICES_H

#include <cuda_runtime.h>

#include "cuda_common.h"

namespace gpumat{

namespace internal{

struct DMtx{
	int stride;
	u_char *data;
};

template< typename T >
inline __device__ DMtx getSubMatrix(Mtx A, int row, int col, int BRows = BLOCKSIZE, int BCols = BLOCKSIZE)
{
	DMtx res;

	T *d = (T*)A.data;
	res.stride = A.cols;
	res.data = (u_char*)&d[A.cols * BRows * row + col * BCols];
	return res;
}

template< typename T >
inline __device__ T getEl(DMtx A, int row, int col)
{
//	T *d = (T*)A.data;
	return ((T*)A.data)[A.stride * row + col];
}

template< typename T >
inline __device__ void setEl(DMtx A, int row, int col, T val)
{
//	T *d = (T*)A.data;
	((T*)A.data)[A.stride * row + col] = val;
}

}

}

#endif // COMMON_DEVICES_H

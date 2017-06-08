#include "cuda_types.h"

#include "gpumat.h"
#include "cuda_common.h"

#include <cuda_runtime.h>

namespace gpumat {

namespace internal{

SmallMtxArray::SmallMtxArray()
{
	count = 0;
	allocate = 0;
	size = 0;
	mtx = 0;
	m_delete = true;
}

SmallMtxArray::~SmallMtxArray()
{
	if(mtx && m_delete){
		cudaFree(mtx);
	}
}

SmallMtxArray::SmallMtxArray(const std::vector<gpumat::GpuMat> &gmat)
{
	//				if(maxcount < gmat.size())
	//					throw new std::invalid_argument("not enough size of array for store matrices");

	count = gmat.size();

	size_t sz = sizeof(Mtx) * count;

	allocate = sz;
	size = sz;

	cudaMalloc(&mtx, sz);

	std::vector< Mtx > tmp;
	tmp.resize(count);

	for(size_t i = 0; i < count; ++i){
		tmp[i] = gmat[i];
	}
	cudaMemcpy(mtx, &tmp[0], sz, cudaMemcpyHostToDevice);
}

void SmallMtxArray::set(const std::vector<gpumat::GpuMat> &gmat)
{
	count = gmat.size();

	size_t sz = sizeof(Mtx) * count;

	if(sz > allocate){
		allocate = sz;
		size = sz;
		cudaMalloc(&mtx, sz);
	}
	size = sz;

	std::vector< Mtx > tmp;
	tmp.resize(count);

	for(size_t i = 0; i < count; ++i){
		tmp[i] = gmat[i];
	}
	cudaMemcpy(mtx, &tmp[0], sz, cudaMemcpyHostToDevice);

}

void SmallMtxArray::copyFrom(const SmallMtxArray &mt)
{
	size = mt.size;
	allocate = mt.allocate;
	count = mt.count;
	mtx = mt.mtx;
	m_delete = mt.m_delete;
}

void SmallMtxArray::setDelete(bool val)
{
	m_delete = val;
}

}

}

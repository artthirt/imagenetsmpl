#include "cuda_common.h"

using namespace gpumat::internal;

SmallMtxArrayStatic::SmallMtxArrayStatic(){
	count = 0;
}

SmallMtxArrayStatic::~SmallMtxArrayStatic(){
}

SmallMtxArrayStatic::SmallMtxArrayStatic(const std::vector<gpumat::GpuMat> &gmat, int beg, int last){
	if(maxcount < last - beg || last > (int)gmat.size() || beg > (int)gmat.size())
		throw new std::invalid_argument("not enough size of array for store matrices");

	count = last - beg;
	for(int i = beg, j = 0; i < last; ++i, ++j){
		mtx[j] = gmat[i];
	}
}

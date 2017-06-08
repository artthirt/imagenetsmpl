#ifndef CUDA_COMMON_H
#define CUDA_COMMON_H

#include "gpumat.h"

#include <cuda_runtime.h>

/**
  size of block for cuda gpu
*/
#define BLOCKSIZE	32

namespace gpumat{

	enum etypefunction{
		LINEAR = 0,
		RELU = 1,
		SIGMOID,
		SOFTMAX,
		TANH
	};

	namespace internal{

		struct Mtx{
			int rows;
			int cols;
			u_char* data;

			__host__ __device__ Mtx(){
				rows = cols = 0;
				data = 0;
			}
			__host__ __device__ Mtx(int rows, int cols, void* data){
				this->rows = rows;
				this->cols = cols;
				this->data = (u_char*)data;
			}
			__host__ __device__ Mtx(const gpumat::GpuMat& mat){
				rows = mat.rows;
				cols = mat.cols;
				data = mat.data;
			}

			__host__ __device__ int total() const{
				return rows * cols;
			}
		};

		template< typename T>
		struct SmallSingleArray{
			enum {maxcount = 64};

			SmallSingleArray(){
				count = 0;
			}
			SmallSingleArray(const std::vector< T >& gv){
				if(maxcount < gv.size())
					throw new std::invalid_argument("not enough size of array for store matrices");

				count = gv.size();
				for(int i = 0; i < count; ++i){
					values[i] = gv[i];
				}
			}
			template< typename C >
			SmallSingleArray(const std::vector< C >& gv){
				if(maxcount < gv.size())
					throw new std::invalid_argument("not enough size of array for store matrices");

				count = gv.size();
				for(int i = 0; i < count; ++i){
					values[i] = gv[i];
				}
			}

			size_t count;
			T values[maxcount];
		};

		struct SmallMtxArrayStatic{
			enum {maxcount = 50};
			SmallMtxArrayStatic();
			~SmallMtxArrayStatic();

			SmallMtxArrayStatic(const std::vector< GpuMat >& gmat, int beg, int last);

			int count;
			internal::Mtx mtx[maxcount];
		};

	}/* @end internal */

}/* @end gpumat */

#endif // CUDA_COMMON_H

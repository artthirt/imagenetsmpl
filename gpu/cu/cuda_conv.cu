#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
#include <thrust/reduce.h>
#include <thrust/execution_policy.h>

#include "gpumat.h"
#include "cuda_common.h"
#include "common_types.h"

#include "common_devices.h"

using namespace gpumat;

///////// begin internal namespace ///////////////

namespace gpumat{

namespace internal{


class Singleton{
public:
	Singleton(){
		std::fill((char*)&m_prop, (char*)&m_prop + sizeof(cudaDeviceProp), '\0');
		cudaError_t err = cudaGetDevice(& m_device);
		if(err == cudaSuccess){
			err = cudaGetDeviceProperties(&m_prop, m_device);
			if(err == cudaSuccess){
				std::cout << "Gpu work. Shared memory: " << m_prop.sharedMemPerBlock << std::endl;
			}else{
				std::cout << "gpu not work\n";
			}
		}
	}

	cudaDeviceProp &prop(){
		return m_prop;
	}

	size_t shared_memory() const{
		return m_prop.sharedMemPerBlock;
	}

	static Singleton &instance(){
		return m_instance;
	}

private:
	static Singleton m_instance;

	int m_device;
	cudaDeviceProp m_prop;
};

Singleton Singleton::m_instance;

template< typename T >
inline __device__ T empty(T val)
{
	return val;
}

template< typename T >
inline __device__ T reLu(T val)
{
	return max(val, T(0));
}

template< typename T >
inline __device__ T deriv_reLu(T val)
{
	return val > 0? T(1) : T(0);
}

////////////

template< typename T >
__global__ void conv2d(Mtx A0, SmallMtxArray W, SmallMtxArray A1,
					   ct::Size szI, ct::Size szO, int stride,
					   SmallMtxArray B, etypefunction func)
{
	int row = threadIdx.y + blockIdx.y * blockDim.y;
	int col = threadIdx.x + blockIdx.x * blockDim.x;

	typedef T (*func_t)(T val);

//	func_t _func = empty;
//	switch (func) {
//		case RELU:
//			_func = reLu;
//			break;
//	}

	if(row < A1.mtx[0].rows && col < A1.mtx[0].cols){
		int yr = col / szO.width;
		int xr = col - yr * szO.width;

		int x = xr * stride;
		int y = yr * stride;

		T *dA0 = (T*)A0.data;
		T *dA0i = &dA0[row * A0.cols];

		for(int w = 0; w < W.count; ++w){
			Mtx& Wi = W.mtx[w];
			Mtx A1I = A1.mtx[w];
			T *dA1 = (T*)A1I.data;
			T *dA1i = &dA1[row * A1I.cols];
			T *dBi = (T*)B.mtx[w].data;

			T *dW = (T*)Wi.data;
			T sum = 0;
			for(int a = 0; a < Wi.rows; ++a){
				if(y + a < szI.height){
					for(int b = 0; b < Wi.cols; ++b){
						if(x + b < szI.width){
							sum += dA0i[(y + a) * szI.width + (x + b)] * dW[a * Wi.cols + b];
						}
					}
				}
			}

			sum += dBi[0];

			switch (func) {
			case RELU:
				sum = reLu(sum);
				break;
			default:
				break;
			}

//			sum = _func(sum);
			dA1i[col] = sum;
		}
	}
}

template< typename T >
__global__ void conv2d(Mtx A0, Mtx W, Mtx A1,
					   ct::Size szI, ct::Size szO, int stride,
					   Mtx B, etypefunction func)
{
	int row = threadIdx.y + blockIdx.y * blockDim.y;
	int col = threadIdx.x + blockIdx.x * blockDim.x;

	typedef T (*func_t)(T val);

//	func_t _func = empty;
//	switch (func) {
//		case RELU:
//			_func = reLu;
//			break;
//	}

	if(row < A1.rows && col < A1.cols){
		int yr = col / szO.width;
		int xr = col - yr * szO.width;

		int x = xr * stride;
		int y = yr * stride;

		T *dA0 = (T*)A0.data;
		T *dA0i = &dA0[row * A0.cols];

		{
			T *dA1 = (T*)A1.data;
			T *dA1i = &dA1[row * A1.cols];
			T *dBi = (T*)B.data;

			T *dW = (T*)W.data;
			T sum = 0;
			for(int a = 0; a < W.rows; ++a){
				if(y + a < szI.height){
					for(int b = 0; b < W.cols; ++b){
						if(x + b < szI.width){
							sum += dA0i[(y + a) * szI.width + (x + b)] * dW[a * W.cols + b];
						}
					}
				}
			}

			sum += dBi[0];

			switch (func) {
			case RELU:
				sum = reLu(sum);
				break;
			default:
				break;
			}

//			sum = _func(sum);
			dA1i[col] = sum;
		}
	}
}

template< typename T >
__global__ void subsample(Mtx A0, Mtx A1, Mtx Mask, ct::Size szA0, ct::Size szA1)
{
	const int kLen = 2;

	int row = threadIdx.y + blockIdx.y * blockDim.y;
	int col = threadIdx.x + blockIdx.x * blockDim.x;

	if(row < A1.rows && col < A1.cols){
		int yr = col / szA1.width;
		int xr = col - yr * szA1.width;

		int x = xr * kLen;
		int y = yr * kLen;

		T *dA0 = (T*)A0.data;
		T *dM = (T*)Mask.data;
		T *dA1 = (T*)A1.data;

		T *dA0i = &dA0[row * A0.cols];
		T *dMi = &dM[row * Mask.cols];
		T *dA1i = &dA1[row * A1.cols];

		T maximum = dA0i[(y) * szA0.width + (x)];
		int xm = x, ym = y;
		for(int a = 0; a < kLen; ++a){
			if(y + a < szA0.height){
				for(int b = 0; b < kLen; ++b){
					if(x + b < szA0.width){
//						dMi[(y + a) * szA0.width + (x + b)] = 0;
						T val = dA0i[(y + a) * szA0.width + (x + b)];
						if(val > maximum){
							xm = x + b; ym = y + a;
							maximum = val;
						}
					}
				}
			}
		}
		dMi[ym * szA0.width + xm] = 1;
		dA1i[yr * szA1.width + xr] = maximum;
	}
}

template< typename T >
__global__ void upsample(Mtx A1, Mtx Mask, Mtx A0, ct::Size szA1, ct::Size szA0)
{
	const int kLen = 2;

	int row = threadIdx.y + blockIdx.y * blockDim.y;
	int col = threadIdx.x + blockIdx.x * blockDim.x;

	if(row < A1.rows && col < A1.cols){
		int yr = col / szA1.width;
		int xr = col - yr * szA1.width;

		int x = xr * kLen;
		int y = yr * kLen;

		T *dA0 = (T*)A0.data;
		T *dM = (T*)Mask.data;
		T *dA1 = (T*)A1.data;

		T *dA0i = &dA0[row * A0.cols];
		T *dMi = &dM[row * Mask.cols];
		T *dA1i = &dA1[row * A1.cols];

		T val = dA1i[yr * szA1.width + xr];

		for(int a = 0; a < kLen; ++a){
			if(y + a < szA0.height){
				for(int b = 0; b < kLen; ++b){
					if(x + b < szA0.width){
						T mask = dMi[(y + a) * szA0.width + (x + b)];
						dA0i[(y + a) * szA0.width + (x + b)] = val * mask;
					}
				}
			}
		}
	}
}

template< typename T >
__global__ void deriv_conv2d(Mtx A0, Mtx gA1, ct::Size szA0, ct::Size szA1, Mtx gW, int stride, Mtx Blocks)
{
	int row = threadIdx.y + blockIdx.y * blockDim.y;
	int col = threadIdx.x + blockIdx.x * blockDim.x;

	int srow = threadIdx.y, scol = threadIdx.x;

	extern __shared__ int iW[];
	T *sW = (T*)iW;


	if(row < gA1.rows && col < gA1.cols){
		int y = col / szA1.width;
		int x = col - y * szA1.width;

		Mtx H(blockDim.y * gW.rows, blockDim.x * gW.cols, sW);

		int blkX = blockDim.x;
		int blkY = blockDim.y;

		DMtx HSub = getSubMatrix<T>(H, srow, scol, gW.rows, gW.cols);

		if(srow == 0 && scol == 0){
			for(int i = 0; i < H.rows * H.cols; ++i){
				sW[i] = 0;
			}
		}

		__syncthreads();

		int x0 = x * stride;
		int y0 = y * stride;

		T *dA0 = (T*)A0.data;
		T *dgA1 = (T*)gA1.data;

		T *dA0i = &dA0[row * A0.cols];
		T *dgA1i = &dgA1[row * gA1.cols];

		T d = dgA1i[y * szA1.width + x];

		for(int a = 0; a < gW.rows; ++a){
			int y1 = y0 + a;
			if(y1 < szA0.height){
				for(int b = 0; b < gW.cols; ++b){
					int x1 = x0 + b;
					if(x1 < szA0.width){
						T a0 = dA0i[y1 * szA0.width + x1];
						setEl<T>(HSub, a, b, d * a0);
					}
				}
			}
		}

		__syncthreads();

		if(srow == 0 && scol == 0){
			int brow = blockIdx.y;
			int bcol = blockIdx.x;

			DMtx BSub = getSubMatrix<T>(Blocks, brow, bcol, gW.rows, gW.cols);

			for(int a = 0; a < gW.rows; ++a){
				for(int b = 0; b < gW.cols; ++b){
					T val = getEl<T>(BSub, a, b);

//					DMtx HSub = getSubMatrix<T>(H, 1, 3, gW.rows, gW.cols);
//					val = getEl<T>(HSub, a, b);
					for(int y = 0; y < blkY; ++y){
						for(int x = 0; x < blkX; ++x){
							DMtx HSub = getSubMatrix<T>(H, y, x, gW.rows, gW.cols);
							val += getEl<T>(HSub, a, b);
						}
					}

					setEl<T>(BSub, a, b, val);
				}
			}
		}

	}
}

template< typename T >
__global__ void reduce_blocks(Mtx Blocks, Mtx W, T val = 1.)
{
	int row = threadIdx.y + blockIdx.y * blockDim.y;
	int col = threadIdx.x + blockIdx.x * blockDim.x;

	if(row < W.rows && col < W.cols){
		T *dW = (T*)W.data;
		T *dB = (T*)Blocks.data;

		int ca = Blocks.rows / W.rows;
		int cb = Blocks.cols / W.cols;
		for(int a = 0; a < ca; ++a){
			for(int b = 0; b < cb; ++b){
				int ra = a * W.rows;
				int rb = b * W.cols;

				T *dBi = &dB[ra * Blocks.cols + rb];
				dW[row * W.cols + col] += dBi[row * Blocks.cols + col];
			}
		}
		dW[row * W.cols + col] *= val;
	}

}

template< typename T >
__global__ void deriv_prev_conv2d(SmallMtxArray deriv, SmallMtxArray W,
								  ct::Size sL, ct::Size sLsub1, int stride, Mtx D)
{
	int row = threadIdx.y + blockIdx.y * blockDim.y;
	int col = threadIdx.x + blockIdx.x * blockDim.x;

	if(row < D.rows && col < D.cols){
		int y = col / sLsub1.width;
		int x = col - y * sLsub1.width;

		int x0 = x / stride;
		int y0 = y / stride;

		T *dD = (T*)D.data;

		T *dDi = &dD[row * D.cols];

		T sum = 0;
		for(int w = 0; w < W.count; ++w){
			T *dDrv = (T*)deriv.mtx[w].data;
			T *dDrvi = &dDrv[row * deriv.mtx[w].cols];

			Mtx& Wi = W.mtx[w];
			T *dW = (T*)Wi.data;

			for(int a = 0; a < Wi.rows; ++a){
				int yi = y0 - a;
				if(yi >=0 && yi < sL.height){
					for(int b = 0; b < Wi.cols; ++b){
						int xi = x0 - b;
						if(xi >=0 && xi < sL.width){
							T d = dDrvi[yi * sL.width + xi];
							T w = dW[a * Wi.cols + b];
							sum += d * w;
						}
					}/* W.cols */
				}
			}/* W.rows */
		}/* W */
		dDi[y * sLsub1.width + x] = sum;
	}
}

template< typename T >
__global__ void hsplit(Mtx Res, SmallMtxArray List)
{
	int row = threadIdx.y + blockIdx.y * blockDim.y;
	int col = threadIdx.x + blockIdx.x * blockDim.x;

	if(row < Res.rows && col < Res.cols){
		T *dR =(T*)Res.data;

		int lid = col / List.mtx[0].cols;
		Mtx& mtx = List.mtx[lid];
		int lcol = col - lid * mtx.cols;
		T* dM = (T*)mtx.data;
		dM[row * mtx.cols + lcol] = dR[row * Res.cols + col];
	}
}

template< typename T >
__global__ void hsplit(int beg, int count, Mtx Res, SmallMtxArrayStatic List)
{
	int row = threadIdx.y + blockIdx.y * blockDim.y;
	int col = threadIdx.x + blockIdx.x * blockDim.x;

	if(row < Res.rows && col < count && beg + col < Res.cols){
		T *dR =(T*)Res.data;

		int lid = col / List.mtx[0].cols;
		Mtx& mtx = List.mtx[lid];
		int lcol = col - lid * mtx.cols;
		T* dM = (T*)mtx.data;
		dM[row * mtx.cols + lcol] = dR[row * Res.cols + beg + col];
	}
}

template< typename T >
__global__ void hconcat(SmallMtxArray List, Mtx Res)
{
	int row = threadIdx.y + blockIdx.y * blockDim.y;
	int col = threadIdx.x + blockIdx.x * blockDim.x;

	if(row < Res.rows && col < Res.cols){
		T *dR =(T*)Res.data;

		int lid = col / List.mtx[0].cols;
		Mtx& mtx = List.mtx[lid];
		int lcol = col - lid * mtx.cols;
		T* dM = (T*)mtx.data;
		dR[row * Res.cols + col] = dM[row * mtx.cols + lcol];
	}
}

template< typename T >
__global__ void hconcat(int beg, int count, Mtx Res, SmallMtxArrayStatic List)
{
	int row = threadIdx.y + blockIdx.y * blockDim.y;
	int col = threadIdx.x + blockIdx.x * blockDim.x;

	if(row < Res.rows && col < count && beg + col < Res.cols){
		T *dR =(T*)Res.data;

		int lid = col / List.mtx[0].cols;
		Mtx& mtx = List.mtx[lid];
		int lcol = col - lid * mtx.cols;
		T* dM = (T*)mtx.data;
		dR[row * Res.cols + beg + col] = dM[row * mtx.cols + lcol];
	}
}

template< typename T >
__global__ void reduce_all(T* dIn, int sizeIn, T* dOut, int sizeOut, int block)
{
	int col = threadIdx.x + blockIdx.x * blockDim.x;

	if(col < sizeOut){
		int id1 = block * col;
		dOut[col] = 0;
		int cnt = min(block, sizeIn - id1);
		for(int i = id1; i < id1 + cnt; ++i){
			dOut[col] += dIn[i];
		}
	}
}

}/*@internal end*/

}/*@gpumat end*/

///////////

extern "C"
void cuda_conv2d(const GpuMat &A0,
				 const ct::Size &szI, const ct::Size &szO,
				 int stride,
				 const std::vector<GpuMat> &W,
				 const std::vector<GpuMat> B,
				 std::vector<GpuMat> &A1,
				 etypefunction func)
{
	int x1 = A1[0].cols / BLOCKSIZE + 1;
	int x2 = A1[0].rows / BLOCKSIZE + 1;

	dim3 dimGrid(x1, x2), dimBlock(BLOCKSIZE, BLOCKSIZE);

//	internal::SmallMtxArray sW(W), sA1(A1), sB(B);

#pragma omp parallel for
	for(int w = 0; w < W.size(); ++w){
		switch (A0.type) {
			case GPU_DOUBLE:{
				internal::conv2d<double> <<<dimGrid, dimBlock>>>(A0, W[w], A1[w], szI, szO, stride, B[w], func);
				break;
			}
			case GPU_FLOAT:{
				internal::conv2d<float> <<<dimGrid, dimBlock>>>(A0, W[w], A1[w], szI, szO, stride, B[w], func);
				break;
			}
		}
	}
}

extern "C"
void cuda_subsample(const GpuMat &A0,
					const ct::Size &szA0,
					const ct::Size &szA1,
					GpuMat &A1,
					GpuMat &Mask)
{
	int x1 = A1.cols / BLOCKSIZE + 1;
	int x2 = A1.rows / BLOCKSIZE + 1;

	dim3 dimGrid(x1, x2), dimBlock(BLOCKSIZE, BLOCKSIZE);

	gpumat::memset(Mask, 0);

	switch (A0.type) {
	case GPU_DOUBLE:
		internal::subsample<double> <<<dimGrid, dimBlock>>>(A0, A1, Mask, szA0, szA1);
		break;
	case GPU_FLOAT:
		internal::subsample<float> <<<dimGrid, dimBlock>>>(A0, A1, Mask, szA0, szA1);
		break;
	}
}

extern "C"
void cuda_upsample(const GpuMat &A1,
				   const ct::Size &szA1,
				   const ct::Size &szA0,
				   const GpuMat &Mask,
				   GpuMat &A0)
{
	int x1 = A1.cols / BLOCKSIZE + 1;
	int x2 = A1.rows / BLOCKSIZE + 1;

	dim3 dimGrid(x1, x2), dimBlock(BLOCKSIZE, BLOCKSIZE);

	switch (A0.type) {
		case GPU_DOUBLE:
		internal::upsample<double> <<<dimGrid, dimBlock>>>(A1, Mask, A0, szA1, szA0);
		break;
	case GPU_FLOAT:
		internal::upsample<float> <<<dimGrid, dimBlock>>>(A1, Mask, A0, szA1, szA0);
		break;
	}
}

template< typename T >
void cuda_reduce_blocks(const GpuMat& Blocks, GpuMat& W, T val)
{
	int x1 = W.cols / BLOCKSIZE + 1;
	int x2 = W.rows / BLOCKSIZE + 1;

	dim3 dimGrid(x1, x2), dimBlock(BLOCKSIZE, BLOCKSIZE);

	internal::reduce_blocks<T> <<< dimGrid, dimBlock >>>(Blocks, W, val);
}

template< typename T >
struct CudaMem{
	T* data;
	int size;
	int allocate;

	CudaMem(){
		data = 0;
		size = 0;
		allocate = 0;
	}
	~CudaMem(){
		release();
	}
	void release(){
		if(data != 0){
			cudaFree(data);
		}
		size = 0;
		allocate = 0;
	}

	bool empty(){
		return data == 0;
	}
	void create(int size){
		if(size > allocate){
			release();
		}else{
			this->size = size;
			return;
		}
		this->size = size;
		this->allocate = size;
		assert(cudaMalloc(&data, allocate) == cudaSuccess);
	}
};

/// ! Not thread safe
template< typename T >
void cuda_reduce(const GpuMat& mat, GpuMat& res)
{
#if 1
	//T *res1, *res2;
	static CudaMem<T> res1, res2;

	int block = 32;

	int size2 = (mat.total() / block + 1);
	int size2_bytes = size2 * sizeof(T);

	res1.create(size2_bytes);
	res2.create(size2_bytes);
//	assert(cudaMalloc(&res1, size2_bytes) == cudaSuccess);
//	assert(cudaMalloc(&res2, size2_bytes) == cudaSuccess);
//	assert(cudaMemcpy(res1, mat.data, mat.size(), cudaMemcpyDeviceToDevice) == cudaSuccess);

//	std::vector< T > vec;
//	vec.resize(size2);

	int size1 = mat.total();
	T *d1 = (T*)mat.data, *d2 = res2.data;
	int cnt = mat.total();
//	if((cnt % 2) == 1)cnt += 1;
	for(int s = 1; s < cnt; s *= block){

		int x1 = size2 / BLOCKSIZE + 1;

		internal::reduce_all<T> <<< x1, BLOCKSIZE >>>(d1, size1, d2, size2, block);

//		cudaMemcpy(&vec[0], d2, sizeof(T) * size2, cudaMemcpyDeviceToHost);
		if(d1 == (T*)mat.data)
			d1 = res1.data;

		std::swap(d1, d2);
		size1 = size2;
		size2 = size2 / block + 1;

	}

	cudaMemcpy(res.data, d1, sizeof(T), cudaMemcpyDeviceToDevice);
//	cudaMemcpy(&vec[0], d1, sizeof(T), cudaMemcpyDeviceToHost);

//	cudaFree(res1);
//	cudaFree(res2);
#else
	res = thrust::reduce(thrust::device, (T*)mat.data, (T*)mat.data + mat.total());
#endif
}

extern "C"
void cuda_reduce_all(const GpuMat& A, GpuMat &res)
{
	if(A.empty())
		return;

	switch (A.type) {
		case gpumat::GPU_DOUBLE:
			cuda_reduce<double>(A, res);
			break;
		case gpumat::GPU_FLOAT:
			cuda_reduce<float>(A, res);
			break;
	}

}

extern "C"
void cuda_deriv_conv2d(const GpuMat &A0, const GpuMat &gradA1,
				  const ct::Size &szA0, const ct::Size &szA1,
				  int stride,
				  GpuMat &gradW, GpuMat &gradB, GpuMat *pblocks)
{
	int blocksize = 8;
	int x1 = gradA1.cols / blocksize + 1;
	int x2 = gradA1.rows / blocksize + 1;

	dim3 dimGrid(x1, x2), dimBlock(blocksize, blocksize);

	int size_shared = gradW.size() * blocksize * blocksize;

	assert(internal::Singleton::instance().shared_memory() > size_shared);

	gpumat::GpuMat inner_mat;
	gpumat::GpuMat &blocks = pblocks != 0? *pblocks : inner_mat;

	blocks.resize(x2 * gradW.rows, x1 * gradW.cols, gradW.type);
	gpumat::memset(blocks, 0);

	switch (A0.type) {
		case GPU_DOUBLE:{
			internal::deriv_conv2d<double> <<<dimGrid, dimBlock, size_shared >>>(A0, gradA1, szA0, szA1,
																   gradW, stride, blocks);
			cuda_reduce_blocks<double>(blocks, gradW, 1./gradA1.rows);
			//double val = thrust::reduce(thrust::device, (double*)gradA1.data, (double*)gradA1.data + gradA1.total());
			cuda_reduce<double>(gradA1, gradB);
			break;
		}
		case GPU_FLOAT:{
			internal::deriv_conv2d<float> <<<dimGrid, dimBlock, size_shared >>>(A0, gradA1, szA0, szA1,
																  gradW, stride, blocks);
//			std::cout << blocks.print() << std::endl;
//			std::cout << blocks.print() << std::endl << A0.print() << std::endl << gradA1.print() << std::endl;
			cuda_reduce_blocks<float>(blocks, gradW, 1.f/gradA1.rows);
//			float val = thrust::reduce(thrust::device, (float*)gradA1.data, (float*)gradA1.data + gradA1.total());
			cuda_reduce<float>(gradA1, gradB);
			break;
		}
	}
	gpumat::mulval(gradB, (double)1./gradA1.total());

}

extern "C"
void cuda_deriv_prev_conv2d(const std::vector<GpuMat> &deriv,
							const std::vector<GpuMat> &W,
							const ct::Size &sL, const ct::Size &sLsub1, int stride,
							GpuMat &D)
{
	int x1 = D.cols / BLOCKSIZE + 1;
	int x2 = D.rows / BLOCKSIZE + 1;

	dim3 dimGrid(x1, x2), dimBlock(BLOCKSIZE, BLOCKSIZE);

	D.sderiv.set(deriv);
	D.sW.set(W);

	D.sderiv.setDelete(false);
	D.sW.setDelete(false);

	switch (D.type) {
	case GPU_DOUBLE:
		internal::deriv_prev_conv2d<double> <<<dimGrid, dimBlock>>>(D.sderiv, D.sW, sL, sLsub1, stride, D);
		break;
	case GPU_FLOAT:
		internal::deriv_prev_conv2d<float> <<<dimGrid, dimBlock>>>(D.sderiv, D.sW, sL, sLsub1, stride, D);
		break;
	}

	D.sderiv.setDelete(true);
	D.sW.setDelete(true);
}

extern "C"
void cuda_hsplit(const GpuMat &res, std::vector<GpuMat> &list)
{
#if 1
	int x1 = res.cols / BLOCKSIZE + 1;
	int x2 = res.rows / BLOCKSIZE + 1;

	dim3 dimGrid(x1, x2), dimBlock(BLOCKSIZE, BLOCKSIZE);

	internal::SmallMtxArray slist(list);

	switch (res.type) {
	case GPU_DOUBLE:
		internal::hsplit<double> <<<dimGrid, dimBlock>>>(res, slist);
		break;
	case GPU_FLOAT:
		internal::hsplit<float> <<<dimGrid, dimBlock>>>(res, slist);
		break;
	}
#else
	int block = internal::SmallMtxArrayStatic::maxcount;
	int xx1, offset;

	//int x1 = res.cols / BLOCKSIZE + 1;
	int x2 = res.rows / BLOCKSIZE + 1;

	int lcols = list[0].cols;

	dim3 dimBlock(BLOCKSIZE, BLOCKSIZE);

//#pragma omp parallel for
	for(int i = 0; i < list.size(); i += block){
		int beg = i;
		int last = beg + min((int)list.size() - i, block);
		internal::SmallMtxArrayStatic slist(list, beg, last);

		offset = lcols * beg;

		xx1 = min(block, res.cols - offset);
		dim3 dimGrid((lcols * xx1) / BLOCKSIZE + 1, x2);

//		std::cout << offset << " " << xx1 << std::endl;

		switch (res.type) {
		case GPU_DOUBLE:
			internal::hsplit<double> <<<dimGrid, dimBlock>>>(offset, lcols * xx1, res, slist);
			break;
		case GPU_FLOAT:
			internal::hsplit<float> <<<dimGrid, dimBlock>>>(offset, lcols * xx1, res, slist);
			break;
		}
	}
#endif
}

extern "C"
void cuda_hconcat(const std::vector<GpuMat> &list, GpuMat &res)
{
#if 1
	int x1 = res.cols / BLOCKSIZE + 1;
	int x2 = res.rows / BLOCKSIZE + 1;

	dim3 dimGrid(x1, x2), dimBlock(BLOCKSIZE, BLOCKSIZE);

	internal::SmallMtxArray slist(list);

	switch (res.type) {
	case GPU_DOUBLE:
		internal::hconcat<double> <<<dimGrid, dimBlock>>>(slist, res);
		break;
	case GPU_FLOAT:
		internal::hconcat<float> <<<dimGrid, dimBlock>>>(slist, res);
		break;
	}
#else
	int block = internal::SmallMtxArrayStatic::maxcount;
	int xx1, offset;

	//int x1 = res.cols / BLOCKSIZE + 1;
	int x2 = res.rows / BLOCKSIZE + 1;

	int lcols = list[0].cols;

	dim3 dimBlock(BLOCKSIZE, BLOCKSIZE);

//#pragma omp parallel for
	for(int i = 0; i < list.size(); i += block){
		int beg = i;
		int last = beg + min((int)list.size() - i, block);
		internal::SmallMtxArrayStatic slist(list, beg, last);

		offset = lcols * beg;

		xx1 = min(block, res.cols - offset);
		dim3 dimGrid((lcols * xx1) / BLOCKSIZE + 1, x2);

//		std::cout << offset << " " << xx1 << std::endl;

		switch (res.type) {
		case GPU_DOUBLE:
			internal::hconcat<double> <<<dimGrid, dimBlock>>>(offset, lcols * xx1, res, slist);
			break;
		case GPU_FLOAT:
			internal::hconcat<float> <<<dimGrid, dimBlock>>>(offset, lcols * xx1, res, slist);
			break;
		}
	}
#endif
}

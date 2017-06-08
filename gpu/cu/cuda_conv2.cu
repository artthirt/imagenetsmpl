#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
#include <thrust/reduce.h>
#include <thrust/execution_policy.h>

#include "gpumat.h"
#include "cuda_common.h"
#include "common_types.h"

#include "common_devices.h"
#include "cuda_types.h"

using namespace gpumat;

///////// begin internal namespace ///////////////

namespace gpumat{

namespace internal{

template< typename T >
__device__ void _im2cols(const Mtx& X, const ct::Size& szA0, int channels, const ct::Size& szW, int stride, Mtx& Res, const ct::Size& szOut)
{
	int col = threadIdx.x + blockIdx.x * blockDim.x;

	int szOutArea = szOut.width * szOut.height;
	int all = szOutArea * channels;

	if(col < all){
		int c = col / szOutArea;
		int offset = col - c * szOutArea;

		int y = offset / szOut.width;
		int x = offset - y * szOut.width;

		int x0 = x * stride;
		int y0 = y * stride;
		int row2 = y * szOut.width + x;

		int szA0area = szA0.width * szA0.height;
		int szWarea = szW.width * szW.height;

		T *dX = (T*)X.data;
		T *dR = (T*)Res.data;
		T *dXi = &dX[c * szA0area];

		for(int a = 0; a < szW.height; ++a){
			for(int b = 0; b < szW.width; ++b){
				int col2 = c * szWarea + (a * szW.width + b);
				if(y0 + a < szA0.height && x0 + b < szA0.width){
					dR[row2 * Res.cols + col2] = dXi[(y0 + a) * szA0.width + (x0 + b)];
				}
			}
		}
	}
}

template< typename T >
__device__ void _im2colsT(const Mtx& X, const ct::Size& szA0, int channels, const ct::Size& szW, int stride, Mtx& Res, const ct::Size& szOut)
{
	int col = threadIdx.x + blockIdx.x * blockDim.x;

	int szOutArea = szOut.width * szOut.height;
	int all = szOutArea * channels;

	if(col < all){
		int c = col / szOutArea;
		int offset = col - c * szOutArea;

		int y = offset / szOut.width;
		int x = offset - y * szOut.width;

		int x0 = x * stride;
		int y0 = y * stride;
		int row2 = y * szOut.width + x;

//		int szA0area = szA0.width * szA0.height;
		int szWarea = szW.width * szW.height;

		T *dR = (T*)Res.data;
		T *dXi = (T*)X.data + c;

		for(int a = 0; a < szW.height; ++a){
			for(int b = 0; b < szW.width; ++b){
				int col2 = c * szWarea + (a * szW.width + b);
				if(y0 + a < szA0.height && x0 + b < szA0.width){
					dR[row2 * Res.cols + col2] = dXi[((y0 + a) * szA0.width + (x0 + b)) * channels];
				}
			}
		}
	}
}

template< typename T >
__global__ void im2cols(Mtx X, ct::Size szA0, int channels, ct::Size szW, int stride, Mtx Res, ct::Size szOut)
{
	_im2cols<T>(X, szA0, channels, szW, stride, Res, szOut);
}

template< typename T >
__global__ void im2cols_vec(SmallMtxArray X, ct::Size szA0, int channels, ct::Size szW, int stride, SmallMtxArray Res, ct::Size szOut)
{
	int row = threadIdx.y + blockDim.y * blockIdx.y;

	if(row < X.count){
		_im2cols<T>(X.mtx[row], szA0, channels, szW, stride, Res.mtx[row], szOut);
	}
}

////////

template< typename T >
__global__ void im2colsT(Mtx X, ct::Size szA0, int channels, ct::Size szW, int stride, Mtx Res, ct::Size szOut)
{
	_im2colsT<T>(X, szA0, channels, szW, stride, Res, szOut);
}

template< typename T >
__global__ void im2colsT_vec(SmallMtxArray X, ct::Size szA0, int channels, ct::Size szW, int stride, SmallMtxArray Res, ct::Size szOut)
{
	int row = threadIdx.y + blockDim.y * blockIdx.y;

	if(row < X.count){
		_im2colsT<T>(X.mtx[row], szA0, channels, szW, stride, Res.mtx[row], szOut);
	}
}

////////

template< typename T >
__device__ void _back_deriv(const Mtx& Delta,
				 const ct::Size& szOut,
				 const ct::Size& szA0,
				 int channels,
				 const ct::Size& szW,
				 int stride,
				 Mtx X)
{
	int col = threadIdx.x + blockIdx.x * blockDim.x;

	int szA0Area = szA0.width * szA0.height;
	int all = szA0Area * channels;
	if(col < all){
//		int c = col / szOutArea;
//		int offset = col - c * szOutArea;

//		int y = offset / szOut.width;
//		int x = offset - y * szOut.width;

//		int x0 = x * stride;
//		int y0 = y * stride;
//		int row2 = y * szOut.width + x;

//		int szA0area = szA0.width * szA0.height;
//		int szWarea = szW.width * szW.height;

//		T *dX = (T*)X.data;
//		T *dR = (T*)Delta.data;
//		T *dXi = &dX[c * szA0area];

//		for(int a = 0; a < szW.height; ++a){
//			for(int b = 0; b < szW.width; ++b){
//				int col2 = c * szWarea + (a * szW.width + b);
//				if(y0 + a < szA0.height && x0 + b < szA0.width){
//					dXi[(y0 + a) * szA0.width + (x0 + b)] += dR[row2 * Delta.cols + col2];
//				}
//			}
//		}
		int c = col / szA0Area;
		int offset = col - c * szA0Area;

		int y = offset / szA0.width;
		int x = offset - y * szA0.width;

		int szWarea = szW.width * szW.height;

		T *dX = (T*)X.data;
		T *dR = (T*)Delta.data;
		T *dXi = &dX[c * szA0Area];

		T sum = 0;
		for(int a = 0; a < szW.height; ++a){
			if((y - a) % stride == 0){
				int y0 = (y - a) / stride;
				for(int b = 0; b < szW.width; ++b){

					if((x - b) % stride == 0){

						int x0 = (x - b) / stride;

						if(y0 >= 0 && y0 < szOut.height &&
								x0 >= 0 && x0 < szOut.width){
							int row2 = y0 * szOut.width + x0;
							int col2 = c * szWarea + (a * szW.width + b);
							T val = dR[row2 * Delta.cols + col2];
							sum += val;
						}
					}
				}
			}
		}
		dXi[y * szA0.width + x] = sum;

	}
}

//////

template< typename T >
__device__ void _back_derivT(const Mtx& Delta,
				 const ct::Size& szOut,
				 const ct::Size& szA0,
				 int channels,
				 const ct::Size& szW,
				 int stride,
				 Mtx X)
{
	int col = threadIdx.x + blockIdx.x * blockDim.x;

	int szA0Area = szA0.width * szA0.height;
	int all = szA0Area * channels;

	if(col < all){
		int c = col / szA0Area;
		int offset = col - c * szA0Area;

		int y = offset / szA0.width;
		int x = offset - y * szA0.width;

//		int x0 = x * stride;
//		int y0 = y * stride;
//		int row2 = y * szOut.width + x;

//		int szA0area = szA0.width * szA0.height;
		int szWarea = szW.width * szW.height;

		T *dR = (T*)Delta.data;
		T *dXi = (T*)X.data + c;

//		for(int a = 0; a < szW.height; ++a){
//			for(int b = 0; b < szW.width; ++b){
//				int col2 = c * szWarea + (a * szW.width + b);
//				if(y0 + a < szA0.height && x0 + b < szA0.width){
//					dXi[((y0 + a) * szA0.width + (x0 + b)) * channels] += dR[row2 * Delta.cols + col2];
//				}
//			}
//		}
		T sum = 0;
		for(int a = 0; a < szW.height; ++a){
			if((y - a) % stride == 0){
				int y0 = (y - a) / stride;
				for(int b = 0; b < szW.width; ++b){

					if((x - b) % stride == 0){

						int x0 = (x - b) / stride;

						if(y0 >= 0 && y0 < szOut.height &&
								x0 >= 0 && x0 < szOut.width){
							int row2 = y0 * szOut.width + x0;
							int col2 = c * szWarea + (a * szW.width + b);
							T val = dR[row2 * Delta.cols + col2];
							sum += val;
						}
					}
				}
			}
		}
		dXi[(y * szA0.width + x) * channels] = sum;
	}
}

/////

template< typename T >
__global__ void back_deriv(Mtx Delta,
						   ct::Size szOut,
						   ct::Size szA0,
						   int channels,
						   ct::Size szW,
						   int stride,
						   Mtx X)
{
	_back_deriv<T>(Delta, szOut, szA0, channels, szW, stride, X);
}

template< typename T >
__global__ void back_deriv_vec(SmallMtxArray Delta,
						   ct::Size szOut,
						   ct::Size szA0,
						   int channels,
						   ct::Size szW,
						   int stride,
						   SmallMtxArray X)
{
	int row = threadIdx.y + blockDim.y * blockIdx.y;

	if(row < X.count){
		_back_deriv<T>(Delta.mtx[row], szOut, szA0, channels, szW, stride, X.mtx[row]);
	}
}

////////

template< typename T >
__global__ void back_derivT(Mtx Delta,
						   ct::Size szOut,
						   ct::Size szA0,
						   int channels,
						   ct::Size szW,
						   int stride,
						   Mtx X)
{
	_back_derivT<T>(Delta, szOut, szA0, channels, szW, stride, X);
}

template< typename T >
__global__ void back_derivT_vec(SmallMtxArray Delta,
						   ct::Size szOut,
						   ct::Size szA0,
						   int channels,
						   ct::Size szW,
						   int stride,
						   SmallMtxArray X)
{
	int row = threadIdx.y + blockDim.y * blockIdx.y;

	if(row < X.count){
		_back_derivT<T>(Delta.mtx[row], szOut, szA0, channels, szW, stride, X.mtx[row]);
	}
}


////////////////

template< typename T >
__device__ void _subsample(const Mtx &X,
						   int K,
						   const ct::Size& szA,
						   Mtx Y,
						   Mtx Mask,
						   const ct::Size& szO)
{
	int col = threadIdx.x + blockIdx.x * blockDim.x;

	int szOutArea = szO.width * szO.height;
	int all = szOutArea * K;

	const int stride = 2;

	if(col < all){
		int k = col / szOutArea;
		int offset = col - k * szOutArea;

		int y = offset / szO.width;
		int x = offset - y * szO.width;

		T *dX = (T*)X.data + k;
		T* dM = (T*)Mask.data + k;
		T *dY = (T*)Y.data + k;

		int y0 = y * stride;
		int x0 = x * stride;

		T mmax = dX[(y0 * szA.width + x0) * X.cols];
		int xm = x0, ym = y0;
		T resM = 0;

		for(int a = 0; a < stride; ++a){
			for(int b = 0; b < stride; ++b){
				if(y0 + a < szA.height && x0 + b < szA.width){
					T val = dX[((y0 + a) * szA.width + (x0 + b)) * X.cols];
					if(val > mmax){
						mmax = val;
						xm = x0 + b;
						ym = y0 + a;
						resM = 1;
					}
				}
			}
		}

		dY[(y * szO.width + x) * Y.cols] = mmax;
		dM[(ym * szA.width + xm) * Mask.cols] = resM;
	}
}

template< typename T >
__global__ void subsample(Mtx X,
						  int K,
						  ct::Size szA,
						  Mtx Y,
						  Mtx Mask,
						  ct::Size szO)
{
	_subsample<T>(X, K, szA, Y, Mask, szO);
}

template< typename T >
__global__ void subsample_vec(SmallMtxArray X,
						  int K,
						  ct::Size szA,
						  SmallMtxArray Y,
						  SmallMtxArray Mask,
						  ct::Size szO)
{
	int row = threadIdx.y + blockDim.y * blockIdx.y;

	if(row < X.count){
		_subsample<T>(X.mtx[row], K, szA, Y.mtx[row], Mask.mtx[row], szO);
	}
}

template< typename T >
__device__ void _upsample(const Mtx &Y,
						 const Mtx &Mask,
						 int K,
						 const ct::Size &szO,
						 const ct::Size &szA,
						 Mtx X)
{
	int col = threadIdx.x + blockIdx.x * blockDim.x;

	int szOutArea = szO.width * szO.height;
	int all = szOutArea * K;

	int stride = 2;

	if(col < all){
		int k = col / szOutArea;
		int offset = col - k * szOutArea;

		int y = offset / szO.width;
		int x = offset - y * szO.width;

		T *dX = (T*)(X.data) + k;
		T* dM = (T*)(Mask.data) + k;
		T *dY = (T*)(Y.data) + k;

		int y0 = y * stride;
		int x0 = x * stride;

		T val = dY[(y * szO.width + x) * K];

		for(int a = 0; a < stride; ++a){
			for(int b = 0; b < stride; ++b){
				if(y0 + a < szA.height && x0 + b < szA.width){
					T m = dM[((y0 + a) * szA.width + (x0 + b)) * Mask.cols];
					dX[((y0 + a) * szA.width + (x0 + b)) * X.cols] = val * m;
				}
			}
		}
	}
}

template< typename T >
__global__ void upsample(Mtx Y,
						 Mtx Mask,
						 int K,
						 ct::Size szO,
						 ct::Size szA,
						 Mtx X)
{
	_upsample<T>(Y, Mask, K, szO, szA, X);
}

template< typename T >
__global__ void upsample_vec(SmallMtxArray Y,
							 SmallMtxArray Mask,
							 int K,
							 ct::Size szO,
							 ct::Size szA,
							 SmallMtxArray X)
{
	int row = threadIdx.y + blockDim.y * blockIdx.y;

	if(row < X.count){
		_upsample<T>(Y.mtx[row], Mask.mtx[row], K, szO, szA, X.mtx[row]);
	}
}

template< typename T >
__global__ void vec2mat(SmallMtxArray vec, Mtx mat)
{
	int col = threadIdx.x + blockDim.x * blockIdx.x;
	int row = threadIdx.y + blockDim.y * blockIdx.y;

	if(row < mat.rows && col < mat.cols){
		T* dV = (T*)vec.mtx[row].data;
		T* dM = (T*)mat.data;

		dM[row * mat.cols + col] = dV[col];
	}
}

template< typename T >
__global__ void mat2vec(Mtx mat, ct::Size sz, SmallMtxArray vec)
{
	int col = threadIdx.x + blockDim.x * blockIdx.x;
	int row = threadIdx.y + blockDim.y * blockIdx.y;

	if(row < mat.rows && col < mat.cols){
		T* dV = (T*)vec.mtx[row].data;
		T* dM = (T*)mat.data;

		int y = col/sz.width;
		int x = col - y * sz.width;

		dV[y * sz.width + x] = dM[row * mat.cols + col];
	}
}

}	/// @endnamespace internal

}	/// @endnamespace gpumat

extern "C"
void cuda_im2cols(const gpumat::GpuMat &X,
				  const ct::Size &szA0,
				  int channels,
				  const ct::Size &szW,
				  int stride,
				  gpumat::GpuMat &Res,
				  ct::Size &szOut)
{
	int x1 = szOut.area() * channels / BLOCKSIZE + 1;
	int x2 = 1;

	dim3 dimGrid(x1, x2), dimBlock(BLOCKSIZE, 1);

	switch (X.type) {
		case GPU_DOUBLE:
			internal::im2cols<double> <<<dimGrid, dimBlock>>>(X, szA0, channels, szW, stride, Res, szOut);
			break;
		case GPU_FLOAT:
			internal::im2cols<float> <<<dimGrid, dimBlock>>>(X, szA0, channels, szW, stride, Res, szOut);
			break;
	}
}

extern "C"
void cuda_im2cols_vec(const std::vector< gpumat::GpuMat > &X,
				  const ct::Size &szA0,
				  int channels,
				  const ct::Size &szW,
				  int stride,
				  std::vector< gpumat::GpuMat > &Res,
				  ct::Size &szOut)
{
	int x1 = szOut.area() * channels / BLOCKSIZE + 1;
	int x2 = (int)X.size() / BLOCKSIZE + 1;

	dim3 dimGrid(x1, x2), dimBlock(BLOCKSIZE, BLOCKSIZE);

	internal::SmallMtxArray sX(X), sRes(Res);

	switch (X[0].type) {
		case GPU_DOUBLE:
			internal::im2cols_vec<double> <<<dimGrid, dimBlock>>>(sX, szA0, channels, szW, stride, sRes, szOut);
			break;
		case GPU_FLOAT:
			internal::im2cols_vec<float> <<<dimGrid, dimBlock>>>(sX, szA0, channels, szW, stride, sRes, szOut);
			break;
	}
}

//////////

extern "C"
void cuda_im2colsT(const gpumat::GpuMat &X,
				  const ct::Size &szA0,
				  int channels,
				  const ct::Size &szW,
				  int stride,
				  gpumat::GpuMat &Res,
				  ct::Size &szOut)
{
	int x1 = szOut.area() * channels / BLOCKSIZE + 1;
	int x2 = 1;

	dim3 dimGrid(x1, x2), dimBlock(BLOCKSIZE, 1);

	switch (X.type) {
		case GPU_DOUBLE:
			internal::im2colsT<double> <<<dimGrid, dimBlock>>>(X, szA0, channels, szW, stride, Res, szOut);
			break;
		case GPU_FLOAT:
			internal::im2colsT<float> <<<dimGrid, dimBlock>>>(X, szA0, channels, szW, stride, Res, szOut);
			break;
	}
}

extern "C"
void cuda_im2colsT_vec(const std::vector< gpumat::GpuMat > &X,
				  const ct::Size &szA0,
				  int channels,
				  const ct::Size &szW,
				  int stride,
				  std::vector< gpumat::GpuMat > &Res,
				  ct::Size &szOut)
{
	int x1 = szOut.area() * channels / BLOCKSIZE + 1;
	int x2 = (int)X.size() / BLOCKSIZE + 1;

	dim3 dimGrid(x1, x2), dimBlock(BLOCKSIZE, BLOCKSIZE);

	internal::SmallMtxArray sX(X), sRes(Res);

	switch (X[0].type) {
		case GPU_DOUBLE:
			internal::im2colsT_vec<double> <<<dimGrid, dimBlock>>>(sX, szA0, channels, szW, stride, sRes, szOut);
			break;
		case GPU_FLOAT:
			internal::im2colsT_vec<float> <<<dimGrid, dimBlock>>>(sX, szA0, channels, szW, stride, sRes, szOut);
			break;
	}
}

//////////

extern "C"
void cuda_back_deriv(const gpumat::GpuMat &Delta,
				const ct::Size &szOut,
				const ct::Size &szA0,
				int channels,
				const ct::Size &szW,
				int stride,
				gpumat::GpuMat &X)
{
	int x1 = szA0.area() * channels / BLOCKSIZE + 1;
	int x2 = 1;

	dim3 dimGrid(x1, x2), dimBlock(BLOCKSIZE, 1);

	switch (X.type) {
		case GPU_DOUBLE:
			internal::back_deriv<double> <<<dimGrid, dimBlock>>>(Delta, szOut, szA0, channels, szW, stride, X);
			break;
		case GPU_FLOAT:
			internal::back_deriv<float> <<<dimGrid, dimBlock>>>(Delta, szOut, szA0, channels, szW, stride, X);
			break;
	}
}

extern "C"
void cuda_back_deriv_vec(const std::vector< gpumat::GpuMat > &Delta,
				const ct::Size &szOut,
				const ct::Size &szA0,
				int channels,
				const ct::Size &szW,
				int stride,
				std::vector< gpumat::GpuMat > &X)
{
	int x1 = szA0.area() * channels / BLOCKSIZE + 1;
	int x2 = (int)X.size() / BLOCKSIZE + 1;

	dim3 dimGrid(x1, x2), dimBlock(BLOCKSIZE, BLOCKSIZE);

	switch (Delta[0].type) {
		case GPU_DOUBLE:
			internal::back_deriv_vec<double> <<<dimGrid, dimBlock>>>(Delta, szOut, szA0, channels, szW, stride, X);
			break;
		case GPU_FLOAT:
			internal::back_deriv_vec<float> <<<dimGrid, dimBlock>>>(Delta, szOut, szA0, channels, szW, stride, X);
			break;
	}
}

//////////////////

extern "C"
void cuda_back_derivT(const gpumat::GpuMat &Delta,
				const ct::Size &szOut,
				const ct::Size &szA0,
				int channels,
				const ct::Size &szW,
				int stride,
				gpumat::GpuMat &X)
{
	int x1 = szA0.area() * channels / BLOCKSIZE + 1;
	int x2 = 1;

	dim3 dimGrid(x1, x2), dimBlock(BLOCKSIZE, 1);

	switch (X.type) {
		case GPU_DOUBLE:
			internal::back_derivT<double> <<<dimGrid, dimBlock>>>(Delta, szOut, szA0, channels, szW, stride, X);
			break;
		case GPU_FLOAT:
			internal::back_derivT<float> <<<dimGrid, dimBlock>>>(Delta, szOut, szA0, channels, szW, stride, X);
			break;
	}
}

extern "C"
void cuda_back_derivT_vec(const std::vector< gpumat::GpuMat > &Delta,
				const ct::Size &szOut,
				const ct::Size &szA0,
				int channels,
				const ct::Size &szW,
				int stride,
				std::vector< gpumat::GpuMat > &X)
{
	int x1 = szA0.area() * channels / BLOCKSIZE + 1;
	int x2 = (int)X.size() / BLOCKSIZE + 1;

	dim3 dimGrid(x1, x2), dimBlock(BLOCKSIZE, BLOCKSIZE);

	switch (Delta[0].type) {
		case GPU_DOUBLE:
			internal::back_derivT_vec<double> <<<dimGrid, dimBlock>>>(Delta, szOut, szA0, channels, szW, stride, X);
			break;
		case GPU_FLOAT:
			internal::back_derivT_vec<float> <<<dimGrid, dimBlock>>>(Delta, szOut, szA0, channels, szW, stride, X);
			break;
	}
}

//////////////////

extern "C"
void cuda_subsample2(const gpumat::GpuMat &X,
							  const ct::Size &szA,
							  gpumat::GpuMat &Y,
							  gpumat::GpuMat &Mask,
							  ct::Size &szO)
{
	int K = X.cols;
	int x1 = szO.area() * K / BLOCKSIZE + 1;
	int x2 = 1;

	dim3 dimGrid(x1, x2), dimBlock(BLOCKSIZE, 1);

	switch (X.type) {
		case GPU_DOUBLE:
			internal::subsample<double> <<<dimGrid, dimBlock>>>(X, K, szA, Y, Mask, szO);
			break;
		case GPU_FLOAT:
			internal::subsample<float> <<<dimGrid, dimBlock>>>(X, K, szA, Y, Mask, szO);
			break;
	}
}

extern "C"
void cuda_subsample2_vec(const std::vector< gpumat::GpuMat > &X,
					const ct::Size &szA,
					std::vector< gpumat::GpuMat > &Y,
					std::vector< gpumat::GpuMat > &Mask,
					ct::Size &szO)
{
	int K = X[0].cols;
	int x1 = szO.area() * K / BLOCKSIZE + 1;
	int x2 = (int)X.size() / BLOCKSIZE + 1;

	dim3 dimGrid(x1, x2), dimBlock(BLOCKSIZE, BLOCKSIZE);

	switch (X[0].type) {
		case GPU_DOUBLE:
			internal::subsample_vec<double> <<<dimGrid, dimBlock>>>(X, K, szA, Y, Mask, szO);
			break;
		case GPU_FLOAT:
			internal::subsample_vec<float> <<<dimGrid, dimBlock>>>(X, K, szA, Y, Mask, szO);
			break;
	}
}

extern "C"
void cuda_upsample2(const gpumat::GpuMat &Y, const gpumat::GpuMat &Mask, const ct::Size &szO,
			  const ct::Size &szA, gpumat::GpuMat &X)
{
	int K = X.cols;
	int x1 = szO.area() * K / BLOCKSIZE + 1;
	int x2 = 1;

	dim3 dimGrid(x1, x2), dimBlock(BLOCKSIZE, 1);

	switch (X.type) {
		case GPU_DOUBLE:
			internal::upsample<double> <<<dimGrid, dimBlock>>>(Y, Mask, K, szO, szA, X);
			break;
		case GPU_FLOAT:
			internal::upsample<float> <<<dimGrid, dimBlock>>>(Y, Mask, K, szO, szA, X);
			break;
	}
}

extern "C"
void cuda_upsample2vec(const std::vector<gpumat::GpuMat> &Y, const std::vector<gpumat::GpuMat> &Mask,
			  const ct::Size &szO, const ct::Size &szA, std::vector<gpumat::GpuMat> &X)
{
	int K = X[0].cols;
	int x1 = szO.area() * K / BLOCKSIZE + 1;
	int x2 = (int)X.size() / BLOCKSIZE + 1;

	dim3 dimGrid(x1, x2), dimBlock(BLOCKSIZE, BLOCKSIZE);

	switch (X[0].type) {
		case GPU_DOUBLE:
			internal::upsample_vec<double> <<<dimGrid, dimBlock>>>(Y, Mask, K, szO, szA, X);
			break;
		case GPU_FLOAT:
			internal::upsample_vec<float> <<<dimGrid, dimBlock>>>(Y, Mask, K, szO, szA, X);
			break;
	}
}


extern "C"
void cuda_vec2mat(const std::vector< GpuMat >& vec, GpuMat& mat)
{
	int rows = mat.rows;
	int cols = mat.cols;

	int x1 = cols / BLOCKSIZE + 1;
	int x2 = rows / BLOCKSIZE + 1;

	dim3 dimGrid(x1, x2), dimBlock(BLOCKSIZE, BLOCKSIZE);

	switch (vec[0].type) {
		case GPU_DOUBLE:
			internal::vec2mat<double> <<<dimGrid, dimBlock>>>(vec, mat);
			break;
		case GPU_FLOAT:
			internal::vec2mat<float> <<<dimGrid, dimBlock>>>(vec, mat);
			break;
	}
}

extern "C"
void cuda_mat2vec(const GpuMat& mat, const ct::Size& sz, std::vector< GpuMat >& vec)
{
	int rows = mat.rows;
	int cols = mat.cols;

	int x1 = cols / BLOCKSIZE + 1;
	int x2 = rows / BLOCKSIZE + 1;

	dim3 dimGrid(x1, x2), dimBlock(BLOCKSIZE, BLOCKSIZE);

	switch (vec[0].type) {
		case GPU_DOUBLE:
			internal::mat2vec<double> <<<dimGrid, dimBlock>>>(mat, sz, vec);
			break;
		case GPU_FLOAT:
			internal::mat2vec<float> <<<dimGrid, dimBlock>>>(mat, sz, vec);
			break;
	}
}

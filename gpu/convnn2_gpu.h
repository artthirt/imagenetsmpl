#ifndef CONV2_GPU_H
#define CONV2_GPU_H

#include "gpumat.h"
#include "helper_gpu.h"
#include "cuda_common.h"

namespace gpumat{

namespace conv2{

class convnn_gpu
{
public:
	std::vector< gpumat::GpuMat > W;		/// weights
	std::vector< gpumat::GpuMat > B;		/// biases
	int kernels;									/// kernels
	int channels;							/// input channels
	int stride;
	ct::Size szA0;							/// input size
	ct::Size szA1;							/// size after convolution
	ct::Size szA2;							/// size after pooling
	ct::Size szW;							/// size of weights
	ct::Size szK;							/// size of output data (set in forward)
	std::vector< gpumat::GpuMat >* pX;		/// input data
	std::vector< gpumat::GpuMat > Xc;		///
	std::vector< gpumat::GpuMat > A1;		/// out after appl nonlinear function
	std::vector< gpumat::GpuMat > A2;		/// out after pooling
	std::vector< gpumat::GpuMat > Dlt;		/// delta after backward pass
	std::vector< gpumat::GpuMat > vgW;		/// for delta weights
	std::vector< gpumat::GpuMat > vgB;		/// for delta bias
	std::vector< gpumat::GpuMat > Mask;		/// masks for bakward pass (created in forward pass)
	gpumat::Optimizer *m_optim;

	std::vector< gpumat::GpuMat > gW;		/// gradient for weights
	std::vector< gpumat::GpuMat > gB;		/// gradient for biases

	bool m_pool_dropout;
	double m_prob_dropout;

	convnn_gpu();

	void setOptimizer(Optimizer* optim);

	void setAlpha(double val);
	void setLambda(double val);

	void setDropout(bool val);
	void setDropout(double val);

	std::vector<gpumat::GpuMat> &XOut();
	/**
	 * @brief XOut1
	 * out after convolution
	 * @return
	 */
	std::vector< gpumat::GpuMat >& XOut1();
	/**
	 * @brief XOut2
	 * out after pooling
	 * @return
	 */
	std::vector< gpumat::GpuMat >& XOut2();

	bool use_pool() const;

	int outputFeatures() const;

	ct::Size szOut() const;

	void init(const ct::Size& _szA0, int _channels, int stride, int _K, const ct::Size& _szW, bool use_pool = true, bool use_transpose = true);

	void forward(const std::vector< gpumat::GpuMat >* _pX, gpumat::etypefunction func);

	void backcnv(const std::vector< gpumat::GpuMat >& D, std::vector< gpumat::GpuMat >& DS);

	void backward(const std::vector< gpumat::GpuMat >& D, bool last_level = false);

	void write(std::fstream& fs);
	void read(std::fstream& fs);

	void write2(std::fstream& fs);
	void read2(std::fstream& fs);

private:
	bool m_use_pool;
	gpumat::etypefunction m_func;
	gpumat::GpuMat m_Dropout;
	double m_lambda;

	gpumat::AdamOptimizer m_innet_optim;

	std::vector< gpumat::GpuMat > dSub2;
	std::vector< gpumat::GpuMat > Dc;		///
//	std::vector< gpumat::GpuMat > DA1;		///
	bool m_use_transpose;
};

/**
 * @brief im2cols
 * @param X
 * @param szA0
 * @param channels
 * @param szW
 * @param stride
 * @param Res
 * @param szOut
 */
void im2cols(const gpumat::GpuMat & X, const ct::Size& szA0, int channels, const ct::Size& szW,
			 int stride, gpumat::GpuMat & Res, ct::Size& szOut);

/**
 * @brief im2colsT
 * @param X
 * @param szA0
 * @param channels
 * @param szW
 * @param stride
 * @param Res
 * @param szOut
 */
void im2colsT(const gpumat::GpuMat & X, const ct::Size& szA0, int channels, const ct::Size& szW,
			 int stride, gpumat::GpuMat & Res, ct::Size& szOut);

/**
 * @brief im2cols
 * @param X
 * @param szA0
 * @param channels
 * @param szW
 * @param stride
 * @param Res
 * @param szOut
 */
void im2cols(const std::vector< gpumat::GpuMat > & X, const ct::Size& szA0, int channels, const ct::Size& szW,
			 int stride, std::vector< gpumat::GpuMat > & Res, ct::Size& szOut);

/**
 * @brief im2colsT
 * @param X
 * @param szA0
 * @param channels
 * @param szW
 * @param stride
 * @param Res
 * @param szOut
 */
void im2colsT(const std::vector< gpumat::GpuMat > & X, const ct::Size& szA0, int channels, const ct::Size& szW,
			 int stride, std::vector< gpumat::GpuMat > & Res, ct::Size& szOut);

/**
 * @brief back_deriv
 * @param Delta
 * @param szOut
 * @param szA0
 * @param channels
 * @param szW
 * @param stride
 * @param X
 */
void back_deriv(const gpumat::GpuMat& Delta, const ct::Size& szOut, const ct::Size& szA0,
				int channels, const ct::Size& szW, int stride, gpumat::GpuMat& X);

/**
 * @brief back_deriv
 * @param Delta
 * @param szOut
 * @param szA0
 * @param channels
 * @param szW
 * @param stride
 * @param X
 */
void back_deriv(const std::vector< gpumat::GpuMat >& Delta,
				const ct::Size& szOut,
				const ct::Size& szA0,
				int channels,
				const ct::Size& szW,
				int stride,
				std::vector< gpumat::GpuMat >& X);

/**
 * @brief back_derivT
 * @param Delta
 * @param szOut
 * @param szA0
 * @param channels
 * @param szW
 * @param stride
 * @param X
 */
void back_derivT(const gpumat::GpuMat& Delta, const ct::Size& szOut, const ct::Size& szA0,
				int channels, const ct::Size& szW, int stride, gpumat::GpuMat& X);

/**
 * @brief back_derivT
 * @param Delta
 * @param szOut
 * @param szA0
 * @param channels
 * @param szW
 * @param stride
 * @param X
 */
void back_derivT(const std::vector< gpumat::GpuMat >& Delta,
				const ct::Size& szOut,
				const ct::Size& szA0,
				int channels,
				const ct::Size& szW,
				int stride,
				std::vector< gpumat::GpuMat >& X);

/**
 * @brief subsample
 * @param X
 * @param szA
 * @param Y
 * @param Mask
 * @param szO
 */
void subsample(const GpuMat& X, const ct::Size& szA, GpuMat& Y, GpuMat& Mask, ct::Size& szO);

/**
 * @brief subsample
 * @param X
 * @param szA
 * @param Y
 * @param Mask
 * @param szO
 */
void subsample(const std::vector< GpuMat >& X, const ct::Size& szA, std::vector< GpuMat >& Y, std::vector< GpuMat >& Mask, ct::Size& szO);

void upsample(const GpuMat& Y,int K, const GpuMat& Mask, const ct::Size& szO,
			  const ct::Size& szA, GpuMat& X);

void upsample(const std::vector< GpuMat >& Y, int K, const std::vector< GpuMat >& Mask, const ct::Size& szO,
			  const ct::Size& szA, std::vector< GpuMat >& X);

/**
 * @brief vec2mat
 * @param vec
 * @param mat
 */
void vec2mat(const std::vector< GpuMat >& vec, GpuMat& mat);

/**
 * @brief mat2vec
 * @param mat
 * @param szOut
 * @param vec
 */
void mat2vec(const GpuMat& mat, const ct::Size& szOut, std::vector< GpuMat >& vec);


}

}

#endif // CONV2_GPU_H

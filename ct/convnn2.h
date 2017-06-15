#ifndef NN2_H
#define NN2_H

#include "custom_types.h"
#include "matops.h"
#include <vector>
#include "nn.h"

#include <exception>

namespace conv2{

template< typename T >
void im2col(const ct::Mat_<T>& X, const ct::Size& szA0, int channels, const ct::Size& szW,
			int stride, ct::Mat_<T>& Res, ct::Size& szOut)
{
	if(X.empty() || !channels)
		return;

	szOut.width = (szA0.width - szW.width)/stride + 1;
	szOut.height = (szA0.height - szW.height)/stride + 1;

	int rows = szOut.area();
	int cols = szW.area() * channels;

	Res.setSize(rows, cols);

	T *dX = X.ptr();
	T *dR = Res.ptr();
#pragma omp parallel for
	for(int c = 0; c < channels; ++c){
		T *dXi = &dX[c * szA0.area()];

		for(int y = 0; y < szOut.height; ++y){
			int y0 = y * stride;
			for(int x = 0; x < szOut.width; ++x){
				int x0 = x * stride;
				int row = y * szOut.width + x;

#ifdef __GNUC__
#pragma omp simd
#endif
				for(int a = 0; a < szW.height; ++a){
					for(int b = 0; b < szW.width; ++b){
						int col = c * szW.area() + (a * szW.width + b);
						if(y0 + a < szA0.height && x0 + b < szA0.width){
							dR[row * Res.cols + col] = dXi[(y0 + a) * szA0.width + (x0 + b)];
						}
					}
				}

			}
		}
	}
}

template< typename T >
void im2colT(const ct::Mat_<T>& X, const ct::Size& szA0, int channels, const ct::Size& szW,
			int stride, ct::Mat_<T>& Res, ct::Size& szOut)
{
	if(X.empty() || !channels)
		return;

	szOut.width = (szA0.width - szW.width)/stride + 1;
	szOut.height = (szA0.height - szW.height)/stride + 1;

	int rows = szOut.area();
	int cols = szW.area() * channels;

	Res.setSize(rows, cols);

	int colsX = channels;

	T *dR = Res.ptr();
#pragma omp parallel for
	for(int c = 0; c < channels; ++c){
		T *dXi = X.ptr() + c;

		for(int y = 0; y < szOut.height; ++y){
			int y0 = y * stride;
			for(int x = 0; x < szOut.width; ++x){
				int x0 = x * stride;
				int row = y * szOut.width + x;

#ifdef __GNUC__
#pragma omp simd
#endif
				for(int a = 0; a < szW.height; ++a){
					for(int b = 0; b < szW.width; ++b){
						int col = c * szW.area() + (a * szW.width + b);
						if(y0 + a < szA0.height && x0 + b < szA0.width){
							dR[row * Res.cols + col] = dXi[((y0 + a) * szA0.width + (x0 + b)) * colsX];
						}
					}
				}

			}
		}
	}
}

template< typename T >
void back_deriv(const ct::Mat_<T>& Delta, const ct::Size& szOut, const ct::Size& szA0,
				int channels, const ct::Size& szW, int stride, ct::Mat_<T>& X)
{
	if(Delta.empty() || !channels)
		return;

	X.setSize(channels, szA0.area());
	X.fill(0);

	T *dX = X.ptr();
	T *dR = Delta.ptr();
#pragma omp parallel for
	for(int c = 0; c < channels; ++c){
		T *dXi = &dX[c * szA0.area()];
		for(int y = 0; y < szOut.height; ++y){
			int y0 = y * stride;
			for(int x = 0; x < szOut.width; ++x){
				int x0 = x * stride;
				int row = y * szOut.width + x;

				for(int a = 0; a < szW.height; ++a){
#ifdef __GNUC__
#pragma omp simd
#endif
					for(int b = 0; b < szW.width; ++b){
						int col = c * szW.area() + (a * szW.width + b);
						if(y0 + a < szA0.height && x0 + b < szA0.width){
							dXi[(y0 + a) * szA0.width + (x0 + b)] += dR[row * Delta.cols + col];
						}
					}
				}

			}
		}
	}
}

template< typename T >
void back_derivT(const ct::Mat_<T>& Delta, const ct::Size& szOut, const ct::Size& szA0,
				int channels, const ct::Size& szW, int stride, ct::Mat_<T>& X)
{
	if(Delta.empty() || !channels)
		return;

	X.setSize(szA0.area(), channels);
	X.fill(0);

	T *dR = Delta.ptr();
#pragma omp parallel for
	for(int c = 0; c < channels; ++c){
		T *dXi = X.ptr() + c;
		for(int y = 0; y < szOut.height; ++y){
			int y0 = y * stride;
			for(int x = 0; x < szOut.width; ++x){
				int x0 = x * stride;
				int row = y * szOut.width + x;

				for(int a = 0; a < szW.height; ++a){
#ifdef __GNUC__
#pragma omp simd
#endif
					for(int b = 0; b < szW.width; ++b){
						int col = c * szW.area() + (a * szW.width + b);
						if(y0 + a < szA0.height && x0 + b < szA0.width){
							dXi[((y0 + a) * szA0.width + (x0 + b)) * channels] += dR[row * Delta.cols + col];
						}
					}
				}

			}
		}
	}
}

template< typename T >
void subsample(const ct::Mat_<T>& X, const ct::Size& szA, ct::Mat_<T>& Y, ct::Mat_<T>& Mask, ct::Size& szO)
{
	if(X.empty() || X.rows != szA.area())
		return;

	szO.width = szA.width / 2;
	szO.height = szA.height / 2;
	int K = X.cols;

	Y.setSize(szO.area(), K);
	Mask.setSize(X.size());
	Mask.fill(0);

	int stride = 2;

#pragma omp parallel for
	for(int k = 0; k < K; ++k){
		T *dX = X.ptr() + k;
		T* dM = Mask.ptr() + k;
		T *dY = Y.ptr() + k;

		for(int y = 0; y < szO.height; ++y){
			int y0 = y * stride;
			for(int x = 0; x < szO.width; ++x){
				int x0 = x * stride;

				T mmax = dX[(y0 * szA.width + x0) * X.cols];
				int xm = x0, ym = y0;
				T resM = 0;
#ifdef __GNUC__
#pragma omp simd
#endif
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
	}
}

template< typename T >
void upsample(const ct::Mat_<T>& Y, int K, const ct::Mat_<T>& Mask, const ct::Size& szO,
			  const ct::Size& szA, ct::Mat_<T>& X)
{
	if(Y.empty() || Mask.empty() || Y.total() != szO.area() * K)
		return;

	X.setSize(szA.area(), K);

	int stride = 2;

#pragma omp parallel for
	for(int k = 0; k < K; ++k){
		T *dX = X.ptr() + k;
		T* dM = Mask.ptr() + k;
		T *dY = Y.ptr() + k;

		for(int y = 0; y < szO.height; ++y){
			int y0 = y * stride;
			for(int x = 0; x < szO.width; ++x){
				int x0 = x * stride;

				T val = dY[(y * szO.width + x) * K];

#ifdef __GNUC__
#pragma omp simd
#endif
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
	}
}

template< typename T >
void vec2mat(const std::vector< ct::Mat_<T> >& vec, ct::Mat_<T>& mat)
{
	if(vec.empty() || vec[0].empty())
		return;

	int rows = (int)vec.size();
	int cols = vec[0].total();

	mat.setSize(rows, cols);

	T *dM = mat.ptr();

#pragma omp parallel for
	for(int i = 0; i < rows; ++i){
		const ct::Mat_<T>& V = vec[i];
		T *dV = V.ptr();
		for(int j = 0; j < V.total(); ++j){
			dM[i * cols + j] = dV[j];
		}
	}
}

template< typename T >
void mat2vec(const ct::Mat_<T>& mat, const ct::Size& szOut, std::vector< ct::Mat_<T> >& vec)
{
	if(mat.empty())
		return;

	int rows = mat.rows;
	int cols = mat.cols;

	vec.resize(rows);

	T *dM = mat.ptr();

#pragma omp parallel for
	for(int i = 0; i < rows; ++i){
		ct::Mat_<T>& V = vec[i];
		V.setSize(szOut);
		T *dV = V.ptr();
		for(int j = 0; j < V.total(); ++j){
			dV[j] = dM[i * cols + j];
		}
	}
}

template< typename T >
void flipW(const ct::Mat_<T>& W, const ct::Size& sz,int channels, ct::Mat_<T>& Wr)
{
	if(W.empty() || W.rows != sz.area() * channels)
		return;

	Wr.setSize(W.size());

#pragma omp parallel for
	for(int k = 0; k < W.cols; ++k){
		for(int c = 0; c < channels; ++c){
			T *dW = W.ptr() + c * sz.area() * W.cols + k;
			T *dWr = Wr.ptr() + c * sz.area() * W.cols + k;

#ifdef __GNUC__
#pragma omp simd
#endif
			for(int a = 0; a < sz.height; ++a){
				for(int b = 0; b < sz.width; ++b){
					dWr[((sz.height - a - 1) * sz.width + b) * W.cols] = dW[((a) * sz.width + b) * W.cols];
				}
			}

		}
	}
}

//-------------------------------------

template< typename T >
class convnn_abstract{
public:
	int K;									/// kernels
	int channels;							/// input channels

	ct::Size szA0;							/// input size
	ct::Size szA1;							/// size after convolution
	ct::Size szA2;							/// size after pooling
	ct::Size szK;							/// size of output data (set in forward)

	virtual std::vector< ct::Mat_<T> >& XOut() = 0;
	virtual int outputFeatures() const = 0;
	virtual ct::Size szOut() const = 0;
};

template< typename T >
class convnn: public convnn_abstract<T>{
public:
	std::vector< ct::Mat_<T> > W;			/// weights
	std::vector< ct::Mat_<T> > B;			/// biases
	int stride;
	ct::Size szW;							/// size of weights
	std::vector< ct::Mat_<T> >* pX;			/// input data
	std::vector< ct::Mat_<T> > Xc;			///
	std::vector< ct::Mat_<T> > A1;			/// out after appl nonlinear function
	std::vector< ct::Mat_<T> > A2;			/// out after pooling
	std::vector< ct::Mat_<T> > Dlt;			/// delta after backward pass
	std::vector< ct::Mat_<T> > vgW;			/// for delta weights
	std::vector< ct::Mat_<T> > vgB;			/// for delta bias
	std::vector< ct::Mat_<T> > Mask;		/// masks for bakward pass (created in forward pass)
	ct::Optimizer< T > *m_optim;
	ct::AdamOptimizer<T> m_adam;

	std::vector< ct::Mat_<T> > gW;			/// gradient for weights
	std::vector< ct::Mat_<T> > gB;			/// gradient for biases

	std::vector< ct::Mat_<T> > dSub;
	std::vector< ct::Mat_<T> > Dc;

	convnn(){
		m_use_pool = false;
		pX = nullptr;
		stride = 1;
		m_use_transpose = true;
		m_Lambda = 0;
		m_optim = &m_adam;
	}

	void setOptimizer(ct::Optimizer<T>* optim){
		if(!optim)
			return;
		m_optim = optim;
	}

	std::vector< ct::Mat_<T> >& XOut(){
		if(m_use_pool)
			return A2;
		return A1;
	}

	const std::vector< ct::Mat_<T> >& XOut() const{
		if(m_use_pool)
			return A2;
		return A1;
	}
	/**
	 * @brief XOut1
	 * out after convolution
	 * @return
	 */
	std::vector< ct::Mat_<T> >& XOut1(){
		return A1;
	}
	/**
	 * @brief XOut2
	 * out after pooling
	 * @return
	 */
	std::vector< ct::Mat_<T> >& XOut2(){
		return A2;
	}

	bool use_pool() const{
		return m_use_pool;
	}

	int outputFeatures() const{
		if(m_use_pool){
			int val = convnn_abstract<T>::szA2.area() * K;
			return val;
		}else{
			int val= convnn_abstract<T>::szA1.area() * K;
			return val;
		}
	}

	ct::Size szOut() const{
		if(m_use_pool)
			return convnn_abstract<T>::szA2;
		else
			return convnn_abstract<T>::szA1;
	}

	void setAlpha(T alpha){
		m_optim->setAlpha(alpha);
	}

	void setLambda(T val){
		m_Lambda = val;
	}

	void init(const ct::Size& _szA0, int _channels, int stride, int _K, const ct::Size& _szW,
			  bool use_pool = true, bool use_transpose = true){
		szW = _szW;
		m_use_pool = use_pool;
		m_use_transpose = use_transpose;
		convnn_abstract<T>::K = _K;
		convnn_abstract<T>::channels = _channels;
		convnn_abstract<T>::szA0 = _szA0;
		this->stride = stride;

		int rows = szW.area() * channels;
		int cols = K;

		ct::get_cnv_sizes(convnn_abstract<T>::szA0, szW, stride, convnn_abstract<T>::szA1, convnn_abstract<T>::szA2);

		T n = (T)1./szW.area();

		W.resize(1);
		B.resize(1);
		gW.resize(1);
		gB.resize(1);

		W[0].setSize(rows, cols);
		W[0].randn(0, n);
		B[0].setSize(1, K);
		B[0].randn(0, n);

		m_optim->init(W, B);

		printf("Out=[%dx%dx%d]\n", szOut().width, szOut().height, K);
	}

	void forward(const std::vector< ct::Mat_<T> >* _pX, ct::etypefunction func){
		if(!_pX)
			return;
		pX = (std::vector< ct::Mat_<T> >*)_pX;
		m_func = func;

		Xc.resize(pX->size());
		A1.resize(pX->size());

		if(m_use_transpose){
			for(size_t i = 0; i < Xc.size(); ++i){
				ct::Mat_<T>& Xi = (*pX)[i];
				ct::Size szOut;

				im2colT(Xi, convnn_abstract<T>::szA0, channels, szW, stride, Xc[i], szOut);
			}
		}else{
			for(size_t i = 0; i < Xc.size(); ++i){
				ct::Mat_<T>& Xi = (*pX)[i];
				ct::Size szOut;

				im2col(Xi, convnn_abstract<T>::szA0, channels, szW, stride, Xc[i], szOut);
			}
		}


		for(size_t i = 0; i < Xc.size(); ++i){
			ct::Mat_<T>& Xi = Xc[i];
			ct::Mat_<T>& A1i = A1[i];
			ct::matmul(Xi, W[0], A1i);
			A1i.biasPlus(B[0]);
		}

		for(size_t i = 0; i < A1.size(); ++i){
			ct::Mat_<T>& Ao = A1[i];
			switch (m_func) {
				case ct::RELU:
					ct::v_relu(Ao);
					break;
				case ct::SIGMOID:
					ct::v_sigmoid(Ao);
					break;
				case ct::TANH:
					ct::v_tanh(Ao);
					break;
				default:
					break;
			}
		}
		if(m_use_pool){
			Mask.resize(Xc.size());
			A2.resize(A1.size());
			for(size_t i = 0; i < A1.size(); ++i){
				ct::Mat_<T> &A1i = A1[i];
				ct::Mat_<T> &A2i = A2[i];
				ct::Size szOut;
				conv2::subsample(A1i, convnn_abstract<T>::szA1, A2i, Mask[i], szOut);
			}
			convnn_abstract<T>::szK = A2[0].size();
		}else{
			convnn_abstract<T>::szK = A1[0].size();
		}
	}

	void forward(const convnn<T> & conv, ct::etypefunction func){
		forward(&conv.XOut(), func);
	}

	inline void backcnv(const std::vector< ct::Mat_<T> >& D, std::vector< ct::Mat_<T> >& DS){
		if(D.data() != DS.data()){
			for(size_t i = 0; i < D.size(); ++i){
				switch (m_func) {
					case ct::RELU:
						ct::elemwiseMult(D[i], derivRelu(A1[i]), DS[i]);
						break;
					case ct::SIGMOID:
						ct::elemwiseMult(D[i], derivSigmoid(A1[i]), DS[i]);
						break;
					case ct::TANH:
						ct::elemwiseMult(D[i], derivTanh(A1[i]), DS[i]);
						break;
					default:
						break;
				}
			}
		}else{
			for(size_t i = 0; i < D.size(); ++i){
				switch (m_func) {
					case ct::RELU:
						ct::elemwiseMult(DS[i], ct::derivRelu(A1[i]));
						break;
					case ct::SIGMOID:
						ct::elemwiseMult(DS[i], ct::derivSigmoid(A1[i]));
						break;
					case ct::TANH:
						ct::elemwiseMult(DS[i], ct::derivTanh(A1[i]));
						break;
					default:
						break;
				}
			}
		}
	}

	void backward(const std::vector< ct::Mat_<T> >& D, bool last_level = false){
		if(D.empty() || D.size() != Xc.size()){
			throw new std::invalid_argument("vector D not complies saved parameters");
		}

		dSub.resize(D.size());

		//printf("1\n");
		if(m_use_pool){
			for(size_t i = 0; i < D.size(); ++i){
				ct::Mat_<T> Di = D[i];
				//Di.set_dims(szA2.area(), K);
				upsample(Di, K, Mask[i],convnn_abstract<T>:: szA2, convnn_abstract<T>::szA1, dSub[i]);
			}
			backcnv(dSub, dSub);
		}else{
			backcnv(D, dSub);
		}

		//printf("2\n");
		vgW.resize(D.size());
		vgB.resize(D.size());
//#pragma omp parallel for
		for(int i = 0; i < D.size(); ++i){
			ct::Mat_<T>& Xci = Xc[i];
			ct::Mat_<T>& dSubi = dSub[i];
			ct::Mat_<T>& Wi = vgW[i];
			ct::Mat_<T>& vgBi = vgB[i];
			matmulT1(Xci, dSubi, Wi);
			vgBi = ct::sumRows(dSubi, 1.f/dSubi.rows);
			//Wi *= (1.f/dSubi.total());
			//vgBi.swap_dims();
		}
		//printf("3\n");
		gW[0].setSize(W[0].size());
		gW[0].fill(0);
		gB[0].setSize(B[0].size());
		gB[0].fill(0);
		for(size_t i = 0; i < D.size(); ++i){
			ct::add(gW[0], vgW[i]);
			ct::add(gB[0], vgB[i]);
		}
		gW[0] *= (T)1./(D.size());
		gB[0] *= (T)1./(D.size());

		//printf("4\n");
		if(m_Lambda > 0){
			ct::add<float>(gW[0],  W[0], 1., (m_Lambda / K));
		}

		//printf("5\n");
		if(!last_level){
			Dlt.resize(D.size());

			//ct::Mat_<T> Wf;
			//flipW(W, szW, channels, Wf);

			Dc.resize(D.size());
			for(size_t i = 0; i < D.size(); ++i){
				ct::matmulT2(dSub[i], W[0], Dc[i]);
				back_derivT(Dc[i], convnn_abstract<T>::szA1, convnn_abstract<T>::szA0, channels, szW, stride, Dlt[i]);
				//ct::Size sz = (*pX)[i].size();
				//Dlt[i].set_dims(sz);
			}
		}

		//printf("6\n");
		m_optim->pass(gW, gB, W, B);

		//printf("7\n");
	}

	void write(std::fstream& fs){
		if(!W.size() || !B.size())
			return;
		ct::write_fs(fs, W[0]);
		ct::write_fs(fs, B[0]);
	}
	void read(std::fstream& fs){
		if(!W.size() || !B.size())
			return;
		ct::read_fs(fs, W[0]);
		ct::read_fs(fs, B[0]);
	}

private:
	bool m_use_pool;
	ct::etypefunction m_func;
	bool m_use_transpose;
	T m_Lambda;
};

template< typename T >
class Pooling: public convnn_abstract<T>{
public:
	std::vector< ct::Mat_<T> >* pX;			/// input data
	std::vector< ct::Mat_<T> > A2;			/// out after pooling
	std::vector< ct::Mat_<T> > Dlt;			/// delta after backward pass
	std::vector< ct::Mat_<T> > Mask;		/// masks for bakward pass (created in forward pass)
//	std::vector< ct::Mat_<T> > dSub;

	Pooling(){
		pX = nullptr;
		channels = 0;
		K = 0;
	}

	ct::Size szOut() const{
		return convnn_abstract<T>::szA2;
	}
	std::vector< ct::Mat_<T> >& XOut(){
		return A2;
	}
	std::vector< ct::Mat_<T> >* pXOut(){
		return &A2;
	}
	int outputFeatures() const{
			int val = szA2.area() * K;
			return val;
	}

	void init(const ct::Size& _szA0, int _channels, int _K){
		convnn_abstract<T>::K = _K;
		channels = _channels;
		convnn_abstract<T>::szA0 = _szA0;

		convnn_abstract<T>::szA2 = ct::Size(szA0.width/2, szA0.height/2);

		printf("Out=[%dx%dx%d]\n", szOut().width, szOut().height, convnn_abstract<T>::K);
	}

	void init(const convnn<T>& conv){
		convnn_abstract<T>::K = conv.K;
		convnn_abstract<T>::channels = conv.channels;
		convnn_abstract<T>::szA0 = conv.szOut();

		convnn_abstract<T>::szA2 = ct::Size(convnn_abstract<T>::szA0.width/2, convnn_abstract<T>::szA0.height/2);

		printf("Out=[%dx%dx%d]\n", szOut().width, szOut().height, convnn_abstract<T>::K);
	}

	void forward(const std::vector< ct::Mat_<T> >* _pX){
		if(!_pX)
			return;
		pX = (std::vector< ct::Mat_<T> >*)_pX;

		std::vector< ct::Mat_<T> >& A1 = pX;			/// out after appl nonlinear function
		Mask.resize(A1.size());
		A2.resize(A1.size());
		for(size_t i = 0; i < A1.size(); ++i){
			ct::Mat_<T> &A1i = A1[i];
			ct::Mat_<T> &A2i = A2[i];
			ct::Size szOut;
			conv2::subsample(A1i, szA0, A2i, Mask[i], szOut);
		}
		convnn_abstract<T>::szK = A2[0].size();
	}

	void forward(convnn<T> & conv){
		pX = &conv.XOut();
		std::vector< ct::Mat_<T> >& A1 = conv.XOut();			/// out after appl nonlinear function
		Mask.resize(A1.size());
		A2.resize(A1.size());
		for(size_t i = 0; i < A1.size(); ++i){
			ct::Mat_<T> &A1i = A1[i];
			ct::Mat_<T> &A2i = A2[i];
			ct::Size szOut;
			conv2::subsample(A1i, convnn_abstract<T>::szA0, A2i, Mask[i], szOut);
		}
		convnn_abstract<T>::szK = A2[0].size();
	}

	void backward(const std::vector< ct::Mat_<T> >& D){
		if(D.empty() || D.size() != pX->size()){
			throw new std::invalid_argument("vector D not complies saved parameters");
		}

		Dlt.resize(D.size());

		for(size_t i = 0; i < D.size(); ++i){
			ct::Mat_<T> Di = D[i];
			//Di.set_dims(szA2.area(), K);
			upsample(Di, convnn_abstract<T>::K, Mask[i], convnn_abstract<T>::szA2, convnn_abstract<T>::szA0, Dlt[i]);
		}
	}

};

template< typename T >
class Concat{
public:
	ct::Mat_<T> m_A1;
	ct::Mat_<T> m_A2;
	ct::Matf D1;
	ct::Matf D2;
	std::vector< ct::Matf > Dlt1;
	std::vector< ct::Matf > Dlt2;

	ct::Mat_<T> Y;

	convnn_abstract<T>* m_c1;
	convnn_abstract<T>* m_c2;

	Concat(){

	}

	void forward(convnn_abstract<T>* c1, convnn_abstract<T>* c2){
		if(!c1 || !c2)
			return;

		m_c1 = c1;
		m_c2 = c2;

		conv2::vec2mat(c1->XOut(), m_A1);
		conv2::vec2mat(c2->XOut(), m_A2);

		std::vector< ct::Matf* > concat;

		concat.push_back(&m_A1);
		concat.push_back(&m_A2);

		ct::hconcat(concat, Y);
	}
	void backward(const ct::Mat_<T>& Dlt){
		if(!m_c1 || !m_c2)
			return;

		std::vector< int > cols;
		std::vector< ct::Matf* > mats;
		cols.push_back(m_c1->outputFeatures());
		cols.push_back(m_c2->outputFeatures());
		mats.push_back(&D1);
		mats.push_back(&D2);
		ct::hsplit(Dlt, cols, mats);

		conv2::mat2vec(D1, m_c1->szK, Dlt1);
		conv2::mat2vec(D2, m_c2->szK, Dlt2);
	}
};

}

#endif // NN2_H

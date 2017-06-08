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
class convnn{
public:
	ct::Mat_<T> W;							/// weights
	ct::Mat_<T> B;							/// biases
	int K;									/// kernels
	int channels;							/// input channels
	int stride;
	ct::Size szA0;							/// input size
	ct::Size szA1;							/// size after convolution
	ct::Size szA2;							/// size after pooling
	ct::Size szW;							/// size of weights
	ct::Size szK;							/// size of output data (set in forward)
	std::vector< ct::Mat_<T> >* pX;			/// input data
	std::vector< ct::Mat_<T> > Xc;			///
	std::vector< ct::Mat_<T> > A1;			/// out after appl nonlinear function
	std::vector< ct::Mat_<T> > A2;			/// out after pooling
	std::vector< ct::Mat_<T> > Dlt;			/// delta after backward pass
	std::vector< ct::Mat_<T> > vgW;			/// for delta weights
	std::vector< ct::Mat_<T> > vgB;			/// for delta bias
	std::vector< ct::Mat_<T> > Mask;		/// masks for bakward pass (created in forward pass)
	ct::AdamOptimizer< T > m_optim;

	ct::Mat_<T> gW;							/// gradient for weights
	ct::Mat_<T> gB;							/// gradient for biases

	convnn(){
		m_use_pool = false;
		pX = nullptr;
		stride = 1;
		m_use_transpose = true;
		m_Lambda = 0;
	}

	std::vector< ct::Mat_<T> >& XOut(){
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
			int val = szA2.area() * K;
			return val;
		}else{
			int val= szA1.area() * K;
			return val;
		}
	}

	ct::Size szOut() const{
		if(m_use_pool)
			return szA2;
		else
			return szA1;
	}

	void setAlpha(T alpha){
		m_optim.setAlpha(alpha);
	}

	void setLambda(T val){
		m_Lambda = val;
	}

	void init(const ct::Size& _szA0, int _channels, int stride, int _K, const ct::Size& _szW,
			  bool use_pool = true, bool use_transpose = true){
		szW = _szW;
		K = _K;
		channels = _channels;
		m_use_pool = use_pool;
		m_use_transpose = use_transpose;
		szA0 = _szA0;
		this->stride = stride;

		int rows = szW.area() * channels;
		int cols = K;

		ct::get_cnv_sizes(szA0, szW, stride, szA1, szA2);

		T n = (T)1./szW.area();

		W.setSize(rows, cols);
		W.randn(0, n);
		B.setSize(1, K);
		B.randn(0, n);

		std::vector< ct::Mat_<T> > vW, vB;
		vW.push_back(W);
		vB.push_back(B);
		m_optim.init(vW, vB);
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

				im2colT(Xi, szA0, channels, szW, stride, Xc[i], szOut);
			}
		}else{
			for(size_t i = 0; i < Xc.size(); ++i){
				ct::Mat_<T>& Xi = (*pX)[i];
				ct::Size szOut;

				im2col(Xi, szA0, channels, szW, stride, Xc[i], szOut);
			}
		}


		for(size_t i = 0; i < Xc.size(); ++i){
			ct::Mat_<T>& Xi = Xc[i];
			ct::Mat_<T>& A1i = A1[i];
			A1i = Xi * W;
			A1i.biasPlus(B);
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
				conv2::subsample(A1i, szA1, A2i, Mask[i], szOut);
			}
			szK = A2[0].size();
		}else{
			szK = A1[0].size();
		}
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

		std::vector< ct::Mat_<T> > dSub;
		dSub.resize(D.size());

		//printf("1\n");
		if(m_use_pool){
			for(size_t i = 0; i < D.size(); ++i){
				ct::Mat_<T> Di = D[i];
				//Di.set_dims(szA2.area(), K);
				upsample(Di, K, Mask[i], szA2, szA1, dSub[i]);
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
			vgBi = (ct::sumRows(dSubi)) * (1.f/dSubi.rows);
			//Wi *= (1.f/dSubi.total());
			//vgBi.swap_dims();
		}
		//printf("3\n");
		gW.setSize(W.size());
		gW.fill(0);
		gB.setSize(B.size());
		gB.fill(0);
		for(size_t i = 0; i < D.size(); ++i){
			gW += vgW[i];
			gB += vgB[i];
		}
		gW *= (T)1./(D.size());
		gB *= (T)1./(D.size());

		//printf("4\n");
		if(m_Lambda > 0){
			gW += W * (m_Lambda / K);
		}

		//printf("5\n");
		if(!last_level){
			Dlt.resize(D.size());

			//ct::Mat_<T> Wf;
			//flipW(W, szW, channels, Wf);

			for(size_t i = 0; i < D.size(); ++i){
				ct::Mat_<T> Dc;
				ct::matmulT2(dSub[i], W, Dc);
				back_derivT(Dc, szA1, szA0, channels, szW, stride, Dlt[i]);
				//ct::Size sz = (*pX)[i].size();
				//Dlt[i].set_dims(sz);
			}
		}

		//printf("6\n");
		std::vector< ct::Mat_<T>> vgW, vgB, vW, vB;
		vgW.push_back(gW);
		vW.push_back(W);
		vgB.push_back(gB);
		vB.push_back(B);

		m_optim.pass(vgW, vgB, vW, vB);
		W = vW[0]; B = vB[0];

		//printf("7\n");
	}

	void write(std::fstream& fs){
		ct::write_fs(fs, W);
		ct::write_fs(fs, B);
	}
	void read(std::fstream& fs){
		ct::read_fs(fs, W);
		ct::read_fs(fs, B);
	}

private:
	bool m_use_pool;
	ct::etypefunction m_func;
	bool m_use_transpose;
	T m_Lambda;
};

}

#endif // NN2_H

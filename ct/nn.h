#ifndef NN_H
#define NN_H

#include "custom_types.h"
#include "matops.h"

#include <vector>
#include <exception>

#ifndef __GNUC__
typedef unsigned int uint;
#endif

namespace ct{

template< typename T >
inline T sqr(T val)
{
	return val * val;
}

template< typename T >
class Optimizer{
public:
	Optimizer(){
		m_alpha = (T)0.001;
		m_iteration = 0;
	}
	T alpha()const{
		return m_alpha;
	}
	void setAlpha(T v){
		m_alpha = v;
	}
	uint32_t iteration() const{
		return m_iteration;
	}
	virtual bool init(const std::vector< ct::Mat_<T> >& W, const std::vector< ct::Mat_<T> >& B){
		return false;
	}

	virtual bool pass(const std::vector< ct::Mat_< T > >& gradW, const std::vector< ct::Mat_< T > >& gradB,
			  std::vector< ct::Mat_<T> >& W, std::vector< ct::Mat_<T> >& b){
		return false;
	}

protected:
	T m_alpha;
	uint32_t m_iteration;
};

/**
 * @brief The AdamOptimizer class
 */
template< typename T >
class AdamOptimizer: public Optimizer<T>{
public:
	AdamOptimizer(): Optimizer<T>(){
		m_betha1 = (T)0.9;
		m_betha2 = (T)0.999;
		m_init = false;
	}

	T betha1() const{
		return m_betha1;
	}

	void setBetha1(T v){
		m_betha1 = v;
	}

	T betha2() const{
		return m_betha2;
	}

	void setBetha2(T v){
		m_betha2 = v;
	}

	bool init(const std::vector< ct::Mat_<T> >& W, const std::vector< ct::Mat_<T> >& B){
		if(W.empty() || B.empty())
			return false;

		using namespace ct;

		Optimizer<T>::m_iteration = 0;

		m_mW.resize(W.size());
		m_mb.resize(W.size());

		m_vW.resize(W.size());
		m_vb.resize(W.size());

		for(size_t i = 0; i < W.size(); i++){

			m_mW[i].setSize(W[i].size());
			m_vW[i].setSize(W[i].size());
			m_mW[i].fill(0);
			m_vW[i].fill(0);

			m_mb[i].setSize(B[i].size());
			m_vb[i].setSize(B[i].size());
			m_mb[i].fill(0);
			m_vb[i].fill(0);
		}
		m_init = true;
		return true;
	}

	bool pass(const std::vector< ct::Mat_< T > >& gradW, const std::vector< ct::Mat_< T > >& gradB,
			  std::vector< ct::Mat_<T> >& W, std::vector< ct::Mat_<T> >& b){
		if(!gradW.size() || gradW.size() != gradB.size() || gradW.size() != W.size())
			return false;

		using namespace ct;

		Optimizer<T>::m_iteration++;
		T sb1 = (T)(1. / (1. - pow(m_betha1, Optimizer<T>::m_iteration)));
		T sb2 = (T)(1. / (1. - pow(m_betha2, Optimizer<T>::m_iteration)));
		T eps = (T)(10e-8);

		for(size_t i = 0; i < gradW.size(); ++i){
			m_mW[i] = m_betha1 * m_mW[i] + (T)(1. - m_betha1) * gradW[i];
			m_mb[i] = m_betha1 * m_mb[i] + (T)(1. - m_betha1) * gradB[i];

			m_vW[i] = m_betha2 * m_vW[i] + (T)(1. - m_betha2) * elemwiseSqr(gradW[i]);
			m_vb[i] = m_betha2 * m_vb[i] + (T)(1. - m_betha2) * elemwiseSqr(gradB[i]);

			Mat_<T> mWs = m_mW[i] * sb1;
			Mat_<T> mBs = m_mb[i] * sb1;
			Mat_<T> vWs = m_vW[i] * sb2;
			Mat_<T> vBs = m_vb[i] * sb2;

			vWs.sqrt(); vBs.sqrt();
			vWs += eps; vBs += eps;
			mWs = elemwiseDiv(mWs, vWs);
			mBs = elemwiseDiv(mBs, vBs);

			W[i] -= Optimizer<T>::m_alpha * mWs;
			b[i] -= Optimizer<T>::m_alpha * mBs;
		}
		return true;
	}
	bool pass(const std::vector< ct::Mat_< T > >& gradW, const std::vector< T >& gradB,
			  std::vector< ct::Mat_<T> >& W, std::vector< T >& b){
		if(!gradW.size() || gradW.size() != gradB.size() || gradW.size() != W.size())
			return false;

		if(!m_init){
			init_simple(gradW.size(), gradW[0].size());
		}

		using namespace ct;

		Optimizer<T>::m_iteration++;
		T sb1 = (T)(1. / (1. - pow(m_betha1, Optimizer<T>::m_iteration)));
		T sb2 = (T)(1. / (1. - pow(m_betha2, Optimizer<T>::m_iteration)));
		T eps = (T)(10e-8);

		for(size_t i = 0; i < gradW.size(); ++i){
			m_mW[i] = m_betha1 * m_mW[i] + (T)(1. - m_betha1) * gradW[i];
			m_mbn[i] = m_betha1 * m_mbn[i] + (T)(1. - m_betha1) * gradB[i];

			m_vW[i] = m_betha2 * m_vW[i] + (T)(1. - m_betha2) * elemwiseSqr(gradW[i]);
			m_vbn[i] = m_betha2 * m_vbn[i] + (T)(1. - m_betha2) * sqr(gradB[i]);

			Mat_<T> mWs = m_mW[i] * sb1;
			T mBs = m_mbn[i] * sb1;
			Mat_<T> vWs = m_vW[i] * sb2;
			T vBs = m_vbn[i] * sb2;

			vWs.sqrt(); vBs = std::sqrt(vBs);
			vWs += eps; vBs += eps;
			mWs = elemwiseDiv(mWs, vWs);
			mBs /= vBs;

			W[i] -= Optimizer<T>::m_alpha * mWs;
			b[i] -= Optimizer<T>::m_alpha * mBs;
		}
		return true;
	}
	bool empty() const{
		return m_mW.empty() || m_mb.empty() || m_vW.empty() || m_vb.empty();
	}


protected:
	T m_betha1;
	T m_betha2;
	bool m_init;

	std::vector< ct::Mat_<T> > m_mW;
	std::vector< ct::Mat_<T> > m_mb;
	std::vector< ct::Mat_<T> > m_vW;
	std::vector< ct::Mat_<T> > m_vb;
	std::vector< T > m_mbn;
	std::vector< T > m_vbn;

	void init_simple(size_t len, const ct::Size& sz){
		m_mW.resize(len);
		m_vW.resize(len);
		m_mbn.resize(len);
		m_vbn.resize(len);
		for(size_t i = 0; i < len; ++i){
			m_mW[i].setSize(sz.height, sz.width);
			m_mW[i].fill(0);
			m_vW[i].setSize(sz.height, sz.width);
			m_vW[i].fill(0);
			m_mbn[i] = 0;
			m_vbn[i] = 0;
		}
		m_init = true;
	}
};

/**
 * @brief The MomentOptimizer class
 */
template< typename T >
class MomentOptimizer: public Optimizer<T>{
public:
	MomentOptimizer(): Optimizer<T>(){
		Optimizer<T>::m_alpha = T(0.01);
		m_betha = T(0.9);
	}

	void setBetha(T val){
		m_betha = val;
	}

	void pass(const std::vector< ct::Mat_<T> > &gradW, const std::vector< T > &gradB,
			  std::vector< ct::Mat_<T> > &W, std::vector< T > &B)
	{
		if(W.empty() || gradW.size() != W.size() || gradB.empty() || gradB.size() != gradW.size())
			throw new std::invalid_argument("MomentOptimizer: wrong parameters");
		if(m_mW.empty()){
			m_mW.resize(W.size());
			m_mb.resize(W.size());
			for(int i = 0; i < m_mW.size(); ++i){
				m_mW[i] = ct::Mat_<T>::zeros(W[i].rows, W[i].cols);
				m_mb[i] = 0;
			}
		}

		for(int i = 0; i < m_mW.size(); ++i){
			ct::Mat_<T> tmp = m_mW[i];
			tmp *= m_betha;
			tmp += (1.f - m_betha) * gradW[i];
			m_mW[i] = tmp;

			m_mb[i] = m_betha * m_mb[i] + (1.f - m_betha) * gradB[i];
		}
		for(int i = 0; i < m_mW.size(); ++i){
			W[i] += ((-Optimizer<T>::m_alpha) * m_mW[i]);
			B[i] += ((-Optimizer<T>::m_alpha) * m_mb[i]);
		}
	}

protected:
	std::vector< ct::Mat_<T> > m_mW;
	std::vector< ct::Mat_<T> > m_mb;

	T m_betha;
};

/**
 * @brief The StohasticGradientOptimizer class
 */
template< typename T >
class StohasticGradientOptimizer: public Optimizer<T>{
public:
	StohasticGradientOptimizer(): Optimizer<T>(){

	}
	bool pass(const std::vector<ct::Mat_<T> > &gradW, const std::vector<ct::Mat_<T> > &gradB,
			  std::vector<ct::Mat_<T> > &W, std::vector<ct::Mat_<T> > &b)
	{
		if(W.empty() || gradW.size() != W.size() || gradB.empty() || gradB.size() != gradW.size())
			throw new std::invalid_argument("StohasticGradientOptimizer: wrong parameters");
		for(size_t i = 0; i < W.size(); ++i){
			W[i] -= Optimizer<T>::m_alpha * gradW[i];
			b[i] -= Optimizer<T>::m_alpha * gradB[i];
		}
		return true;
	}
};


/**
 * @brief The SimpleAutoencoder class
 */
template<class T>
class SimpleAutoencoder
{
public:

	typedef ct::Mat_<T> (*tfunc)(const ct::Mat_<T>& t);

	SimpleAutoencoder(){
		func = 0;
		deriv = 0;
		m_neurons = 0;
	}

	T m_alpha;
	int m_neurons;

	std::vector< ct::Mat_<T> > W;
	std::vector< ct::Mat_<T> > b;
	std::vector< ct::Mat_<T> > dW, db;

	tfunc func;
	tfunc deriv;

	void init(ct::Mat_<T>& _W, ct::Mat_<T>& _b, int samples, int neurons, tfunc fn, tfunc dfn){
		using namespace ct;

		func = fn;
		deriv = dfn;
		m_neurons = neurons;

		std::vector< int > layers;
		layers.push_back(neurons);
		layers.push_back(samples);

		W.resize(2);
		b.resize(2);

		dW.resize(2);
		db.resize(2);

		W[0] = _W;
		b[0] = _b;

		W[1] = _W.t();
		b[1] = Mat_<T>::zeros(samples, 1);

		adam.init(W, b);
//		W[0].randn(0, 0.1, 1);
//		b[0].randn(0, 0.1, 1);
//		W[1].randn(0, 0.1, 1);
//		b[1].randn(0, 0.1, 1);
	}

	void pass(const ct::Mat_<T>& X){
		if(X.empty() || X.cols != W[0].rows || !func || !deriv)
			return;
		using namespace ct;

		Mat_<T> a[3];
		Mat_<T> z[2], d, di, sz;
		a[0] = X;
		for(int i = 0; i < 2; i++){
			z[i] = a[i] * W[i];
			z[i].biasPlus(b[i]);
			if(i == 0){
				a[i + 1] = (*func)(z[i]);
			}else{
				a[i + 1] = z[i];
			}
		}

		T m = (T)X.rows;

		d = a[2] - X;

		for(int i = 1; i > -1; --i){
			if(i > 0){
				sz = (*deriv)(a[i]);
				matmulT2(d, W[i], di);
				elemwiseMult(di, sz);
			}
			matmulT1(a[i], d, dW[i]);
			dW[i] *= (T)(1./m);
			db[i] = (sumRows(d) * (T)(1./m)).t();
			if(i > 0)
				d = di;
		}
		dW[0] += dW[1].t();
		dW[1] = dW[0].t();
		db[1] = Mat_<T>::zeros(db[1].size());

		adam.pass(dW, db, W, b);
	}
	T l2(const ct::Mat_<T>& X) const{
		using namespace ct;

		if(X.empty() || W[0].empty())
			return -1.;

		Mat_<T> a[3];
		Mat_<T> z[2], d;
		a[0] = X;
		for(int i = 0; i < 2; i++){
			z[i] = a[i] * W[i];
			z[i].biasPlus(b[i]);
			if(i == 0){
				a[i + 1] = (*func)(z[i]);
			}else{
				a[i + 1] = z[i];
			}
		}
		T m = (T)X.rows;
		d = a[2] - X;
		elemwiseMult(d, d);
		T res = d.sum() / m;
		return res;
	}
	AdamOptimizer<T> adam;
private:
};

/**
 * @brief get_cnv_sizes
 * @param sizeIn
 * @param szW
 * @param stride
 * @param szA1
 * @param szA2
 */
void get_cnv_sizes(const ct::Size sizeIn, const ct::Size szW, int stride, ct::Size& szA1, ct::Size &szA2);

template<typename T>
inline T linear_func(T val)
{
	return val;
}

template< typename T >
inline T apply_func(T val, ct::etypefunction func)
{
	switch (func) {
		case ct::LINEAR:
			return val;
		case ct::RELU:
			return std::max(val, T(0));
		case ct::SIGMOID:
			return (T)(1. / (1. + std::exp(-val)));
		case ct::TANH:{
			T e = std::exp(2 * val);
			return (T)((e - 1.)/(e + 1.));
		}
		default:
			throw new std::invalid_argument("this function not applied");
			break;
	}
}

/**
 * @brief conv2D
 * @param A0
 * @param szI
 * @param stride
 * @param W
 * @param B
 * @param A1
 * @param func
 * @return
 */
template< typename T >
ct::Size conv2D(const ct::Mat_<T>& A0,
				const ct::Size& szI,
				int stride,
				const std::vector< ct::Mat_<T> >& W,
				const std::vector< T > B,
				std::vector< ct::Mat_<T> > &A1,
				ct::etypefunction func)
{
	if(A0.empty() || W.empty()){
		std::cout << "conv2D wrong parameters\n";
		return ct::Size(0, 0);
	}

	int w_rows = W[0].rows;
	int w_cols = W[0].cols;

	int m = A0.rows;

	ct::Size szO;
	szO.width	= (szI.width - w_cols + 1) / stride;
	szO.height	= (szI.height - w_rows + 1) / stride;

	int sz = szO.area();

	A1.resize(W.size());
	for(size_t i = 0; i < A1.size(); ++i)
		A1[i].setSize(A0.rows, sz);

#pragma omp parallel for
	for(int i = 0; i < m; ++i){
		const T *dA0i = &A0.at(i);

#pragma omp parallel for
		for(int y_res = 0; y_res < szO.height; y_res++){
			int y = y_res * stride;

#ifdef __GNUC__
#pragma omp simd
#else
#pragma omp parallel for
#endif
			for(int x_res = 0; x_res < szO.width; x_res++){
				int x = x_res * stride;

				for(size_t w = 0; w < W.size(); ++w){
					T *dA1i = &(A1[w].at(i));
					T *dW = W[w].ptr();
					T sum = 0;

					for(int a = 0; a < w_rows; ++a){
						if(y + a < szI.height){
							for(int b = 0; b < w_cols; ++b){
								if(x + b < szI.width){
									T w = dW[a * w_cols + b];
									T g = dA0i[(y + a) * szI.width + x + b];
									sum += w * g;
								}
							}
						}
					}
					sum += B[w];
					dA1i[y_res * szO.width + x_res] = apply_func(sum, func);
				}
			}
		}
	}
	return szO;
}

/**
 * @brief subsample
 * @param A0
 * @param szA0
 * @param A1
 * @param Mask
 * @param szA1
 * @return
 */
template< typename T >
bool subsample(const ct::Mat_<T> &A0,
			   const ct::Size& szA0,
			   ct::Mat_<T>& A1,
			   ct::Mat_<T>& Mask,
			   ct::Size& szA1)
{
	if(A0.empty())
		return false;

	int rows = A0.rows;
	int cols = A0.cols;

	if(!rows || !cols)
		return false;

	szA1 = ct::Size(szA0.width/2, szA0.height/2);

	A1.setSize(rows, szA1.area());
	Mask.setSize(rows, szA0.area());
	Mask.fill(0);
//	indexes.setSize(rows, cols);

	int kLen = 2;

#pragma omp parallel for
	for(int i = 0; i < rows; ++i){
		const T* dA0i = &A0.at(i);
		T* dA1i = &A1.at(i);
		T* dMi = &Mask.at(i);
#pragma omp parallel for
		for(int y = 0; y < szA1.height; ++y){
			int y0 = y * kLen;

#ifdef __GNUC__
#pragma omp simd
#else
#pragma omp parallel for
#endif
			for(int x = 0; x < szA1.width; ++x){
				int x0 = x * kLen;

				int xm = -1, ym = -1;
				T maxV = T(-99999999);
				for(int a = 0; a < kLen; ++a){
					for(int b = 0; b < kLen; ++b){
						if(y0 + a < szA0.height && x0 + b < szA0.width){
							T v = dA0i[(y0 + a) * szA0.width + (x0 + b)];
							if(v > maxV){
								maxV = v;
								xm = x0 + b, ym = y0 + a;
							}
						}
					}
				}
				if(xm < 0 || ym < 0)
					continue;
				dA1i[y * szA1.width + x] = maxV;
				dMi[ym * szA0.width + xm] = T(1);
			}
		}
	}

	return true;
}

/**
 * @brief subsample
 * @param A0
 * @param szA0
 * @param A1
 * @param Masks
 * @param szA1
 * @return
 */
template< typename T >
bool subsample(const std::vector< ct::Mat_<T> > &A0,
			   const ct::Size& szA0, std::vector< ct::Mat_<T> > &A1,
			   std::vector< ct::Mat_<T> > &Masks,
			   ct::Size& szA1)
{
	if(A0.empty())
		return false;
	A1.resize(A0.size());
	Masks.resize(A0.size());

	for(size_t i = 0; i < A0.size(); i++){
		if(!subsample(A0[i], szA0, A1[i], Masks[i], szA1))
			return false;
	}
	return true;
}

/**
 * @brief attach_vector
 * @param attached
 * @param slice
 */
template< typename T >
void attach_vector(std::vector< T >& attached, const std::vector< T >& slice)
{
	for(int i = 0; i < slice.size(); i++)
		attached.push_back(slice[i]);
}

/**
 * @brief upsample
 * @param A1
 * @param szA1
 * @param szA0
 * @param Mask
 * @param A0
 * @return
 */
template< typename T >
bool upsample(const ct::Mat_<T> &A1,
			  const ct::Size& szA1,
			  const ct::Size& szA0,
			  const ct::Mat_<T> &Mask,
			  ct::Mat_<T>& A0)
{
	if(A1.empty() || Mask.empty())
		return false;

	int m = A1.rows;

	A0.setSize(m, szA0.area());

	int kLen = 2;

#pragma omp parallel for
	for(int i = 0; i < m; ++i){
		const T *dA1i = &A1.at(i);
		T *dA0i = &A0.at(i);

#pragma omp parallel for
		for(int y = 0; y < szA1.height; ++y){
			int y0 = y * kLen;
#ifdef __GNUC__
#pragma omp simd
#else
#pragma omp parallel for
#endif
			for(int x = 0; x < szA1.width; ++x){
				int x0 = x * kLen;

				T v = dA1i[y * szA1.width + x];
				for(int a = 0; a < kLen; ++a){
					for(int b = 0; b < kLen; ++b){
						if(y0 + a < szA0.height && x0 + b < szA0.width){
							dA0i[(y0 + a) * szA0.width + (x0 + b)] = v;
						}
					}
				}
			}
		}
	}

//	ct::save_mat(Mask, "Mask.txt");
//	ct::save_mat(A0, "A0_before.txt");
	ct::elemwiseMult(A0, Mask);
//	ct::save_mat(A0, "A0_after.txt");

	return true;
}

/**
 * @brief upsample
 * @param A1
 * @param szA1
 * @param szA0
 * @param Masks
 * @param A0
 * @return
 */
template< typename T >
bool upsample(const std::vector< ct::Mat_<T> > &A1,
			  ct::Size& szA1,
			  const ct::Size& szA0,
			  const std::vector< ct::Mat_<T> > &Masks,
			  std::vector< ct::Mat_<T> >& A0, int first = -1, int last = -1)
{
	if(A1.empty() || Masks.empty())
		return false;
	if(first >= 0 && last > first){
		A0.resize(last - first);
	}else{
		A0.resize(A1.size());
		first = 0;
		last = (int)A1.size();
	}

	for(int i = first, j = 0; i < last; ++i, ++j){
		if(!upsample(A1[i], szA1, szA0, Masks[j], A0[j]))
			return false;
	}
	return true;
}


/**
 * @brief hconcat
 * @param list
 * @param res
 */
template< typename T >
void hconcat(const std::vector< ct::Mat_<T> >& list, ct::Mat_<T>& res)
{
	if(list.empty())
		return;
	int rows		= list[0].rows;
	int loc_cols	= list[0].cols;
	int cols		= loc_cols * (int)list.size();

	res.setSize(rows, cols);

	T *dR = res.ptr();

#pragma omp parallel for
	for(int i = 0; i < rows; ++i){
#pragma omp parallel for
		for(int j = 0; j < (int)list.size(); ++j){
			T* dL = list[j].ptr();
#ifdef __GNUC__
#pragma omp simd
#endif
			for(int k = 0; k < loc_cols; ++k){
				dR[i * cols + j * loc_cols + k] = dL[i * loc_cols + k];
			}
		}
	}
}

/**
 * @brief hsplit
 * @param res
 * @param cols
 * @param list
 */
template< typename T >
void hsplit(const ct::Mat_<T>& res, size_t cols, std::vector< ct::Mat_<T> >& list)
{
	if(res.empty() || (res.cols % cols) != 0)
		throw new std::invalid_argument("hsplit: wrong parameters");

	size_t len = res.cols / cols;

	list.resize(cols);

	for(size_t i = 0; i < cols; ++i){
		list[i].setSize(res.rows, (int)len);
	}

	T *dR = res.ptr();
#pragma omp parallel for
	for(int i = 0; i < (int)cols; ++i){
		T *dLi = list[i].ptr();
#ifdef __GNUC__
#pragma omp simd
#else
#pragma omp parallel for
#endif
		for(int j = 0; j < res.rows; ++j){
			for(size_t k = 0; k < len; ++k){
				dLi[j * len + k] = dR[j * res.cols + i * len + k];
			}
		}
	}
}

/**
 * @brief deriv_conv2D
 * @param A0
 * @param gradA1
 * @param szA0
 * @param szA1
 * @param szW
 * @param stride
 * @param gradW
 * @param gradB
 */
template< typename T >
void deriv_conv2D(const ct::Mat_<T>& A0,
				  const ct::Mat_<T>& gradA1,
				  const ct::Size& szA0,
				  const ct::Size& szA1,
				  const ct::Size &szW,
				  int stride,
				  ct::Mat_<T> &gradW,
				  T &gradB)
{
	if(A0.empty() || gradA1.empty() || !stride){
		std::cout << "deriv_conv2D wrong parameters\n";
	}

	gradW = ct::Mat_<T>::zeros(szW.height, szW.width);
	gradB = 0;

	int m = A0.rows;

	T *dA = A0.ptr();
	T *dgA1 = gradA1.ptr();
	T *dgW = gradW.ptr();

	for(int i = 0; i < m; ++i){
		T *dAi		= &dA[A0.cols * i];
		T *dgA1i	= &dgA1[gradA1.cols * i];

		for(int y = 0; y < szA1.height; ++y){
			int y0 = y * stride;
			for(int x = 0; x < szA1.width; ++x){
				int x0 = x * stride;
				T d = dgA1i[szA1.width * y + x];

#ifdef __GNUC__
#pragma omp simd
#else
#pragma omp parallel for
#endif
				for(int a = 0; a < szW.height; ++a){
					int y1 = y0 + a;
					if(y1 < szA0.height){
						for(int b = 0; b < szW.width; ++b){
							int x1 = x0 + b;
							if(x1 < szA0.width){
								T a0 = dAi[szA0.width * y1 + x1];
								dgW[a * szW.width + b] += a0 * d;
							}
						}
					}
				}
			}
		}
	}

	gradW *= (T)(1./m);

	gradB = 0;
	for(int i = 0; i < gradA1.total(); ++i){
		gradB += dgA1[i];
	}
	gradB /= (T)gradA1.total();
}

/**
 * @brief deriv_conv2D
 * @param A0
 * @param gradA1
 * @param szA0
 * @param szA1
 * @param szW
 * @param stride
 * @param gradW
 * @param gradB
 */
template< typename T >
void deriv_conv2D(const ct::Mat_<T> & A0,
				  const std::vector< ct::Mat_<T> >& gradA1,
				  const ct::Size& szA0,
				  const ct::Size& szA1,
				  const ct::Size &szW,
				  int stride,
				  std::vector< ct::Mat_<T> > &gradW,
				  std::vector< T > &gradB)
{
	gradW.resize(gradA1.size());
	gradB.resize(gradA1.size());

#pragma omp parallel for
	for(int i = 0; i < (int)gradA1.size(); ++i){
		deriv_conv2D(A0, gradA1[i], szA0, szA1, szW, stride, gradW[i], gradB[i]);
	}
}

/**
 * @brief deriv_prev_cnv
 * @param deriv
 * @param W
 * @param sL
 * @param sLsub1
 * @param D
 */
template< typename T >
void deriv_prev_cnv(std::vector< ct::Mat_<T> >& deriv,
					const std::vector< ct::Mat_<T> >& W,
					const ct::Size& sL, const ct::Size& sLsub1,
					ct::Mat_<T>& D)
{
	if(deriv.empty() || W.empty())
		return;

	int m = deriv[0].rows;
	int w_rows = W[0].rows;
	int w_cols = W[0].cols;

	D.setSize(deriv[0].rows, sLsub1.area());
	D.fill(0);

	T *dD = D.ptr();


#pragma omp parallel for
	for(int i = 0; i < m; ++i){
		T *dDi = &dD[i * D.cols];

#pragma omp parallel for
		for(int y = 0; y < sLsub1.height; ++y){
			for(int x = 0; x < sLsub1.width; ++x){
				T sum = 0;

				for(size_t w = 0; w < W.size(); ++w){
					T *dA = deriv[w].ptr();
					T *dAi = &dA[i * deriv[w].cols];
					T *dW = W[w].ptr();

#ifdef __GNUC__
#pragma omp simd
#else
#pragma omp parallel for
#endif
					for(int a = 0; a < w_rows; ++a){
						if(y - a >= 0 && y - a < sL.height){
							for(int b = 0; b < w_cols; ++b){
								if(x - b >=0 && x - b < sL.width){
									int idx = (y - a) * sL.width + (x - b);

									T d = dAi[idx];
									T w = dW[(a) * w_cols + (b)];
									sum += d * w;
								}
							}
						}
					}
				}

				dDi[y * sLsub1.width + x] += sum;// / (sz);
			}
		}
	}
}

/**
 * @brief deriv_prev_cnv
 * @param deriv
 * @param W
 * @param sL
 * @param sLsub1
 * @param D
 */
template< typename T >
void deriv_prev_cnv(const std::vector< ct::Mat_<T> >& deriv,
					const std::vector< ct::Mat_<T> >& W,
					const ct::Size& sL, const ct::Size& sLsub1,
					std::vector< ct::Mat_<T> >& D)
{
	D.resize(deriv.size());
	for(size_t i = 0; i < D.size(); ++i){
		deriv_prev_cnv(deriv[i], W[i], sL, sLsub1, D[i]);
	}
}

}

#endif // NN_H

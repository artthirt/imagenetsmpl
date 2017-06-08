#ifndef MATOPS_H
#define MATOPS_H

#include "custom_types.h"

namespace ct{

template< typename T >
inline Mat_<T> operator* (const Mat_<T>& m1, const Mat_<T>& m2)
{
	if(m1.cols != m2.rows)
		return Mat_<T>();
	int r = m1.rows;
	int c = m2.cols;
	Mat_<T> res(r, c);

	T* valr = &(*res.val)[0];
	T* val1 = &(*m1.val)[0];
	T* val2 = &(*m2.val)[0];

//#pragma omp parallel for
	for(int i = 0; i < m1.rows; i++){

#pragma omp parallel for
		for(int k = 0; k < m2.cols; k++){
			T s = 0;
			for(int j = 0; j < m1.cols; j++){
				s += val1[i * m1.cols + j]/*at(i, j)*/ * val2[j * m2.cols + k]/*at(j, k)*/;
			}
			valr[i * res.cols + k] = s;
//			res.at(i, k) = s;
		}
	}

	return res;
}

template< typename T >
inline Mat_<T> operator* (const Mat_<T>& m1, T v)
{
	if(!m1.total())
		return Mat_<T>();
	Mat_<T> res(m1.rows, m1.cols);

	T* valr = &(*res.val)[0];
	T* val1 = &(*m1.val)[0];
#ifdef __GNUC__
#pragma omp simd
#else
#pragma omp parallel for
#endif
	for(int i = 0; i < m1.rows * m1.cols; i++){
		valr[i] = val1[i] * v;
	}

	return res;
}

template< typename T >
inline Mat_<T> operator* (T v, const Mat_<T>& m1)
{
	if(!m1.total())
		return Mat_<T>();
	Mat_<T> res(m1.rows, m1.cols);

	T* valr = &(*res.val)[0];
	T* val1 = &(*m1.val)[0];
#ifdef __GNUC__
#pragma omp simd
#else
#pragma omp parallel for
#endif
	for(int i = 0; i < m1.rows * m1.cols; i++){
		valr[i] = val1[i] * v;
	}

	return res;
}

template< typename T >
inline Mat_<T> operator+ (const Mat_<T>& m1, const Mat_<T>& m2)
{
	if(m1.cols != m2.cols || m1.rows != m2.rows)
		return Mat_<T>();
	Mat_<T> res(m1.rows, m1.cols);

	T* valr = &(*res.val)[0];
	T* val1 = &(*m1.val)[0];
	T* val2 = &(*m2.val)[0];
#ifdef __GNUC__
#pragma omp simd
#endif
	for(int i = 0; i < m1.rows * m1.cols; i++){
		valr[i] = val1[i] + val2[i];
	}

	return res;
}

template< typename T >
inline Mat_<T> operator+ (const Mat_<T>& m1, const T& v)
{
	if(!m1.total())
		return Mat_<T>();
	Mat_<T> res(m1.rows, m1.cols);

	T* res_val = &(*res.val)[0];
	T* m1_val = &(*m1.val)[0];
#pragma omp parallel for
	for(int i = 0; i < m1.rows * m1.cols; i++){
		res_val[i] = m1_val[i] + v;
	}

	return res;
}

template< typename T >
inline Mat_<T> operator+ (const T& v, const Mat_<T>& m1)
{
	if(!m1.total())
		return Mat_<T>();
	Mat_<T> res(m1.rows, m1.cols);

	T* res_val = &(*res.val)[0];
	T* m1_val = &(*m1.val)[0];
#pragma omp parallel for
	for(int i = 0; i < m1.rows * m1.cols; i++){
		res_val[i] = m1_val[i] + v;
	}

	return res;
}

template< typename T >
inline Mat_<T> operator- (const Mat_<T>& m1, const Mat_<T>& m2)
{
	if(m1.cols != m2.cols || m1.rows != m2.rows)
		return Mat_<T>();
	Mat_<T> res(m1.rows, m1.cols);

	T* res_val = &(*res.val)[0];
	T* m1_val = &(*m1.val)[0];
	T* m2_val = &(*m2.val)[0];
#ifdef __GNUC__
#pragma omp simd
#endif
	for(int i = 0; i < m1.rows * m1.cols; i++){
		res_val[i] = m1_val[i] - m2_val[i];
	}

	return res;
}

template< typename T >
inline Mat_<T> operator- (const Mat_<T>& m1, const T& v)
{
	if(!m1.total())
		return Mat_<T>();
	Mat_<T> res(m1.rows, m1.cols);

	T* res_val = &(*res.val)[0];
	T* m1_val = &(*m1.val)[0];
#ifdef __GNUC__
#pragma omp simd
#endif
	for(int i = 0; i < m1.rows * m1.cols; i++){
		res_val[i] = m1_val[i] - v;
	}

	return res;
}

template< typename T >
inline Mat_<T> operator- (const T& v, const Mat_<T>& m1)
{
	if(!m1.total())
		return Mat_<T>();
	Mat_<T> res(m1.rows, m1.cols);

	T* res_val = &(*res.val)[0];
	T* m1_val = &(*m1.val)[0];
#ifdef __GNUC__
#pragma omp simd
#endif
	for(int i = 0; i < m1.rows * m1.cols; i++){
		res_val[i] = v - m1_val[i];
	}

	return res;
}

template< typename T, int count >
inline Mat_<T> operator* (const Mat_<T>& m1, const Vec_< T, count >& v)
{
	Mat_<T> res(m1.rows, 1);

	if(m1.cols != count)
		return res;

	T* res_val = &(*res.val)[0];
	T* m1_val = &(*m1.val)[0];

#pragma omp parallel for
	for(int i = 0; i < m1.rows; i++){
		T s = 0;
#ifdef __GNUC__
#pragma omp simd
#endif
		for(int j = 0; j < m1.cols; j++){
			s += m1_val[i * m1.cols + j] * v.val[j];
		}
		res_val[i] = s;
	}

	return res;
}

////***************************

template< typename T >
bool operator==(const Mat_<T>& A, const Mat_<T>& B)
{
	if(A.cols != B.cols || A.rows != B.rows)
		return false;

	T* val1 = &(*A.val)[0];
	T* val2 = &(*B.val)[0];
	T eps = 0;
#pragma omp parallel for shared(eps)
	for(int i = 0; i < A.total(); i++){
		eps += std::abs(val1[i] - val2[i]);
	}
	if(eps < 1e-9)
		return true;
	return false;
}

////*************************

/**
 * @brief elemMult
 * @param A = A .* B
 * @param B
 */
template< typename T >
inline void elemwiseMult(Mat_<T > &A, const Mat_<T > &B)
{
	if(A.cols != B.cols || A.rows != B.rows)
		return;

	T* dA = A.ptr();
	T* dB = B.ptr();
#ifdef __GNUC__
#pragma omp simd
#else
#pragma omp parallel for
#endif
	for(int i = 0; i < A.total(); i++){
		dA[i] *= dB[i];
	}
}

/**
 * @brief elemwiseMult
 * @param m1
 * @param m2
 * @param C  = m1 .* m2
 */
template< typename T >
inline void elemwiseMult(const Mat_<T > &m1, const Mat_<T > &m2, Mat_<T>& C)
{
	if(m1.empty() || m2.empty() || m1.cols != m2.cols || m1.rows != m2.rows)
		return;
	if(C.ptr() != m1.ptr() && C.ptr() != m2.ptr()){
		C.setSize(m1.rows, m1.cols);
	}else{
		if(C.ptr() == m1.ptr())
			elemwiseMult(C, m2);
		if(C.ptr() == m2.ptr())
			elemwiseMult(C, m1);
		return;
	}

	T* res_val = C.ptr();
	T* m1_val = m1.ptr();
	T* m2_val = m2.ptr();
#ifdef __GNUC__
#pragma omp simd
#else
#pragma omp parallel for
#endif
	for(int i = 0; i < m1.total(); i++){
		res_val[i] = m1_val[i] * m2_val[i];
	}
}

template< typename T >
inline void flip(const Mat_<T > &A, Mat_<T > &B)
{
	if(A.empty())
		return;

	B.setSize(A.rows, A.cols);

	T *dA = A.ptr();
	T *dB = B.ptr();

	for(int i = 0; i < A.rows; ++i){
		for(int j = 0; j < A.cols; ++j){
			dB[(B.rows - i - 1) * B.cols + j] = dA[i * A.cols + j];
		}
	}
}

template< typename T >
inline Mat_<T> sumRows(const Mat_<T > &m)
{
	Mat_<T> res;
	if(m.rows == 0 || m.cols == 0)
		return res;
	res = Mat_<T>::zeros(1, m.cols);

	T* res_val = &(*res.val)[0];
	T* m_val = &(*m.val)[0];

	for(int i = 0; i < m.rows; i++){
#ifdef __GNUC__
#pragma omp simd
#else
//#pragma omp parallel for
#endif
		for(int j = 0; j < m.cols; j++)
			res_val[j] += m_val[i * m.cols + j];
	}
	return res;
}

/**
 * @brief exp
 * @param m
 * @return exp(m)
 */
template< typename T >
inline Mat_<T> exp(const Mat_<T>& m)
{
	Mat_<T> res(m.rows, m.cols);

	T* res_val = &(*res.val)[0];
	T* m_val = &(*m.val)[0];
//#pragma omp parallel for
#ifdef __GNUC__
#pragma omp simd
#endif
	for(int i = 0; i < m.total(); i++){
		res_val[i] = std::exp(m_val[i]);
	}
	return res;
}

/**
 * @brief log
 * @param m
 * @return log(m)
 */
template< typename T >
inline Mat_<T> log(const Mat_<T>& m)
{
	Mat_<T> res(m.rows, m.cols);

	T* res_val = &(*res.val)[0];
	T* m_val = &(*m.val)[0];
//#pragma omp parallel for
#ifdef __GNUC__
#pragma omp simd
#endif
	for(int i = 0; i < m.total(); i++){
		res_val[i] = std::log(m_val[i]);
	}
	return res;
}

/**
 * @brief expi
 * @param m
 * @return exp(-m)
 */
template< typename T >
inline Mat_<T> expi(const Mat_<T>& m)
{
	Mat_<T> res(m.rows, m.cols);

//#pragma omp parallel for
	T* res_val = &(*res.val)[0];
	T* m_val = &(*m.val)[0];
#ifdef __GNUC__
#pragma omp simd
#endif
	for(int i = 0; i < m.total(); i++){
		res_val[i] = std::exp(-m_val[i]);
	}
	return res;
}

/**
 * @brief sigmoid
 * @param m
 * @return
 */
template< typename T >
inline Mat_<T> sigmoid(const Mat_<T>& m)
{
	Mat_<T> res(m.rows, m.cols);

	T* res_val = &(*res.val)[0];
	T* m_val = &(*m.val)[0];

	//#pragma omp parallel for
#pragma omp parallel for
	for(int i = 0; i < m.total(); i++){
		res_val[i] = (T)(1. / (1. + std::exp(-m_val[i])));
	}
	return res;
}

/**
 * @brief v_sigmoid
 * @param m
 */
template< typename T >
void v_sigmoid(Mat_<T>& m)
{
	T* m_val = &(*m.val)[0];

	//#pragma omp parallel for
#pragma omp parallel for
	for(int i = 0; i < m.total(); i++){
		m_val[i] = (T)(1. / (1. + std::exp(-m_val[i])));
	}
}

/**
 * @brief v_sigmoid
 * @param m
 */
template< typename T >
void v_sigmoid(Mat_<T>& m, Mat_<T>& r)
{
	r.setSize(m.size());
	T* m_val = m.ptr();
	T* r_val = r.ptr();

	//#pragma omp parallel for
#pragma omp parallel for
	for(int i = 0; i < m.total(); i++){
		r_val[i] = 1. / (1. + std::exp(-m_val[i]));
	}
}

/**
 * @brief derivRelu
 * @param m
 * @return
 */
template< typename T >
inline void v_derivSigmoid(const Mat_<T>& m, Mat_<T>& C)
{
	C.setSize(m.size());

	T* res_val = C.ptr();
	T* m_val = m.ptr();

	//#pragma omp parallel for
#ifdef __GNUC__
#pragma omp simd
#else
#pragma omp parallel for
#endif
	for(int i = 0; i < m.total(); i++){
		res_val[i] = m_val[i] * (1 - m_val[i]);
	}
}

/**
 * @brief derivRelu
 * @param m
 * @return
 */
template< typename T >
inline void v_derivSigmoid(Mat_<T>& m)
{
	T* m_val = m.ptr();

	//#pragma omp parallel for
#ifdef __GNUC__
#pragma omp simd
#else
#pragma omp parallel for
#endif
	for(int i = 0; i < m.total(); i++){
		m_val[i] = m_val[i] * (1 - m_val[i]);
	}
}

/**
 * @brief tanh
 * @param m
 * @return
 */
template< typename T >
inline Mat_<T> tanh(const Mat_<T>& m)
{
	Mat_<T> res(m.rows, m.cols);

	T* res_val = &(*res.val)[0];
	T* m_val = &(*m.val)[0];
//#pragma omp parallel for
#pragma omp parallel for
	for(int i = 0; i < m.total(); i++){
		T e = std::exp(2 * m_val[i]);
		res_val[i] = (T)((e - 1.) / (e + 1.));
	}
	return res;
}

/**
 * @brief v_tanh
 * @param m
 */
template< typename T >
void v_tanh(Mat_<T>& m)
{
	T* m_val = &(*m.val)[0];
//#pragma omp parallel for
#pragma omp parallel for
	for(int i = 0; i < m.total(); i++){
		T e = std::exp(2 * m_val[i]);
		m_val[i] = (T)((e - 1.) / (e + 1.));
	}
}

/**
 * @brief v_tanh
 * @param m
 */
template< typename T >
void v_tanh(Mat_<T>& m, Mat_<T>& r)
{
	r.setSize(m.size());
	T* m_val = m.ptr();
	T* r_val = r.ptr();
//#pragma omp parallel for
#pragma omp parallel for
	for(int i = 0; i < m.total(); i++){
		T e = std::exp(2 * m_val[i]);
		r_val[i] = (e - 1.) / (e + 1.);
	}
}

/**
 * @brief v_derivTanh
 * @param m
 * @return
 */
template< typename T >
inline void v_derivTanh(const Mat_<T>& m, Mat_<T>& C)
{
	C.setSize(m.size());

	T* res_val = C.ptr();
	T* m_val = m.ptr();

	//#pragma omp parallel for
#ifdef __GNUC__
#pragma omp simd
#else
#pragma omp parallel for
#endif
	for(int i = 0; i < m.total(); i++){
		res_val[i] = (1 - m_val[i] * m_val[i]);
	}
}

/**
 * @brief v_derivTanh
 * @param m
 * @return
 */
template< typename T >
inline void v_derivTanh(Mat_<T>& m)
{
	T* m_val = m.ptr();

	//#pragma omp parallel for
#ifdef __GNUC__
#pragma omp simd
#else
#pragma omp parallel for
#endif
	for(int i = 0; i < m.total(); i++){
		m_val[i] = (1 - m_val[i] * m_val[i]);
	}
}

/**
 * @brief relu
 * @param m
 * @return
 */
template< typename T >
inline Mat_<T> relu(const Mat_<T>& m)
{
	Mat_<T> res(m.rows, m.cols);

	T* res_val = &(*res.val)[0];
	T* m_val = &(*m.val)[0];
#ifdef __GNUC__
#pragma omp simd
#else
#pragma omp parallel for
#endif
	for(int i = 0; i < m.total(); i++){
		res_val[i] = std::max(T(0), m_val[i]);
	}
	return res;
}

/**
 * @brief v_relu
 * @param m
 * @return
 */
template< typename T >
inline void v_relu(Mat_<T>& m)
{
	T* m_val = &(*m.val)[0];
#ifdef __GNUC__
#pragma omp simd
#else
#pragma omp parallel for
#endif
	for(int i = 0; i < m.total(); i++){
		m_val[i] = std::max(T(0), m_val[i]);
	}
}

/**
 * @brief v_relu
 * @param m
 * @return
 */
template< typename T >
inline void v_relu(const Mat_<T>& m, Mat_<T>& r)
{
	r.setSize(m.size());
	T *m_val = m.ptr();
	T *r_val = r.ptr();
#ifdef __GNUC__
#pragma omp simd
#else
#pragma omp parallel for
#endif
	for(int i = 0; i < m.total(); i++){
		r_val[i] = std::max(T(0), m_val[i]);
	}
}

/**
 * @brief derivRelu
 * @param m
 * @return
 */
template< typename T >
inline Mat_<T> derivRelu(const Mat_<T>& m)
{
	Mat_<T> res(m.rows, m.cols);

	T* res_val = &(*res.val)[0];
	T* m_val = &(*m.val)[0];

	//#pragma omp parallel for
#ifdef __GNUC__
#pragma omp simd
#else
#pragma omp parallel for
#endif
	for(int i = 0; i < m.total(); i++){
		res_val[i] = m_val[i] > T(0) ? T(1) : T(0);
	}
	return res;
}

/**
 * @brief derivSigmoid
 * @param m
 * @return
 */
template< typename T >
inline Mat_<T> derivSigmoid(const Mat_<T>& m)
{
	Mat_<T> res(m.rows, m.cols);

	T* res_val = &(*res.val)[0];
	T* m_val = &(*m.val)[0];

	//#pragma omp parallel for
#ifdef __GNUC__
#pragma omp simd
#else
#pragma omp parallel for
#endif
	for(int i = 0; i < m.total(); i++){
		T val = m_val[i];
		res_val[i] = val * (1 - val);
	}
	return res;
}

/**
 * @brief derivTanh
 * @param m
 * @return
 */
template< typename T >
inline Mat_<T> derivTanh(const Mat_<T>& m)
{
	Mat_<T> res(m.rows, m.cols);

	T* res_val = &(*res.val)[0];
	T* m_val = &(*m.val)[0];

	//#pragma omp parallel for
#ifdef __GNUC__
#pragma omp simd
#else
#pragma omp parallel for
#endif
	for(int i = 0; i < m.total(); i++){
		T val = m_val[i];
		res_val[i] = (1 - val * val);
	}
	return res;
}

/**
 * @brief derivRelu
 * @param m
 * @return
 */
template< typename T >
inline void v_derivRelu(Mat_<T>& m)
{
	T* m_val = &(*m.val)[0];

	//#pragma omp parallel for
#ifdef __GNUC__
#pragma omp simd
#else
#pragma omp parallel for
#endif
	for(int i = 0; i < m.total(); i++){
		m_val[i] = m_val[i] > T(0) ? T(1) : T(0);
	}
}

/**
 * @brief derivRelu
 * @param m
 * @return
 */
template< typename T >
inline void v_derivRelu(const Mat_<T>& m, Mat_<T>& C)
{
	C.setSize(m.size());

	T* res_val = C.ptr();
	T* m_val = m.ptr();

	//#pragma omp parallel for
#ifdef __GNUC__
#pragma omp simd
#else
#pragma omp parallel for
#endif
	for(int i = 0; i < m.total(); i++){
		res_val[i] = m_val[i] > T(0) ? T(1) : T(0);
	}
}

namespace math{

template<typename T >
inline void max_rows(const Mat_<T>& A, Mat_<T>& Max)
{
	Max.setSize(1, A.cols);

	T* dA = &(*A.val)[0];
	T* dM = &(*Max.val)[0];

#pragma omp parallel for
	for(int j = 0; j < A.cols; j++){
		T sC = dA[0 * A.cols + j];
#ifdef __GNUC__
#pragma omp simd
#endif
		for(int i = 1; i < A.rows; i++){
			sC = std::max(dA[i * A.cols + j], sC);
		}
		dM[j] = sC;
	}
}

template<typename T >
inline void max_cols(const Mat_<T>& A, Mat_<T>& Max)
{
	if(A.empty())
		return;
	Max.setSize(A.rows, 1);

	T* dA = &(*A.val)[0];
	T* dM = &(*Max.val)[0];

#pragma omp parallel for
	for(int i = 0; i < A.rows; i++){
		T sC = dA[i * A.cols + 0];
#ifdef __GNUC__
#pragma omp simd
#endif
		for(int j = 1; j < A.cols; j++){
			sC = std::max(dA[i * A.cols + j], sC);
		}
		dM[i] = sC;
	}
}

///

template<typename T >
inline void sum_rows(const Mat_<T>& A, Mat_<T>& Max)
{
	if(A.empty())
		return;

	Max.setSize(1, A.cols);

	T* dA = &(*A.val)[0];
	T* dM = &(*Max.val)[0];

#pragma omp parallel for
	for(int j = 0; j < A.cols; j++){
		T sC = dA[0 * A.cols + j];
#ifdef __GNUC__
#pragma omp simd
#endif
		for(int i= 1; i < A.rows; i++){
			sC += dA[i * A.cols + j];
		}
		dM[j] = sC;
	}
}

template<typename T >
inline void sum_cols(const Mat_<T>& A, Mat_<T>& Max)
{
	if(A.empty())
		return;

	Max.setSize(A.rows, 1);

	T* dA = &(*A.val)[0];
	T* dM = &(*Max.val)[0];

#pragma omp parallel for
	for(int i = 0; i < A.rows; i++){
		T sC = dA[i * A.cols + 0];
#ifdef __GNUC__
#pragma omp simd
#endif
		for(int j = 1; j < A.cols; j++){
			sC += dA[i * A.cols + j];
		}
		dM[i] = sC;
	}
}

///

template< typename T >
inline void exp_rows(const Mat_<T>& A, Mat_<T>& Max, Mat_<T>& C)
{
	if(A.empty() || Max.empty())
		return;

	C.setSize(A.rows, A.cols);

	T* dA = &(*A.val)[0];
	T* dC = &(*C.val)[0];
	T* dM = &(*Max.val)[0];

#pragma omp parallel for
	for(int i = 0; i < A.rows; ++i){
#ifdef __GNUC__
#pragma omp simd
#endif
		for(int j = 0; j < A.cols; ++j){
			T val = dA[i * A.cols + j] - dM[j];
			val = std::exp(val);
			dC[i * A.cols + j] = val;
		}
	}
}

template< typename T >
inline void exp_cols(const Mat_<T>& A, Mat_<T>& Max, Mat_<T>& C)
{
	if(A.empty() || Max.empty())
		return;

	C.setSize(A.rows, A.cols);

	T* dA = &(*A.val)[0];
	T* dC = &(*C.val)[0];
	T* dM = &(*Max.val)[0];

#pragma omp parallel for
	for(int i = 0; i < A.rows; ++i){
#ifdef __GNUC__
#pragma omp simd
#endif
		for(int j = 0; j < A.cols; ++j){
			T val = dA[i * A.cols + j] - dM[i];
			val = std::exp(val);
			dC[i * A.cols + j] = val;
		}
	}
}

////

template< typename T >
inline void sub_ln_rows(Mat_<T>& A, const Mat_<T>& Sum)
{
	if(A.empty() || Sum.empty())
		return;

	T* dA = &(*A.val)[0];
	T* dM = &(*Sum.val)[0];

#pragma omp parallel for
	for(int i = 0; i < A.rows; ++i){
#ifdef __GNUC__
#pragma omp simd
#endif
		for(int j = 0; j < A.cols; ++j){
			T val = std::log(dA[i * A.cols + j]) - std::log(dM[j]);
			dA[i * A.cols + j] = val;
		}
	}
}

template< typename T >
inline void sub_ln_cols(Mat_<T>& A, const Mat_<T>& Sum)
{
	if(A.empty() || Sum.empty())
		return;

	T* dA = &(*A.val)[0];
	T* dM = &(*Sum.val)[0];

#pragma omp parallel for
	for(int i = 0; i < A.rows; ++i){
#ifdef __GNUC__
#pragma omp simd
#endif
		for(int j = 0; j < A.cols; ++j){
			T val = std::log(dA[i * A.cols + j]) - std::log(dM[i]);
			dA[i * A.cols + j] = val;
		}
	}
}

template< typename T >
inline void _exp(Mat_<T>& A)
{
	if(A.empty())
		return;

	T* dA = &(*A.val)[0];

#pragma omp parallel for
	for(int i = 0; i < A.rows; ++i){
#ifdef __GNUC__
#pragma omp simd
#endif
		for(int j = 0; j < A.cols; ++j){
			T val = std::exp(dA[i * A.cols + j]);
			dA[i * A.cols + j] = val;
		}
	}
}

}

/**
 * @brief softmax
 * @param m
 * @return softmax(m)
 */
template< typename T >
inline Mat_<T> softmax(const Mat_<T>& m, int axis = 0)
{
	Mat_<T> res(m.rows, m.cols);

	Mat_<T> Max;
//#pragma omp parallel for

	if(axis == 0){
		math::max_rows<T>(m, Max);
		math::exp_rows<T>(m, Max, res);
		math::sum_rows<T>(res, Max);
		math::sub_ln_rows<T>(res, Max);
		math::_exp(res);
	}else
	if(axis == 1){
		math::max_cols<T>(m, Max);
		math::exp_cols<T>(m, Max, res);
		math::sum_cols<T>(res, Max);
		math::sub_ln_cols<T>(res, Max);
		math::_exp(res);
	}

	return res;
}

/**
 * @brief sqrt
 * @param m
 * @return
 */
template< typename T >
inline Mat_<T> elemwiseSqrt(const Mat_<T>& m)
{
	Mat_<T> res(m.rows, m.cols);

	T* res_val = &(*res.val)[0];
	T* m_val = &(*m.val)[0];

	//#pragma omp parallel for
#ifdef __GNUC__
#pragma omp simd
#endif
	for(int i = 0; i < m.total(); i++){
		res_val[i] = std::sqrt(m_val[i]);
	}
	return res;
}

/**
 * @brief sqrt
 * @param m
 * @return
 */
template< typename T >
void v_elemwiseSqrt(Mat_<T>& m)
{
	if(m.empty())
		throw new std::invalid_argument("v_elemwiseSqrt: matrix is empty");

	T* m_val = m.ptr();

	//#pragma omp parallel for
#ifdef __GNUC__
#pragma omp simd
#endif
	for(int i = 0; i < m.total(); i++){
		m_val[i] = sqrt(m_val[i]);
	}
}

/**
 * @brief sqr
 * @param m
 * @return
 */
template< typename T >
inline Mat_<T> elemwiseSqr(const Mat_<T>& m)
{
	Mat_<T> res(m.rows, m.cols);

	T* res_val = &(*res.val)[0];
	T* m_val = &(*m.val)[0];

	//#pragma omp parallel for
#ifdef __GNUC__
#pragma omp simd
#endif
	for(int i = 0; i < m.total(); i++){
		res_val[i] = m_val[i] * m_val[i];
	}
	return res;
}

/**
 * @brief sqr
 * @param m
 * @return
 */
template< typename T >
inline Mat_<T> elemwiseSqr(const Mat_<T>& m, T eps)
{
	Mat_<T> res(m.rows, m.cols);

	T* res_val = &(*res.val)[0];
	T* m_val = &(*m.val)[0];

	//#pragma omp parallel for
#ifdef __GNUC__
#pragma omp simd
#endif
	for(int i = 0; i < m.total(); i++){
		res_val[i] = m_val[i] * m_val[i] + eps;
	}
	return res;
}

/**
 * @brief sqr
 * @param m
 * @return
 */
template< typename T >
void v_elemwiseSqr(Mat_<T>& m)
{
	if(m.empty())
		throw new std::invalid_argument("v_elemwiseSqr: matrix is empty");

	T* m_val = m.ptr();

	//#pragma omp parallel for
#ifdef __GNUC__
#pragma omp simd
#endif
	for(int i = 0; i < m.total(); i++){
		m_val[i] = m_val[i] * m_val[i];
	}
}

/**
 * @brief division
 * @param m
 * @return
 */
template< typename T >
inline Mat_<T> elemwiseDiv(const Mat_<T>& m1, const Mat_<T>& m2)
{
	if(m1.rows != m2.rows || m1.cols != m2.cols)
		return Mat_<T>();

	Mat_<T> res(m1.rows, m1.cols);

	T* res_val = &(*res.val)[0];
	T* m1_val = &(*m1.val)[0];
	T* m2_val = &(*m2.val)[0];
//#pragma omp parallel for
#ifdef __GNUC__
#pragma omp simd
#endif
	for(int i = 0; i < m1.total(); i++){
		res_val[i] = m1_val[i] / m2_val[i];
	}
	return res;
}

/**
 * @brief matmulT1
 * @param At
 * @param B
 * @param C = A' * B
 */
template< typename T >
void matmulT1(const Mat_<T>& At, const Mat_<T>& B, Mat_<T>& C)
{
	if(At.rows != B.rows)
		return;
	int r = At.cols;
	int c = B.cols;
	if(C.rows != r || C.cols != c)
		C.setSize(r, c);

	T* valr = &(*C.val)[0];
	T* val1 = &(*At.val)[0];
	T* val2 = &(*B.val)[0];

//#pragma omp parallel for
	for(int i = 0; i < At.cols; i++){

#pragma omp parallel for
		for(int k = 0; k < B.cols; k++){
			T s = 0;
#ifdef __GNUC__
#pragma omp simd
#endif
			for(int j = 0; j < At.rows; j++){
				s += val1[j * At.cols + i]/*at(i, j)*/ * val2[j * B.cols + k]/*at(j, k)*/;
			}
			valr[i * C.cols + k] = s;
//			res.at(i, k) = s;
		}
	}

}

/**
 * @brief matmulT1
 * @param A
 * @param Bt
 * @param C = A * B'
 */
template< typename T >
void matmulT2(const Mat_<T>& A, const Mat_<T>& Bt, Mat_<T>& C)
{
	if(A.cols != Bt.cols)
		return;
	int r = A.rows;
	int c = Bt.rows;
	if(C.rows != r || C.cols != c)
		C.setSize(r, c);

	T* valr = &(*C.val)[0];
	T* val1 = &(*A.val)[0];
	T* val2 = &(*Bt.val)[0];

//#pragma omp parallel for
	for(int i = 0; i < A.rows; i++){

#pragma omp parallel for
		for(int k = 0; k < Bt.rows; k++){
			T s = 0;
#ifdef __GNUC__
#pragma omp simd
#endif
			for(int j = 0; j < A.cols; j++){
				s += val1[i * A.cols + j]/*at(i, j)*/ * val2[k * Bt.cols + j]/*at(j, k)*/;
			}
			valr[i * C.cols + k] = s;
		}
	}

}

template< typename T >
void dropout(Mat_<T>& mat, T p, Mat_<T>& D, Mat_<T>& Dt, int seed = 0)
{
	std::binomial_distribution<int> bi(1, p);
	//std::normal_distribution< double > nrm(0, 1);
	if(seed > 0)
		generator.seed(seed);

	D = Mat_<T>::ones(mat.rows, mat.cols);
	Dt = Mat_<T>::ones(mat.cols, mat.rows);

	T* val1 = &(*D.val)[0];
	T* val2 = &(*Dt.val)[0];

//#pragma omp parallel for
	for(int i = 0; i < mat.rows; i++){
//#ifdef __GNUC__
//#pragma omp simd
//#endif
		for(int j = 0; j < mat.cols; j++){
			int pi = bi(generator);
			val1[i * D.cols + j] = (T)pi;
			val2[j * D.rows + i] = (T)pi;
		}
	}
	elemwiseMult(mat, D);
}

template< typename T >
void dropout(int rows, int cols, T p, Mat_<T>& D, int seed = 0)
{
	std::binomial_distribution<int> bi(1, p);
	//std::normal_distribution< double > nrm(0, 1);
	if(seed > 0)
		generator.seed(seed);

	D.setSize(rows, cols);// = Mat_<T>::ones(rows, cols);

	T* val1 = &(*D.val)[0];

//#pragma omp parallel for
	for(int i = 0; i < rows; i++){
//#ifdef __GNUC__
//#pragma omp simd
//#endif
		for(int j = 0; j < cols; j++){
			int pi = bi(generator);
			val1[i * D.cols + j] = T(pi);
		}
	}
}

/**
 * @brief dropout_transpose
 * @param mat
 * @param D
 */
template< typename T >
void dropout_transpose(Mat_<T>& mat, const Mat_<T>& D)
{
	elemwiseMult(mat, D);
}


/**
 * @brief subInd
 * @param mat
 * @param ind
 * @return mat[ind] - 1
 */
template< typename T >
inline Mat_<T> subIndOne(const Mat_<T>& mat, const Mat_<T>& ind)
{
	Mat_<T> res(mat.rows, mat.cols, mat.ptr());

	T* dI = ind.ptr();
	T* dR = res.ptr();

#ifdef __GNUC__
#pragma omp simd
#else
#pragma omp parallel for
#endif
	for(int i = 0; i < mat.rows; ++i){
		int index = (int)dI[i * ind.cols];
		dR[i * mat.cols + index] -= 1.;
	}
	return res;
}

template< typename T >
void get_mean(const ct::Mat_<T>& X, ct::Mat_<T>& Y, int axis = 0)
{
	if(X.empty())
		throw new std::invalid_argument("median: empty input matrix");

	if(axis == 0){

		T m = (T)X.rows;

		Y.setSize(1, X.cols);
		Y.fill(0);

		T *dY = Y.ptr();
		T *dX = X.ptr();
		for(int j = 0; j < X.cols; ++j){
			for(int i = 0; i < X.rows; ++i){
				dY[j] += dX[i * X.cols + j];
			}
		}
		for(int j = 0; j < X.cols; ++j){
			dY[j] /= m;
		}
	}else{
		T m = (T)X.cols;

		Y.setSize(X.rows, 1);
		Y.fill(0);

		T *dY = Y.ptr();
		T *dX = X.ptr();
		for(int i = 0; i < X.rows; ++i){
			for(int j = 0; j < X.cols; ++j){
				dY[i] += dX[i * X.cols + j];
			}
		}
		for(int i = 0; i < X.rows; ++i){
			dY[i] /= m;
		}
	}
}

template< typename T >
void get_std(const ct::Mat_<T>& X, const ct::Mat_<T>& mean, ct::Mat_<T>& Y, int axis = 0)
{
	if(X.empty() || mean.empty() || !(X.cols == mean.cols && mean.rows == 1 && axis == 0)
			&& !(X.rows == mean.rows && mean.cols == 1 && axis == 1))
	{
		throw new std::invalid_argument("get_std: dimensions error");
	}

	Y.setSize(mean.size());
	Y.fill(0);

	T *dX = X.ptr();
	T *dM = mean.ptr();
	T *dY = Y.ptr();

	if(axis == 0){
		T m = (T)X.rows;
		for(int j = 0; j < X.cols; ++j){
			for(int i = 0; i < X.rows; ++i){
				T val1 = dX[i * X.cols + j];
				T val2 = dM[j];
				T val3 = val1 - val2;
				dY[j] += val3 * val3;
			}
		}
		for(int j = 0; j < X.cols; ++j){
			dY[j] = dY[j] / (m - 1);
		}
	}else{
		T m = (T)X.cols;
		for(int i = 0; i < X.rows; ++i){
			for(int j = 0; j < X.cols; ++j){
				T val = (dX[i * X.cols + j] - dM[i]);
				dY[i] += val * val;
			}
		}
		for(int i = 0; i < X.rows; ++i){
			dY[i] = dY[i] / (m - 1);
		}
	}
}

template< typename T >
void get_norm(const ct::Mat_<T>& X, const ct::Mat_<T>& mean, const ct::Mat_<T>& std, ct::Mat_<T>& Y, int axis = 0, T eps = 1e-8)
{
	if(X.empty() || mean.empty() || std.empty())
		throw new ::std::invalid_argument("get_norm: empty matrices");

	Y.setSize(X.size());

	T *dX = X.ptr();
	T *dM = mean.ptr();
	T *dS = std.ptr();
	T *dY = Y.ptr();

	if(axis == 0){
		if(X.cols != mean.cols || mean.rows != 1)
			throw new ::std::invalid_argument("get_norm: dimensions wrong");

		for(int i = 0; i < X.rows; ++i){
			for(int j = 0; j < X.cols; ++j){
				dY[i * Y.cols + j] = (dX[i * X.cols + j] - dM[j])/(sqrt(dS[j] + eps));
			}
		}

	}else{
		if(X.rows != mean.rows || mean.cols != 1)
			throw new ::std::invalid_argument("get_norm: dimensions wrong");

		for(int i = 0; i < X.rows; ++i){
			for(int j = 0; j < X.cols; ++j){
				dY[i * Y.cols + j] = (dX[i * X.cols + j] - dM[i])/(sqrt(dS[i] + eps));
			}
		}

	}
}

/**
 * @brief write_fs
 * @param fs
 * @param mat
 */
template< typename T >
void write_fs(std::fstream& fs, const ct::Mat_<T>& mat)
{
	fs.write((char*)mat.ptr(), sizeof(T) * mat.total());
}

/**
 * @brief read_fs
 * @param fs
 * @param mat
 */
template< typename T >
void read_fs(std::fstream& fs, ct::Mat_<T>& mat)
{
	fs.read((char*)mat.ptr(), sizeof(T) * mat.total());
}

template< typename T >
void read_mat(const std::string& name, ct::Mat_<T>& mat)
{
	std::fstream fs;
	fs.open(name, std::ios_base::in);

	read_fs<T>(fs, mat);

	fs.close();
}

}

#endif // MATOPS_H

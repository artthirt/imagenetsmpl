#ifndef CUSTOM_TYPES_H
#define CUSTOM_TYPES_H

#include <assert.h>
#include <vector>
#include <memory>
#include <algorithm>
#include <math.h>
#include <sstream>
#include <random>
#include <iostream>
#include <fstream>
#include <iomanip>

#include "common_types.h"

#include "shared_memory.h"

#define PRINT_MAT10(mat) {		\
	std::string s = mat.print(10);			\
	qDebug("%s\n", s.c_str());	\
}

namespace ct{

extern std::mt19937 generator;

template< typename T, int count >
class Vec_{
public:
	enum{depth = sizeof(T)};
	T val[count];

	Vec_(){
		std::fill((char*)val, (char*)val + sizeof(val), 0);
	}
	Vec_(const Vec_<T, count >& v){
		std::copy((char*)v.val, (char*)v.val + sizeof(val), (char*)val);
	}
	/**
	 * @brief Vec_
	 * example: Vec4f v = Vec4f(Vec3f())
	 * @param v
	 */
	Vec_(const Vec_<T, count-1 >& v){
		std::copy((char*)v.val, (char*)v.val + sizeof(val), (char*)val);
		std::fill((char*)val + (count - 1) * sizeof(T), (char*)val + sizeof(T), 0);
	}
	/**
	 * @brief Vec_
	 * example: Vec3f v = Vec3f(Vec4f())
	 * @param v
	 */
	Vec_(const Vec_<T, count+1 >& v){
		std::copy((char*)v.val, (char*)v.val + sizeof(val), (char*)val);
	}

	Vec_<T, count>& operator=(const Vec_<T, count >& v){
		std::copy((char*)v.val, (char*)v.val + sizeof(val), (char*)val);
		return *this;
	}
	Vec_<T, count>& operator=(const T& v){
		for(int i = 0; i < count; i++)
			val[i] = v;
		return *this;
	}
	/**
	 * @brief Vec_
	 * example: Vec4f v = (Vec3f())
	 * @param v
	 */
	Vec_<T, count>& operator=(const Vec_<T, count-1 >& v){
		std::copy((char*)v.val, (char*)v.val + sizeof(val), (char*)val);
		std::fill((char*)val + (count - 1) * sizeof(T), (char*)val + sizeof(T), 0);
		return *this;
	}
	/**
	 * @brief Vec_
	 * example: Vec3f v = (Vec4f())
	 * @param v
	 */
	Vec_<T, count>& operator=(const Vec_<T, count+1 >& v){
		std::copy((char*)v.val, (char*)v.val + sizeof(val), (char*)val);
		return *this;
	}

	Vec_(T a0){
		for(int i = 0; i < count; i++)
			val[i] = a0;
	}
	Vec_(T a0, T a1){
		if(count < 2)
			return;
		val[0] = a0;
		val[1] = a1;
		std::fill((char*)val + 2 * sizeof(T), (char*)val + sizeof(val), 0);
	}
	Vec_(T a0, T a1, T a2){
		if(count < 3)
			return;
		val[0] = a0;
		val[1] = a1;
		val[2] = a2;
		std::fill((char*)val + 3 * sizeof(T), (char*)val + sizeof(val), 0);
	}
	Vec_(T a0, T a1, T a2, T a3){
		if(count < 4)
			return;
		val[0] = a0;
		val[1] = a1;
		val[2] = a2;
		val[3] = a3;
		std::fill((char*)val + 4 * sizeof(T), (char*)val + sizeof(val), 0);
	}
	T& operator[] (int index){
		assert(index < count);
		return val[index];
	}
	const T& operator[] (int index) const{
		assert(index < count);
		return val[index];
	}

	inline Vec_<T, count>& operator+=( const Vec_<T, count>& v){
		for(int i = 0; i < count; i++){
			val[i] += v[i];
		}
		return *this;
	}
	inline Vec_<T, count>& operator-=( const Vec_<T, count>& v){
		for(int i = 0; i < count; i++){
			val[i] -= v[i];
		}
		return *this;
	}
	inline Vec_<T, count>& operator*=( const Vec_<T, count>& v){
		for(int i = 0; i < count; i++){
			val[i] *= v[i];
		}
		return *this;
	}
	inline Vec_<T, count>& operator+=(T v){
		for(int i = 0; i < count; i++){
			val[i] += v;
		}
		return *this;
	}
	inline Vec_<T, count>& operator-=(T v){
		for(int i = 0; i < count; i++){
			val[i] -= v;
		}
		return *this;
	}
	inline Vec_<T, count>& operator*=(T v){
		for(int i = 0; i < count; i++){
			val[i] *= v;
		}
		return *this;
	}
	inline Vec_<T, count>& operator/=(T v){
		for(int i = 0; i < count; i++){
			val[i] /= v;
		}
		return *this;
	}
	inline bool operator== (const Vec_<T, count>& v) const{
		const T eps = 1e-12;
		double res = 0;
		for(int i = 0; i < count; i++){
			res += abs(val[i] - v.val[i]);
		}
		return res < eps;
	}
	inline bool empty() const{
		bool notempty = false;
		for(int i = 0; i < count; i++){
			notempty |= val[i] != 0;
		}
		return !notempty;
	}
	inline Vec_<T, count> conj(){
		Vec_<T, count > res;
		for(int i = 0; i < count; i++){
			res.val[i] = -val[i];
		}
		return res;
	}

	inline T sum() const{
		T res = T(0);
		for(int i = 0; i < count; i++)
			res += val[i];
		return res;
	}
	inline T min() const{
		T res = val[0];
		for(int i = 1; i < count; i++)
			res = std::min(res, val[i]);
		return res;
	}
	inline T max() const{
		T res = val[0];
		for(int i = 1; i < count; i++)
			res = std::max(res, val[i]);
		return res;
	}

	inline T norm() const{
		T ret = T(0);
		for(int i = 0; i < count; i++){
			ret += val[i] * val[i];
		}
		return ::sqrt(ret);
	}
	inline T dot(const Vec_<T, count>& v) const{
		T ret = 0;
		for(int i = 0; i < count; i++){
			ret += val[i] * v.val[i];
		}
		return ret;

	}
	inline Vec_< T, count > cross(const Vec_< T, count > & v2){
		Vec_< T, count > res;
		if(count != 3)
			return res;
		res[0] = val[1] * v2.val[2] - val[2] * v2.val[1];
		res[1] = val[2] * v2.val[0] - val[0] * v2.val[2];
		res[2] = val[0] * v2.val[1] - val[1] * v2.val[0];
		return res;
	}

	inline T* ptr(){
		return val;
	}
	inline const T* ptr() const{
		return val;
	}

	inline int size() const{
		return count;
	}

	operator std::string() const{
		std::stringstream ss;
		ss << "[";
		for(int i = 0; i < count; i++){
			ss << val[i] << " ";
		}
		ss << "]";
		return ss.str();
	}

	template< typename C >
	operator Vec_< C, count >() const{
		Vec_< C, count > res;
		for(int i = 0; i < count; i++){
			res.val[i] = val[i];
		}
		return res;
	}

	///******************************
	static inline Vec_< T, count > zeros(){
		Vec_< T, count > res;
		for(int i = 0; i < count; i++){
			res.val[i] = 0;
		}
		return res;
	}
	static inline Vec_< T, count > ones(){
		Vec_< T, count > res;
		for(int i = 0; i < count; i++){
			res.val[i] = 1;
		}
		return res;
	}

private:
};

template<typename T, int count>
inline Vec_<T, count> operator+ (const Vec_<T, count>& v1, const Vec_<T, count>& v2)
{
	Vec_<T, count> ret;

	for(int i = 0; i < count; i++){
		ret.val[i] = v1.val[i] + v2.val[i];
	}
	return ret;
}

template<typename T, int count>
inline Vec_<T, count> operator/ (const Vec_<T, count>& v1, const Vec_<T, count>& v2)
{
	Vec_<T, count> ret;

	for(int i = 0; i < count; i++){
		ret.val[i] = v1.val[i] / v2.val[i];
	}
	return ret;
}

template<typename T, int count>
inline Vec_<T, count> operator- (const Vec_<T, count>& v1, const Vec_<T, count>& v2)
{
	Vec_<T, count> ret;

	for(int i = 0; i < count; i++){
		ret.val[i] = v1.val[i] - v2.val[i];
	}
	return ret;
}

template<typename T, int count>
inline Vec_<T, count> operator* (const Vec_<T, count>& v1, const Vec_<T, count>& v2)
{
	Vec_<T, count> ret;

	for(int i = 0; i < count; i++){
		ret.val[i] = v1.val[i] * v2.val[i];
	}
	return ret;
}

template<typename T, int count>
inline Vec_<T, count> operator+ (const Vec_<T, count>& v1, T v2)
{
	Vec_<T, count> ret;

	for(int i = 0; i < count; i++){
		ret.val[i] = v1.val[i] + v2;
	}
	return ret;
}

template<typename T, int count>
inline Vec_<T, count> operator- (const Vec_<T, count>& v1, T v2)
{
	Vec_<T, count> ret;

	for(int i = 0; i < count; i++){
		ret.val[i] = v1.val[i] - v2;
	}
	return ret;
}

template<typename T, int count>
inline Vec_<T, count> operator* (const Vec_<T, count>& v1, T v2)
{
	Vec_<T, count> ret;

	for(int i = 0; i < count; i++){
		ret.val[i] = v1.val[i] * v2;
	}
	return ret;
}

template<typename T, int count>
inline Vec_<T, count> operator/ (const Vec_<T, count>& v1, T v2)
{
	Vec_<T, count> ret;

	for(int i = 0; i < count; i++){
		ret.val[i] = v1.val[i] / v2;
	}
	return ret;
}

template<typename T, int count>
inline Vec_<T, count> max (const Vec_<T, count>& v1, T v2)
{
	Vec_<T, count > res;
	for(int i = 0; i < count; i++){
		res[i] = std::max(v1.val[i], v2);
	}
	return res;
}

template<typename T, int count>
inline Vec_<T, count> min (const Vec_<T, count>& v1, T v2)
{
	Vec_<T, count > res;
	for(int i = 0; i < count; i++){
		res[i] = std::min(v1.val[i], v2);
	}
	return res;
}

template<typename T, int count>
inline Vec_<T, count> min (const Vec_<T, count>& v1, const Vec_<T, count>& v2)
{
	Vec_<T, count > res;
	for(int i = 0; i < count; i++){
		res[i] = std::min(v1.val[i], v2.val[i]);
	}
	return res;
}

template<typename T, int count>
inline Vec_<T, count> max (const Vec_<T, count>& v1, const Vec_<T, count>& v2)
{
	Vec_<T, count > res;
	for(int i = 0; i < count; i++){
		res[i] = std::max(v1.val[i], v2.val[i]);
	}
	return res;
}

template<typename T, int count>
inline Vec_<T, count> sign(const Vec_<T, count>& v1)
{
	Vec_<T, count > res;
	for(int i = 0; i < count; i++){
		res.val[i] = v1.val[i] >= 0? 1 : -1;
	}
	return res;
}

template<typename T, int count>
inline Vec_<T, count> elemwiseSqrt(const Vec_<T, count>& v1)
{
	Vec_<T, count > res;
	for(int i = 0; i < count; i++){
		res[i] = elemwiseSqrt(v1.val[i]);
	}
	return res;
}

/**
 * @brief crop_angles
 * @param v1
 * @return values in [-M_PI/2, M_PI/2]
 */
template<typename T, int count>
Vec_<T, count> crop_angles(const Vec_<T, count>& v1)
{
	Vec_<T, count > res;
	for(int i = 0; i < count; i++){
		res[i] = atan2(sin(v1.val[i]), cos(v1.val[i]));
	}
	return res;
}

/**
 * @brief crop_angle
 * @param value
 * @return value in [-M_PI/2, M_PI/2]
 */
template< typename T >
inline T crop_angle(const T& value)
{
	return atan2(sin(value), cos(value));
}

template< typename T >
inline T value2range(T value, T min_range, T max_range)
{
	return std::max(min_range, std::min(max_range, value));
}

template< typename T >
inline T values2range(T value, double min_range, double max_range)
{
	T res(value);
	res = min(res, max_range);
	res = max(res, min_range);
	return res;
}


typedef Vec_<int, 2> Vec2i;
typedef Vec_<float, 2> Vec2f;
typedef Vec_<double, 2> Vec2d;

typedef Vec_<float, 3> Vec3f;
typedef Vec_<double, 3> Vec3d;
typedef Vec_<int, 3> Vec3i;
typedef Vec_<unsigned int, 3> Vec3u;

typedef Vec_<float, 4> Vec4f;
typedef Vec_<double, 4> Vec4d;
typedef Vec_<int, 4> Vec4i;
typedef Vec_<unsigned int, 4> Vec4u;

template< typename T, int count >
std::ostream& operator<< (std::ostream& stream, const Vec_<T, count >& v)
{
	std::stringstream ss;
	ss << "[";
	for(int i = 0; i < count; i++){
		ss << v.val[i] << " ";
	}
	ss << "]";
	stream << ss.str();
	return stream;
}

/////////////////////////////////////////////

typedef std::vector< unsigned char > vector_uchar;
typedef std::shared_ptr< vector_uchar > shared_uvector;

template< typename T >
class Mat_{
public:
	typedef std::vector< T > vtype;
	typedef sm::shared_memory< vtype > shared_vtype;

	enum {depth = sizeof(T)};
	shared_vtype val;
	int cols;
	int rows;

	Mat_(){
		cols = rows = 0;
	}

	Mat_(int rows, int cols){
		this->rows = rows;
		this->cols = cols;
		val = sm::make_shared<vtype>();
		val().resize(rows * cols);
	}

	Mat_(const Mat_<T>& m){
		this->val = m.val;
		this->rows = m.rows;
		this->cols = m.cols;
	}
	Mat_(int rows, int cols, void* data){
		this->rows = rows;
		this->cols = cols;
		val = sm::make_shared<vtype>();
		val().resize(rows * cols);
		std::copy((char*)data, (char*)data + rows * cols * depth, (char*)&val()[0]);
	}

	template< int count >
	Mat_(const Vec_< T, count>& v){
		rows = 1;
		cols = count;
		val = sm::make_shared<vtype>();
		val().resize(rows * cols);
		std::copy((char*)v.val, (char*)v.val + sizeof(v.val), (char*)&val()[0]);
	}

	///**********************
	inline Mat_<T>& operator= (const Mat_<T>& m){
		this->rows = m.rows;
		this->cols = m.cols;
		this->val = m.val;
//		val.resize(rows * cols);
//		std::copy(data, data + rows * cols * depth, (char*)&val[0]);
		return *this;
	}

	template< int count >
	inline Mat_<T>& operator= (const Vec_<T, count>& v){
		this->rows = count;
		this->cols = 1;
		T* val = &(*this->val)[0];
		for(int i = 0; i < count; i++)
			val[i] = v.val[i];
		return *this;
	}

	///***********************
	void copyTo(Mat_<T>& mat) const {
		if(empty())
			return;
		mat = Mat_<T>(rows, cols, ptr());
	}

	///***********************
	///
	inline int total() const{
		return rows * cols;
	}

	void setSize(const ct::Size& sz){
		setSize(sz.height, sz.width);
	}

	void setSize(int rows, int cols){
		if(!val.empty()){
			if(rows == this->rows && cols == this->rows)
				return;
			this->rows = rows;
			this->cols = cols;
			val().resize(rows * cols);
		}else{
			this->rows = rows;
			this->cols = cols;
			val = sm::make_shared<vtype>();
			val().resize(rows * cols);
		}
	}

	void clear(){
		if(empty()){
			return;
		}
		rows = 0;
		cols = 0;
		val().clear();
	}

	void fill(T val){
		if(empty())
			return;
		T* d = ptr();
#pragma omp parallel for
		for(int i = 0; i < rows; i++){
			for(int j = 0; j < cols; j++){
				int offset = i * cols + j;
				d[offset] = val;
			}
		}
	}

	void swap_dims(){
		std::swap(rows, cols);
	}

	void set_dims(int _rows, int _cols){
		if(_rows * _cols != total())
			throw new std::invalid_argument("Mat::swap_dims: need equal area");
		rows = _rows;
		cols = _cols;
	}

	void set_dims(ct::Size& sz){
		if(sz.area() != total())
			throw new std::invalid_argument("Mat::swap_dims: need equal area");
		set_dims(sz.height, sz.width);
	}

	///********************
	inline T* ptr(){
		if(empty())
			return 0;
		T* val = &(*this->val)[0];
		return &val[0];
	}

	inline T* ptr() const{
		T* val = &(*this->val)[0];
		return &val[0];
	}

	inline Size size() const{
		return Size(cols, rows);
	}

	///****************
	inline char* bytes(){
		return (char*)val[0];
	}

	inline char* bytes() const{
		return (char*)val[0];
	}
	///*********************
	inline T& at(int i0, int i1){
		T* val = &(*this->val)[0];
		return val[i0 * cols + i1];
	}

	inline T& at(int i0){
		T* val = &(*this->val)[0];
		return val[i0 * cols];
	}

	inline const T& at(int i0, int i1)const{
		T* val = &(*this->val)[0];
		return val[i0 * cols + i1];
	}

	inline const T& at(int i0)const {
		T* val = &(*this->val)[0];
		return val[i0 * cols];
	}

	///********************
	inline Mat_<T>& operator += (const Mat_<T>& v){
		if(rows != v.rows || cols != v.cols)
			return *this;

		T* val1 = &(*this->val)[0];
		T* val2 = &(*v.val)[0];
#pragma omp parallel for
		for(int i = 0; i < rows; i++){
			for(int j = 0; j < cols; j++){
				int offset = i * cols + j;
				val1[offset] += val2[offset];
			}
		}
		return *this;
	}

	inline Mat_<T>& operator -= (const Mat_<T>& v){
		if(rows != v.rows || cols != v.cols)
			return *this;

		T* val1 = &(*this->val)[0];
		T* val2 = &(*v.val)[0];
#pragma omp parallel for
		for(int i = 0; i < rows; i++){
			for(int j = 0; j < cols; j++){
				int offset = i * cols + j;
				val1[offset] -= val2[offset];
			}
		}
		return *this;
	}

	///********************
	inline Mat_<T>& operator *= (T v){
		T* val = &(*this->val)[0];
//#pragma omp parallel for
#pragma omp parallel for
		for(int i = 0; i < rows; i++){
			for(int j = 0; j < cols; j++){
				int offset = i * cols + j;
				val[offset] *= v;
			}
		}
		return *this;
	}

	inline Mat_<T>& operator += (T v){
		T* val = &(*this->val)[0];
//#pragma omp parallel for
#pragma omp parallel for
		for(int i = 0; i < rows; i++){
			for(int j = 0; j < cols; j++){
				int offset = i * cols + j;
				val[offset] += v;
			}
		}
		return *this;
	}

	inline Mat_<T>& operator -= (T v){
		T* val = &(*this->val)[0];
//#pragma omp parallel for
#pragma omp parallel for
		for(int i = 0; i < rows; i++){
			for(int j = 0; j < cols; j++){
				int offset = i * cols + j;
				val[offset] -= v;
			}
		}
		return *this;
	}

	inline Mat_<T>& biasPlus(const Mat_<T > & m){
		if(m.cols != 1 || cols != m.rows)
			return *this;
		T* val1 = &(*this->val)[0];
		T* val2 = &(*m.val)[0];
#pragma omp parallel for
		for(int i = 0; i < rows; i++){
#ifdef __GNUC__
#pragma omp simd
#endif
			for(int j = 0; j < cols; j++){
				val1[i * cols + j] += val2[j];
			}
		}
		return *this;
	}

	inline Mat_<T>& biasMinus(const Mat_<T > & m){
		if(m.cols != 1 || cols != m.rows)
			return *this;
		T* val1 = &(*this->val)[0];
		T* val2 = &(*m.val)[0];
#pragma omp parallel for
		for(int i = 0; i < cols; i++){
#ifdef __GNUC__
#pragma omp simd
#endif
			for(int j = 0; j < rows; j++){
				val1[i * cols + j] -= val2[j];
			}
		}
		return *this;
	}
	///**************************
	void randn(double _mean = 0., double _std = 1., int seed = 0){
		std::normal_distribution< T > nrm((T)_mean, (T)_std);
		if(seed != 0)
			generator.seed(seed);
		T* val = &(*this->val)[0];
		for(int i = 0; i < total(); i++){
			val[i] = nrm(generator);
		}
	}

	///**************************
	///
	Mat_<T> t() const{
		Mat_<T> res(cols, rows);

		T* val1 = &(*res.val)[0];
		T* val2 = &(*this->val)[0];

#pragma omp parallel for
		for(int i = 0; i < rows; i++){
#ifdef __GNUC__
#pragma omp simd
#endif
			for(int j = 0; j < cols; j++){
				val1[j * rows + i] = val2[i * cols + j];
			}
		}

		return res;
	}

	Mat_<T> getRows(std::vector<int> rowsI) const{
		if((int)rowsI.size() > rows)
			return Mat_<T>();
		Mat_<T> res((int)rowsI.size(), cols);

		T* val1 = &(*res.val)[0];
		T* val2 = &(*this->val)[0];
#pragma omp parallel for
		for(int i = 0; i < (int)rowsI.size(); i++){
			int id = rowsI[i];
			if(id < rows){
#ifdef __GNUC__
#pragma omp simd
#endif
				for(int j = 0; j < cols; j++){
					val1[i * cols + j] = val2[id * cols + j];
				}
			}
		}

		return res;
	}

	Mat_<T> getRows(int index, int count) const{
		if(index >= rows)
			return Mat_<T>();
		count = std::min(count, rows - index);
		Mat_<T> res(count, cols);

		T* val1 = &(*res.val)[0];
		T* val2 = &(*this->val)[0];
#pragma omp parallel for
		for(int i = 0; i < count; i++){
			int id = index + i;
#ifdef __GNUC__
#pragma omp simd
#endif
			for(int j = 0; j < cols; j++){
				val1[i * cols + j] = val2[id * cols + j];
			}
		}

		return res;
	}

	Mat_<T> row(int index){
		Mat_<T> res;
		if(empty() || index < 0 || index >= rows)
			return res;
		res.setSize(1, cols);

		T *dM = ptr();
		T *dR = res.ptr();
		for(int i = 0; i < cols; ++i){
			dR[i] = dM[index * cols + i];
		}
		return res;
	}

	Mat_<T> col(int index){
		Mat_<T> res;
		if(empty() || index < 0 || index >= cols)
			return res;
		res.setSize(rows, 1);

		T *dM = ptr();
		T *dR = res.ptr();
		for(int i = 0; i < rows; ++i){
			dR[i] = dM[i * cols + index];
		}
		return res;
	}

	T sum() const{
		T res(0);
		T* val = &(*this->val)[0];
//#pragma omp parallel for shared(res)
		for(int i = 0; i < total(); i++){
			res += val[i];
		}
		return res;
	}
	T max() const{
		T res(0);
		T* val = &(*this->val)[0];
//#pragma omp parallel for shared(res)
		res = val[0];
		for(int i = 1; i < total(); i++){
			res = std::max(res, val[i]);
		}
		return res;
	}

	T min() const{
		T res(0);
		T* val = &(*this->val)[0];
//#pragma omp parallel for shared(res)
		res = val[0];
		for(int i = 1; i < total(); i++){
			res = std::min(res, val[i]);
		}
		return res;
	}

	bool empty() const{
		return val.empty() || val.get()->empty();
	}

	/**
	 * @brief mean
	 * @param axis 0 - mean for all elements; 1 - mean for rows; 2 - mean for cols
	 * @return
	 */
	Mat_<T> mean(int axis = 0){
		Mat_<T> res;
		if(empty())
			return res;
		if(axis == 0){
			res = Mat_<T>::zeros(1, 1);
			T* val1 = &(*res.val)[0];
			T* val2 = &(*this->val)[0];
			for(int i = 0; i < total(); i++){
				val1[0] += val2[i];
			}
			val1[0] = 1. / total();
		}
		if(axis == 1){
			res = Mat_<T>::zeros(1, cols);
			T* val1 = &(*res.val)[0];
			T* val2 = &(*this->val)[0];

#pragma omp parallel for
			for(int i = 0; i < rows; i++){
#ifdef __GNUC__
#pragma omp simd
#endif
				for(int j = 0; j < cols; j++){
					val1[j] += val2[i * cols + j];
				}
			}
			for(int j = 0; j < cols; j++){
				val1[j] *= 1./rows;
			}
		}
		if(axis == 2){
			res = Mat_<T>::zeros(rows, 1);
			T* val1 = &(*res.val)[0];
			T* val2 = &(*this->val)[0];

#pragma omp parallel for
			for(int j = 0; j < cols; j++){
#ifdef __GNUC__
#pragma omp simd
#endif
				for(int i = 0; i < rows; i++){
					val1[i] += val2[i * cols + j];
				}
			}
			for(int i = 0; i < rows; i++){
				val1[i] *= 1./cols;
			}
		}
		return res;
	}

	void sqrt(){
		T* val = &(*this->val)[0];
#pragma omp parallel for
		for(int i = 0; i < rows; i++){
			for(int j = 0; j < cols; j++){
				int offset = i * cols + j;
				val[offset] = std::sqrt(val[offset]);
			}
		}
	}

	///***********************
	/**
	 * @brief argmax
	 * @param index
	 * @param axes - =0 - in row; =1 - in column
	 * @return index of maximum element in row or column
	 */
	int argmax(int index, int axes) const{
		T* val = &(*this->val)[0];
		int res = 0;
		if(axes == 0){
			T mm = val[0 * cols + index];
			for(int i = 1; i < rows; i++){
				if(val[i * cols + index] > mm){
					mm = val[i * cols + index];
					res = i;
				}
			}
		}
		if(axes == 1){
			T mm = val[index * cols + 0];
			for(int i = 1; i < cols; i++){
				if(val[index * cols + i] > mm){
					mm = val[index * cols + i];
					res = i;
				}
			}
		}
		return res;
	}

	///***********************
	///
	template < int count >
	inline Vec_<T, count > toVecCol(int col = 0) const{
		Vec_< T, count > res;

		if(count != rows)
			return res;

		T* val = &(*this->val)[0];
		for(int i = 0; i < rows; i++){
			res.val[i] = val[i * cols + col];
		}
		return res;
	}

	template < int count >
	inline Vec_<T, count > toVecRow(int row = 0) const{
		Vec_< T, count > res;

		if(count != cols)
			return res;

		T* val = &(*this->val)[0];
#pragma omp simd
		for(int i = 0; i < cols; i++){
			res.val[i] = val[row * cols + i];
		}
		return res;
	}

	template < int count >
	inline Vec_<T, count > toVec() const{
		Vec_< T, count > res;

		if((cols == 1 || count == rows) && (rows == 1 || count == cols))
			return res;

		T* val = &(*this->val)[0];
#pragma omp simd
		for(int i = 0; i < count; i++){
			res.val[i] = val[i];
		}
		return res;
	}

	///**************************
	///
	operator std::string() const{
		if(this->val.empty())
			return "";
		std::stringstream res;
		T* val = &(*this->val)[0];
		res << "[";
		for(int i = 0; i < rows; i++){
			for(int j = 0; j < cols; j++){
				res << std::setprecision(4) << val[i * cols + j] << "\t";
			}
			res << ";\n";
		}
		res << "]";
		return res.str();
	}

	std::string print(int _rows = -1) const{
		if(this->val.empty())
			return "";

		if(_rows < 0)
			_rows = rows;
		if(_rows > rows)
			_rows = rows;

		std::stringstream res;
		T* val = &(*this->val)[0];
		res << "[";
		for(int i = 0; i < _rows; i++){
			for(int j = 0; j < cols; j++){
				res << std::setprecision(4) << val[i * cols + j] << "\t";
			}
			res << ";\n";
		}
		res << "]";
		return res.str();
	}

	///**************************
	static inline Mat_< T > zeros(int rows, int cols){
		Mat_< T > res(rows, cols);
		T* val = &(*res.val)[0];
#pragma omp parallel for
		for(int i = 0; i < rows; i++){
			for(int j = 0; j < cols; j++){
				int offset = i * cols + j;
				val[offset] = 0;
			}
		}
		return res;
	}

	static inline Mat_< T > ones(int rows, int cols){
		Mat_< T > res(rows, cols);
		T* val = &(*res.val)[0];
#pragma omp parallel for
		for(int i = 0; i < rows; i++){
			for(int j = 0; j < cols; j++){
				int offset = i * cols + j;
				val[offset] = 1.;
			}
		}
		return res;
	}

	static inline Mat_< T > zeros(const Size& size){
		return Mat_<T>::zeros(size.height, size.width);
	}

	static inline Mat_< T > ones(const Size& size){
		return Mat_<T>::ones(size.height, size.width);
	}

	static inline Mat_< T > eye(int rows, int cols){
		Mat_< T > res = zeros(rows, cols);
		T* val = &(*res.val)[0];
#ifdef __GNUC__
#pragma omp simd
#else
#pragma omp parallel for
#endif
		for(int i = 0; i < std::min(rows, cols); ++i){
			val[i * cols + i] = 1;
		}
		return res;
	}

private:
};

//////////////////////////////////////////

typedef Mat_<int> Mati;
typedef Mat_<float> Matf;
typedef Mat_<double> Matd;

//////////////////////////////////////////

/**
 * @brief get_tangage_mat
 * @param angle
 * @return
 */
template< typename T >
inline Mat_< T > get_tangage_mat(T angle)
{
	T data[9] = {
		1,	0,				0,
		0,	sin(angle),		cos(angle),
		0,	cos(angle),		-sin(angle),
	};
	return Mat_<T>(3, 3, data);
}

/**
 * @brief get_roll_mat
 * @param angle
 * @return
 */
template< typename T >
inline Mat_< T > get_roll_mat(T angle)
{
	T data[9] = {
		sin(angle),		0,		cos(angle),
		0,				1,				0,
		cos(angle),		0,		-sin(angle),
	};
	return Mat_<T>(3, 3, data);
}

/**
 * @brief get_yaw_mat
 * @param angle
 * @return
 */
template< typename T >
inline Mat_< T > get_yaw_mat(T angle)
{
	T data[9] = {
		sin(angle),		cos(angle),		0,
		cos(angle),		-sin(angle),	0,
		0,				0,				1
	};
	return Mat_<T>(3, 3, data);
}

template< typename T >
inline Mat_< T > get_eiler_mat(const Vec_< T, 3 >& angles)
{
	T yaw, tangage, bank;
	Mat_< T > m(3, 3);

	yaw		= angles[0];
	tangage	= angles[1];
	bank	= angles[2];

	m.at(0, 0) = cos(yaw) * cos(bank) - sin(yaw) * sin(tangage) * sin(bank);
	m.at(0, 1) = -cos(yaw) * sin(bank) - sin(yaw) * sin(tangage) * cos(bank);
	m.at(0, 2) = -sin(yaw) * cos(tangage);

	m.at(1, 0) = cos(tangage) * sin(bank);
	m.at(1, 1) = cos(tangage) * cos(bank);
	m.at(1, 2) = -sin(tangage);

	m.at(2, 0) = sin(yaw) * cos(bank) + cos(yaw) * sin(tangage) * sin(bank);
	m.at(2, 1) = -sin(yaw) * sin(bank) + cos(yaw) * sin(tangage) * cos(bank);
	m.at(2, 2) = cos(yaw) * cos(tangage);

	return m;
}

template< typename T >
inline Mat_< T > get_eiler_mat4(const Vec_< T, 3 >& angles)
{
	T yaw, tangage, bank;
	Mat_< T > m = Mat_< T >::eye(4, 4);

	yaw		= angles[0];
	tangage	= angles[1];
	bank	= angles[2];

	m.at(0, 0) = cos(yaw) * cos(bank) - sin(yaw) * sin(tangage) * sin(bank);
	m.at(0, 1) = -cos(yaw) * sin(bank) - sin(yaw) * sin(tangage) * cos(bank);
	m.at(0, 2) = -sin(yaw) * cos(tangage);

	m.at(1, 0) = cos(tangage) * sin(bank);
	m.at(1, 1) = cos(tangage) * cos(bank);
	m.at(1, 2) = -sin(tangage);

	m.at(2, 0) = sin(yaw) * cos(bank) + cos(yaw) * sin(tangage) * sin(bank);
	m.at(2, 1) = -sin(yaw) * sin(bank) + cos(yaw) * sin(tangage) * cos(bank);
	m.at(2, 2) = cos(yaw) * cos(tangage);

	return m;
}

/**
 * @brief get_yaw_mat2
 * @param yaw
 * @return
 */
template< typename T >
inline Mat_<T> get_yaw_mat2(T yaw)
{
	T data[9] = {
		cos(yaw),	0,		-sin(yaw),
		0.,			1,		0,
		sin(yaw),	0,		cos(yaw),
	};
	Mat_<T> res(3, 3, data);
	return res;
}

///////////////////////////

#define M_PI       3.14159265358979323846

template< typename T >
inline T angle2rad( T val)
{
	return static_cast< T > (val * M_PI / 180.);
}

template< typename T >
inline T rad2angle(T val)
{
	return static_cast< T > (val * 180. / M_PI);
}

////////////////////////////

template< typename T >
void save_mat(const Mat_<T>& mat, const std::string& fn)
{
	std::string s = mat;			\
	std::fstream fs;
	fs.open(fn.c_str(), std::ios_base::out);

	fs << s;

	fs.close();
}

template< typename T >
void save_mat10(const Mat_<T>& mat, const std::string& fn)
{
	std::string s = mat.print(10);			\
	std::fstream fs;
	fs.open(fn.c_str(), std::ios_base::out);

	fs << s;

	fs.close();
}

}

#endif // CUSTOM_TYPES_H

#include "gpumat.h"

#include <iomanip>
#include <sstream>
#include <fstream>

#include <cuda_runtime.h>
#include <assert.h>
#include <exception>

#include "cuda_types.h"

using namespace gpumat;

///////////////////////////////////

/**
 * @brief cuda_memset
 * @param A
 * @param B
 * @param C - out C = A .+ B
 */
extern "C"
void cuda_memset(GpuMat& A, double val);

/**
 * @brief add
 * @param A
 * @param B
 * @param C - out C = A .+ B
 */
extern "C"
void cuda_add(const GpuMat& A, const GpuMat& B, GpuMat& C);

/**
 * @brief cuda_add_params
 * @param A
 * @param val1
 * @param B
 * @param val2
 * @param C = val1 * A + val2 * B
 */
extern "C"
void cuda_add_params(const GpuMat& A, const GpuMat& B, double val1, double val2, GpuMat& C);

/**
 * @brief cuda_add_paramsA
 * @param A -> A += val1 * B
 * @param val
 * @param B
 */
extern "C"
void cuda_add_paramsA(GpuMat& A, const GpuMat& B, double val1, double val2);

/**
 * @brief sub
 * @param A
 * @param B
 * @param C - out C = A .- B
 */
extern "C"
void cuda_sub(const GpuMat& A, const GpuMat& B, GpuMat& C, double valA, double valB);

/**
 * @brief cuda_subA
 * @param A = A * valA - B * valB
 * @param B
 * @param valA
 * @param valB
 */
extern "C"
void cuda_subA(GpuMat& A, const GpuMat& B, double valA, double valB);

/**
 * @brief matmul
 * @param A
 * @param B
 * @param C - out C = A * B
 */
extern "C"
void cuda_matmul(const GpuMat& A, const GpuMat& B, GpuMat& C);

/**
 * @brief matmul_shared
 * @param A
 * @param B
 * @param C - out C = A * B
 */
extern "C"
void cuda_matmul_shared(const GpuMat& A, const GpuMat& B, GpuMat& C);

/**
 * @brief matmulT1
 * @param At - used as transposed matrix
 * @param B
 * @param C - out C = A' * B
 */
extern "C"
void cuda_matmulT1(const GpuMat& At, const GpuMat& B, GpuMat& C);

/**
 * @brief matmulT1_shared
 * @param At - used as transposed matrix
 * @param B
 * @param C - out C = A' * B
 */
extern "C"
void cuda_matmulT1_shared(const GpuMat& At, const GpuMat& B, GpuMat& C);


/**
 * @brief matmulT2
 * @param A
 * @param Bt - used as transposed matrix
 * @param C - out C = A * B'
 */
extern "C"
void cuda_matmulT2(const GpuMat& A, const GpuMat& Bt, GpuMat& C);

/**
 * @brief matmulT2_shared
 * @param A
 * @param Bt - used as transposed matrix
 * @param C - out C = A * B'
 */
extern "C"
void cuda_matmulT2_shared(const GpuMat& A, const GpuMat& Bt, GpuMat& C);

/**
 * @brief mulval
 * @param A
 * @param value - mat 1x1
 * @param C - out C = A * value
 */
extern "C"
void cuda_mulval(const GpuMat& A, double value, GpuMat& C);

/**
 * @brief mulval
 * @param A -> A *= value
 * @param value - mat 1x1
 */
extern "C"
void cuda_mulvalA(const GpuMat& A, double value);

/**
 * @brief addval
 * @param A
 * @param value - mat 1x1
 * @param C - out C = A + value
 */
extern "C"
void cuda_addval(const GpuMat& A, double value, GpuMat& C);

/**
 * @brief addval
 * @param A
 * @param value - mat 1x1
 * @param C - out C = A + value
 */
extern "C"
void cuda_addvalA(GpuMat& A, double value);

/**
 * @brief subval
 * @param A
 * @param value - mat 1x1
 * @param C - out C = A - value
 */
extern "C"
void cuda_subval_AvaltoC(const GpuMat& A, double value, GpuMat& C);

/**
 * @brief subval
 * @param A
 * @param value - mat 1x1
 * @param C - out C = value - C
 */
extern "C"
void cuda_subval_valA(GpuMat& A, double value);

/**
 * @brief subval
 * @param A
 * @param value - mat 1x1
 * @param C - out C = A - value
 */
extern "C"
void cuda_subval_Aval(GpuMat& A, double value);

/**
 * @brief subval
 * @param A
 * @param value - mat 1x1
 * @param C - out C = value - C
 */
extern "C"
void cuda_subval_valAtoC(double value, const GpuMat& A, GpuMat& C);
/**
 * @brief biasPlus
 * @param A - out A[i] = A[i] + bias
 * @param bias
 */
extern "C"
void cuda_biasPlus(GpuMat& A, const GpuMat& bias);

/**
 * @brief elemwiseMul
 * @param A
 * @param B
 * @param C - out C = A .* B
 */
extern "C"
void cuda_elemwiseMul(const GpuMat& A, const GpuMat& B, GpuMat& C);

/**
 * @brief elemwiseMul
 * @param A = A .* B
 * @param B
 */
extern "C"
void cuda_elemwiseMulA(GpuMat& A, const GpuMat& B);

/**
 * @brief elemwiseDiv
 * @param A
 * @param B
 * @param C - out C = A ./ B
 */
extern "C"
void cuda_elemwiseDiv(const GpuMat& A, const GpuMat& B, GpuMat& C);

/**
 * @brief elemwiseSqrt
 * @param A
 * @param C - out C = sqrt(A)
 */
extern "C"
void cuda_elemwiseSqrt(const GpuMat& A, GpuMat& C);

/**
 * @brief elemwiseSqr
 * @param A
 * @param C - out C = A .* A
 */
extern "C"
void cuda_elemwiseSqr(const GpuMat& A, GpuMat& C);

/**
 * @brief cuda_sumrows
 * @param A
 * @param C - out C[i] = val * sum(A[i, j])(j = [1..cols])
 */
extern "C"
void cuda_sumrows(const GpuMat& A, GpuMat& C, double val);

/**
 * @brief cuda_sumrows_shared
 * @param A
 * @param C - out C[i] = val * sum(A[i, j])(j = [1..cols])
 */
extern "C"
void cuda_sumrows_shared(const GpuMat& A, GpuMat& C, double val);

/**
 * @brief cuda_transpose
 * @param A
 * @param C = A'
 */
extern "C"
void cuda_transpose(const GpuMat& A, GpuMat& C);

/**
 * @brief cuda_reLu
 * @param A
 * @param C = reLu(A)
 */
extern "C"
void cuda_reLu(const GpuMat& A, GpuMat& C);

/**
 * @brief cuda_reLu
 * @param A
 * @param C = reLu(A)
 */
extern "C"
void cuda_reLu2(GpuMat& A);

/**
 * @brief cuda_derivReLu
 * @param A
 * @param C = derivRelu(A)
 */
extern "C"
void cuda_derivReLu(const GpuMat& A, GpuMat& C);
extern "C"
void cuda_derivReLu2(GpuMat& A);

/**
 * @brief cuda_sigmoid
 * @param A
 * @param C = sigmoid(A)
 */
extern "C"
void cuda_sigmoid(const GpuMat& A, GpuMat& C);

/**
 * @brief cuda_sigmoid
 * @param A = sigmoid(A)
 */
extern "C"
void cuda_sigmoid2(GpuMat& A);

/**
 * @brief cuda_deriv_sigmoid
 * @param A
 * @param C = deriv_sigmoid(A)
 */
extern "C"
void cuda_deriv_sigmoid(const GpuMat& A, GpuMat& C);
extern "C"
void cuda_deriv_sigmoid2(GpuMat& A);
/**
 * @brief cuda_tanh
 * @param A
 * @param C = tanh(A)
 */
extern "C"
void cuda_tanh(const GpuMat& A, GpuMat& C);

/**
 * @brief cuda_tanh
 * @param A = tanh(A)
 */
extern "C"
void cuda_tanh2(GpuMat& A);

/**
 * @brief cuda_deriv_tanh
 * @param A
 * @param C = deriv_tanh(A)
 */
extern "C"
void cuda_deriv_tanh(const GpuMat& A, GpuMat& C);
extern "C"
void cuda_deriv_tanh2(GpuMat& A);

/**
 * @brief cuda_softmax
 * @param A
 * @param axis -> 0 - in row, 1 - in col
 * @param C = softmax(A)
 */
extern "C"
void cuda_softmax(const GpuMat& A, int axis, GpuMat& C, GpuMat& partZ);

/**
 * @brief cuda_softmax
 * @param A = softmax(A)
 * @param axis -> 0 - in row, 1 - in col
 */
extern "C"
void cuda_softmax2(GpuMat& A, int axis, GpuMat& partZ);

/**
 * @brief cuda_adamgrad
 * @param A = -alpha * (sb1 * mA / (sqrt(sb2 * vA) + eps)
 * @param mA
 * @param vA
 * @param alpha
 * @param sb1
 * @param sb2
 */
extern "C"
void cuda_adamgrad(GpuMat& A, const GpuMat& mA, const GpuMat& vA,
				   double alpha, double sb1, double sb2);

/**
 * @brief cuda_subIndOne
 * @param A
 * @param Ind
 * @param B = A : A[row, col == Ind[row]] - 1
 */
extern "C"
void cuda_subIndOne(const GpuMat& A, const GpuMat& Ind, GpuMat& B);


/////////////////////////////////////////////////

///////////////////////////////////
///////////////////////////////////

GpuMat::GpuMat()
{
	rows = 0;
	cols = 0;
	type = 0;
	data = nullptr;
}

GpuMat::GpuMat(int rows, int cols, int type)
{
	this->rows = 0;
	this->cols = 0;
	this->type = 0;
	this->data = nullptr;

	if(rows && cols){
		this->rows = rows;
		this->cols = cols;
		this->type = type;

		int size = rows * cols * depth();

		cudaError_t err = cudaMalloc(&data, size);
		assert(err == cudaSuccess);
	}
}

GpuMat::GpuMat(int rows, int cols, int type, void *data)
{
	this->rows = 0;
	this->cols = 0;
	this->type = 0;
	this->data = nullptr;

	if(rows && cols && data){
		this->rows = rows;
		this->cols = cols;
		this->type = type;

		int size = rows * cols * depth();

		cudaError_t err = cudaMalloc(&this->data, size);

		if(data && this->data){
			err = cudaMemcpy(this->data, data, size, cudaMemcpyHostToDevice);
		}
		assert(err == cudaSuccess);
	}
}

GpuMat::GpuMat(const GpuMat &mat)
{
	rows = mat.rows;
	cols = mat.cols;
	type = mat.type;
	data = nullptr;

	if(mat.data){
		cudaError_t err = cudaMalloc((void**)&data, mat.size());
		if(data && mat.data)
			err = cudaMemcpy(data, mat.data, mat.size(), cudaMemcpyDeviceToDevice);
		assert(err == cudaSuccess);
	}
}

GpuMat::~GpuMat()
{
	release();
}

GpuMat &GpuMat::operator =(const GpuMat &mat)
{
	if(mat.empty())
		return *this;

	cudaError_t err = cudaSuccess;
	if(mat.rows != rows || mat.cols != cols || mat.type != type){
		release();

		rows = mat.rows;
		cols = mat.cols;
		type = mat.type;

		err = cudaMalloc(&data, mat.size());
		assert(err == cudaSuccess);
	}

	if(mat.data && err == cudaSuccess ){
		err = cudaMemcpy(data, mat.data, mat.size(), cudaMemcpyDeviceToDevice);
		assert(err == cudaSuccess);
	}
	return *this;
}

GpuMat &GpuMat::ones()
{
	if(empty())
		return *this;

	memset(*this, 1);

	return *this;
}

GpuMat &GpuMat::zeros()
{
	if(empty())
		return *this;

	memset(*this, 0);

	return *this;
}

int GpuMat::depth() const
{
	return SIZEOF_TYPE(type);
}

int GpuMat::size() const
{
	return rows * cols * depth();
}

int GpuMat::total() const
{
	return rows * cols;
}

bool GpuMat::empty() const
{
	return data == nullptr;
}

ct::Size GpuMat::sz() const
{
	return ct::Size(cols, rows);
}

void GpuMat::resize(int rows, int cols, int type)
{
	if(!rows || ! cols)
		return;

	int sz = rows * cols * SIZEOF_TYPE(type);

	if(sz == size()){
		this->rows = rows;
		this->cols = cols;
		this->type = type;

		return;
	}
	release();

	this->rows = rows;
	this->cols = cols;
	this->type = type;

	cudaError_t err = cudaMalloc(&data, size());
	assert(err == cudaSuccess);
}

void GpuMat::resize(const ct::Size &sz, int type)
{
	resize(sz.height, sz.width, type);
}

void GpuMat::resize(const GpuMat &mat)
{
	if(mat.empty())
		return;

	resize(mat.rows, mat.cols, mat.type);
}

void GpuMat::copyTo(GpuMat &mat) const
{
	if(empty())
		return;

	mat = *this;
}

void GpuMat::setData(void *data)
{
	if(!data || !this->data || !rows || !cols)
		return;

	cudaError_t err = cudaMemcpy(this->data, data, size(), cudaMemcpyHostToDevice);
	assert(err == cudaSuccess);
}

void GpuMat::getData(void *data) const
{
	if(!this->data || !data || !rows || !cols)
		return;

	cudaError_t err = cudaMemcpy(data, this->data, size(), cudaMemcpyDeviceToHost);
	assert(err == cudaSuccess);
}

void GpuMat::free()
{
	release();
}

void GpuMat::swap_dims()
{
	std::swap(rows, cols);
}

//************

template<typename T >
std::string getString(void* data, int rows, int cols)
{
	if(!rows || !cols)
		return "";
	std::vector<T> vec;
	vec.resize(rows * cols);

	int size = rows * cols * sizeof(T);

	cudaError_t err = cudaMemcpy(&vec[0], data, size, cudaMemcpyDeviceToHost);
	assert(err == cudaSuccess);

	std::stringstream stream;

	stream << std::setprecision(4) << "[";
	for(int i = 0; i < rows; i++){
		for(int j = 0; j < cols; j++){
			stream << vec[i * cols + j] << "\t";
		}
		if(i != rows - 1)stream << ";\n ";
	}
	stream << "]\n";
	return stream.str();
}

//************

std::string GpuMat::operator()() const
{
	if(!data)
		return "";

	switch (type) {
		case GPU_FLOAT:
			return getString<float>(data, rows, cols);
		case GPU_DOUBLE:
			return getString<double>(data, rows, cols);
	}
	return "";
}

std::string GpuMat::print(int _rows) const
{
	if(!data)
		return "";

	if(_rows < 0)
		_rows = rows;
	if(_rows > rows)
		_rows = rows;

	std::string res;
	switch (type) {
		case GPU_FLOAT:
			res = getString<float>(data, _rows, cols);
		break;
		case GPU_DOUBLE:
			res = getString<double>(data, _rows, cols);
		break;
	}

//	std::fstream fs;
//	fs.open("temp.txt", std::ios_base::out);
//	fs.write(res.c_str(), res.size());
//	fs.close();

	return res;
}

void GpuMat::save(const std::string filename) const
{
	std::string res = (*this)();

	std::fstream fs;
	fs.open(filename.c_str(), std::ios_base::out);
	fs.write(res.c_str(), res.size());
	fs.close();
}

void GpuMat::release()
{
	if(data != nullptr){
		cudaError_t err = cudaFree(data);
		assert(err == cudaSuccess);
		data = nullptr;
	}
	rows = cols = type = 0;
}

/////////////////////////////////////////////////

namespace gpumat {

/**
 * @brief memset
 * @param A
 * @param val
 */
void memset(GpuMat& A, double val)
{
	if(A.empty()){
		throw new std::invalid_argument("memset");
	}

	cuda_memset(A, val);
}

void add(const GpuMat &A, const GpuMat &B, GpuMat &C)
{
	if(A.rows != B.rows || A.cols != B.cols || A.type != B.type){
		throw new std::invalid_argument("add");
	}

	if(C.rows != A.rows || C.cols != A.cols || C.type != A.type)
		C.resize(A);

	cuda_add(A, B, C);
}


void add(const GpuMat &A, const GpuMat &B, GpuMat &C, double valA, double valB)
{
	if(A.rows != B.rows || A.cols != B.cols || A.type != B.type){
		throw new std::invalid_argument("add");
	}

	if(C.rows != A.rows || C.cols != A.cols || C.type != A.type)
		C.resize(A);

	cuda_add_params(A, B, valA, valB, C);
}

void add(GpuMat &A, const GpuMat &B, double valA, double valB)
{
	if(A.rows != B.rows || A.cols != B.cols || A.type != B.type){
		throw new std::invalid_argument("add");
	}

	cuda_add_paramsA(A, B, valA, valB);
}

void sub(const GpuMat &A, const GpuMat &B, GpuMat &C, double valA, double valB)
{
	if(A.rows != B.rows || A.cols != B.cols || A.type != B.type){
		throw new std::invalid_argument("sub");
	}

	if(C.rows != A.rows || C.cols != A.cols || C.type != A.type)
		C.resize(A);

	cuda_sub(A, B, C, valA, valB);
}


void sub(GpuMat &A, const GpuMat &B, double valA, double valB)
{
	if(A.rows != B.rows || A.cols != B.cols || A.type != B.type){
		throw new std::invalid_argument("sub");
	}

	cuda_subA(A, B, valA, valB);
}

void matmul(const GpuMat &A, const GpuMat &B, GpuMat &C)
{
	if(A.cols != B.rows || A.type != B.type){
		throw new std::invalid_argument("matmul");
	}

	if(C.rows != A.rows || C.cols != B.cols || C.type != A.type)
		C.resize(A.rows, B.cols, A.type);

	cuda_matmul(A, B, C);
}

void matmul_shared(const GpuMat &A, const GpuMat &B, GpuMat &C)
{
	if(A.cols != B.rows || A.type != B.type){
		throw new std::invalid_argument("matmul_shared");
	}

	if(C.rows != A.rows || C.cols != B.cols || C.type != A.type)
		C.resize(A.rows, B.cols, A.type);

	cuda_matmul_shared(A, B, C);
}

void matmulT1(const GpuMat &At, const GpuMat &B, GpuMat &C)
{
	if(At.rows != B.rows || At.type != B.type){
		throw new std::invalid_argument("matmulT1");
	}

	if(C.rows != At.cols || C.cols != B.cols || C.type != At.type)
		C.resize(At.cols, B.cols, At.type);

	cuda_matmulT1(At, B, C);
}

void matmulT1_shared(const GpuMat &At, const GpuMat &B, GpuMat &C)
{
	if(At.rows != B.rows || At.type != B.type){
		throw new std::invalid_argument("matmulT1_shared");
	}

	if(C.rows != At.cols || C.cols != B.cols || C.type != At.type)
		C.resize(At.cols, B.cols, At.type);

	cuda_matmulT1_shared(At, B, C);
}


void matmulT2(const GpuMat &A, const GpuMat &Bt, GpuMat &C)
{
	if(A.cols != Bt.cols || A.type != Bt.type){
		throw new std::invalid_argument("matmulT2");
	}

	if(C.rows != A.rows || C.cols != Bt.rows || C.type != A.type)
		C.resize(A.rows, Bt.rows, A.type);

	cuda_matmulT2(A, Bt, C);
}

void matmulT2_shared(const GpuMat &A, const GpuMat &Bt, GpuMat &C)
{
	if(A.cols != Bt.cols || A.type != Bt.type){
		throw new std::invalid_argument("matmulT2_shared");
	}

	if(C.rows != A.rows || C.cols != Bt.rows || C.type != A.type)
		C.resize(A.rows, Bt.rows, A.type);

	cuda_matmulT2_shared(A, Bt, C);
}


void mulval(const GpuMat &A, double value, GpuMat &C)
{
	if(C.rows != A.rows || C.cols != A.cols || C.type != A.type)
		C.resize(A);

	cuda_mulval(A, value, C);
}

void mulval(GpuMat& A, double value)
{
	cuda_mulvalA(A, value);
}

void addval(const GpuMat &A, double value, GpuMat &C)
{
	if(C.rows != A.rows || C.cols != A.cols || C.type != A.type)
		C.resize(A);

	cuda_addval(A, value, C);
}

void addval(GpuMat& A, double value)
{
	cuda_addvalA(A, value);
}

void subval(const GpuMat &A, double value, GpuMat &C)
{
	if(C.rows != A.rows || C.cols != A.cols || C.type != A.type)
		C.resize(A);

	cuda_subval_AvaltoC(A, value, C);
}

void subval(double value, const GpuMat &A, GpuMat &C)
{
	if(C.rows != A.rows || C.cols != A.cols || C.type != A.type)
		C.resize(A);

	cuda_subval_valAtoC(value, A, C);
}

void subval(GpuMat &A, double value)
{
	cuda_subval_Aval(A, value);
}

void subval(double value, GpuMat &A)
{
	cuda_subval_valA(A, value);
}

void biasPlus(GpuMat &A, const GpuMat &bias)
{
	if((A.cols != bias.cols || bias.rows != 1) && (A.cols != bias.rows || bias.cols != 1)){
		throw new std::invalid_argument("biasPlus");
	}

	cuda_biasPlus(A, bias);
}

void elemwiseMult(const GpuMat &A, const GpuMat &B, GpuMat &C)
{
	if(A.rows != B.rows || A.cols != B.cols || A.type != B.type){
		throw new std::invalid_argument("elemwiseMult");
	}

	if(C.rows != A.rows || C.cols != A.cols || C.type != A.type)
		C.resize(A);

	cuda_elemwiseMul(A, B, C);
}


void elemwiseMult(GpuMat &A, const GpuMat &B)
{
	if(A.rows != B.rows || A.cols != B.cols || A.type != B.type){
		throw new std::invalid_argument("elemwiseMuls");
	}

	cuda_elemwiseMulA(A, B);
}

void elemwiseDiv(const GpuMat &A, const GpuMat &B, GpuMat &C)
{
	if(A.rows != B.rows || A.cols != B.cols || A.type != B.type){
		throw new std::invalid_argument("elemwiseDiv");
	}

	if(C.rows != A.rows || C.cols != A.cols || C.type != A.type)
		C.resize(A);

	cuda_elemwiseDiv(A, B, C);
}


void transpose(const GpuMat &A, GpuMat &C)
{
	if(A.empty()){
		throw new std::invalid_argument("transpose");
	}

	if(C.rows != A.cols || C.cols != A.rows || C.type != A.type)
		C.resize(A.cols, A.rows, A.type);

	cuda_transpose(A, C);

}

void elemwiseSqrt(const GpuMat &A, GpuMat &C)
{
	if(A.empty()){
		throw new std::invalid_argument("elemwiseSqrt");
	}

	if(C.rows != A.rows || C.cols != A.cols || C.type != A.type)
		C.resize(A);

	cuda_elemwiseSqrt(A, C);
}

void elemwiseSqr(const GpuMat &A, GpuMat &C)
{
	if(A.empty()){
		throw new std::invalid_argument("elemwiseSqr");
	}

	if(C.rows != A.rows || C.cols != A.cols || C.type != A.type)
		C.resize(A);

	cuda_elemwiseSqr(A, C);
}

void reLu(const GpuMat &A, GpuMat &C)
{
	if(A.empty()){
		throw new std::invalid_argument("reLu");
	}

	if(C.rows != A.rows || C.cols != A.cols || C.type != A.type)
		C.resize(A);

	cuda_reLu(A, C);
}

void reLu(GpuMat &A)
{
	if(A.empty()){
		throw new std::invalid_argument("reLu");
	}

	cuda_reLu2(A);
}

void deriv_reLu(const GpuMat &A, GpuMat &C)
{
	if(A.empty()){
		throw new std::invalid_argument("deriv_reLu");
	}

	if(C.rows != A.rows || C.cols != A.cols || C.type != A.type)
		C.resize(A);

	cuda_derivReLu(A, C);
}

void deriv_reLu(GpuMat &A)
{
	if(A.empty()){
		throw new std::invalid_argument("deriv_reLu");
	}


	cuda_derivReLu2(A);
}

void sigmoid(const GpuMat &A, GpuMat &C)
{
	if(A.empty()){
		throw new std::invalid_argument("sigmoid");
	}

	if(C.rows != A.rows || C.cols != A.cols || C.type != A.type)
		C.resize(A);

	cuda_sigmoid(A, C);
}

void sigmoid(GpuMat &A)
{
	if(A.empty()){
		throw new std::invalid_argument("sigmoid");
	}

	cuda_sigmoid2(A);
}

void deriv_sigmoid(const GpuMat &A, GpuMat &C)
{
	if(A.empty()){
		throw new std::invalid_argument("deriv_sigmoid");
	}

	if(C.rows != A.rows || C.cols != A.cols || C.type != A.type)
		C.resize(A);

	cuda_deriv_sigmoid(A, C);
}

void deriv_sigmoid(GpuMat &A)
{
	if(A.empty()){
		throw new std::invalid_argument("deriv_sigmoid");
	}

	cuda_deriv_sigmoid2(A);
}

void tanh(const GpuMat &A, GpuMat &C)
{
	if(A.empty()){
		throw new std::invalid_argument("tanh");
	}

	if(C.rows != A.rows || C.cols != A.cols || C.type != A.type)
		C.resize(A);

	cuda_tanh(A, C);
}

void tanh(GpuMat &A)
{
	if(A.empty()){
		throw new std::invalid_argument("tanh");
	}

	cuda_tanh2(A);
}

void deriv_tanh(const GpuMat &A, GpuMat &C)
{
	if(A.empty()){
		throw new std::invalid_argument("deriv_tanh");
	}

	if(C.rows != A.rows || C.cols != A.cols || C.type != A.type)
		C.resize(A);

	cuda_deriv_tanh(A, C);
}

void deriv_tanh(GpuMat &A)
{
	if(A.empty()){
		throw new std::invalid_argument("deriv_tanh");
	}

	cuda_deriv_tanh2(A);
}

void softmax(const GpuMat &A, int axis, GpuMat &C, GpuMat &partZ)
{
	if(A.empty()){
		throw new std::invalid_argument("softmax");
	}

	if(C.rows != A.rows || C.cols != A.cols || C.type != A.type)
		C.resize(A);

	if(axis == 0){
		if(partZ.cols != A.cols || partZ.rows != 1){
			partZ.resize(1, A.cols, A.type);
		}
	}
	if(axis == 1){
		if(partZ.rows != A.rows || partZ.cols != 1){
			partZ.resize(A.rows, 1, A.type);
		}
	}
	cuda_softmax(A, axis, C, partZ);
}


void softmax(GpuMat &A, int axis, GpuMat &partZ)
{
	if(A.empty()){
		throw new std::invalid_argument("softmax");
	}

	if(axis == 0){
		if(partZ.cols != A.cols || partZ.rows != 1){
			partZ.resize(1, A.cols, A.type);
		}
	}
	if(axis == 1){
		if(partZ.rows != A.rows || partZ.cols != 1){
			partZ.resize(A.rows, 1, A.type);
		}
	}
	cuda_softmax2(A, axis, partZ);
}


void sumRows(const GpuMat &A, GpuMat &C, double val)
{
	if(A.empty()){
		throw new std::invalid_argument("sumRows");
	}

	if(C.rows != 1 || C.cols != A.cols || A.type != C.type){
		C.resize(1, A.cols, A.type);
	}

	cuda_sumrows(A, C, val);
}

void sumRows_shared(const GpuMat &A, GpuMat &C, double val)
{
	if(A.empty()){
		throw new std::invalid_argument("sumRows_shared");
	}

	if(C.rows != 1 || C.cols != A.cols || A.type != C.type){
		C.resize(1, A.cols, A.type);
	}

	cuda_sumrows_shared(A, C, val);
}

void sub_adamGrad(GpuMat &A, const GpuMat &mA, const GpuMat &vA, double alpha, double sb1, double sb2)
{
	if(A.empty() || mA.empty() || vA.empty() ||
			A.type != mA.type || A.type != vA.type ||
			A.rows != mA.rows || A.cols != mA.cols ||
			A.rows != vA.rows || A.cols != vA.cols){
		throw new std::invalid_argument("sub_adamGrad");
	}

	cuda_adamgrad(A, mA, vA, alpha, sb1, sb2);
}

void subIndOne(const GpuMat &A, const GpuMat &Ind, GpuMat &B)
{
	if(A.empty() || Ind.empty() || A.rows != Ind.rows || Ind.cols != 1){
		throw new std::invalid_argument("subIndOne");
	}

	B.resize(A);

	cuda_subIndOne(A, Ind, B);
}

}

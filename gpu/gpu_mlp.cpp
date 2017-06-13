#include "gpu_mlp.h"

#include "matops.h"

using namespace gpumat;

template< typename T >
inline void init(GpuMat& mat, T n){
	ct::Mat_<T> m;
	m.setSize(mat.rows, mat.cols);
	m.randn(0, n);
	gpumat::convert_to_gpu(m, mat);
}

////////////////////////////////////////////////

mlp::mlp(){
	m_func = RELU;
	m_init = false;
	m_is_dropout = false;
	m_prob = 0.95;
	pA0 = nullptr;
	m_lambda = 0.;
}

void mlp::setLambda(double val)
{
	m_lambda = val;
}

void mlp::setDropout(bool val)
{
	m_is_dropout = val;
}

void mlp::setDropout(double val)
{
	m_prob = val;
}

bool mlp::isInit() const
{
	return m_init;
}

void mlp::init(int input, int output, int type){
	double n = 1./sqrt(input);

	W.resize(input, output, type);
	B.resize(output, 1, type);

	switch (type) {
		case GPU_DOUBLE:
			::init<double>(W, n);
			::init<double>(B, n);
			break;
		case GPU_FLOAT:
			::init<float>(W, (float)n);
			::init<float>(B, (float)n);
			break;
	}

	m_init = true;
}

void mlp::apply_func(const GpuMat &Z, GpuMat &A, etypefunction func){
	switch (func) {
		default:
		case RELU:
			reLu(Z, A);
			break;
		case SOFTMAX:
			softmax(Z, 1, A, PartZ);
			break;
		case SIGMOID:
			sigmoid(Z, A);
			break;
		case TANH:
			tanh(Z, A);
			break;
	}
}

void mlp::apply_back_func(const GpuMat &D1, GpuMat &D2, etypefunction func){
	switch (func) {
		default:
		case RELU:
			deriv_reLu(A1, DA1);
			break;
		case SOFTMAX:
			//				A = softmax(A, 1);
			D1.copyTo(D2);
			return;
		case SIGMOID:
			deriv_sigmoid(A1, DA1);
			break;
		case TANH:
			deriv_tanh(A1, DA1);
			break;
	}
	elemwiseMult(D1, DA1, D2);
}

etypefunction mlp::funcType() const{
	return m_func;
}

inline void apply_dropout(const GpuMat& X, double prob, GpuMat& XDropout, GpuMat& Dropout)
{
	switch (X.type) {
		case GPU_DOUBLE:{
			ct::Matd _Dropout;
			ct::dropout(X.rows, X.cols, prob, _Dropout);
			convert_to_gpu(_Dropout, Dropout);
			elemwiseMult(X, Dropout, XDropout);
			break;
		}
		case GPU_FLOAT:{
			ct::Matf _Dropout;
			ct::dropout(X.rows, X.cols, (float)prob, _Dropout);
			convert_to_gpu(_Dropout, Dropout);
			elemwiseMult(X, Dropout, XDropout);
			break;
		}
	}
}

void mlp::forward(const GpuMat *mat, etypefunction func, bool save_A0)
{
	if(!m_init || !mat)
		throw new std::invalid_argument("mlp::forward: not initialized. wrong parameters");
	pA0 = (GpuMat*)mat;
	m_func = func;

	if(m_is_dropout && std::abs(m_prob - 1) > 1e-6){
		apply_dropout(*pA0, m_prob, XDropout, Dropout);
		matmul(XDropout, W, A1);
	}else{
		matmul(*pA0, W, A1);
	}

	biasPlus(A1, B);
	apply_func(A1, A1, func);

	if(!save_A0)
		pA0 = nullptr;
}

void mlp::backward(const GpuMat &Delta, bool last_layer)
{
	if(!pA0 || !m_init)
		throw new std::invalid_argument("mlp::backward: not initialized. wrong parameters");

//	apply_back_func(Delta, DA1, m_func);

	double m = Delta.rows;

	gpumat::GpuMat* pDA1 = &DA1;

	if(m_func != gpumat::SOFTMAX){
		apply_back_func(Delta, DA1, m_func);
	}else{
		pDA1 = (GpuMat*)&Delta;
	}

	if(m_is_dropout && std::abs(m_prob - 1) > 1e-6){
		matmulT1(XDropout, *pDA1, gW);
	}else{
		matmulT1(*pA0, *pDA1, gW);
	}
	mulval(gW, 1. / m);


	if(m_lambda > 0){
		gpumat::add(gW, W, 1, m_lambda / m);
	}

	gB.swap_dims();
	sumRows(*pDA1, gB, 1.f / m);
	gB.swap_dims();

	if(!last_layer){
		matmulT2(*pDA1, W, DltA0);
	}
}

void mlp::write(std::fstream &fs)
{
	gpumat::write_fs(fs, W);
	gpumat::write_fs(fs, B);
}

void mlp::read(std::fstream &fs)
{
	gpumat::read_fs(fs, W);
	gpumat::read_fs(fs, B);
}

///**************************

bool MlpOptim::init(const std::vector<mlp> &_mlp)
{
	if(_mlp.empty())
		return false;

	m_iteration = 0;

	m_mW.resize(_mlp.size());
	m_mb.resize(_mlp.size());

	m_vW.resize(_mlp.size());
	m_vb.resize(_mlp.size());

	sW.resize(_mlp.size());
	sB.resize(_mlp.size());

	for(size_t i = 0; i < _mlp.size(); i++){
		const gpumat::mlp& _mlpi = _mlp[i];
		m_mW[i].resize(_mlpi.W);
		m_vW[i].resize(_mlpi.W);
		m_mW[i].zeros();
		m_vW[i].zeros();

		m_mb[i].resize(_mlpi.B);
		m_vb[i].resize(_mlpi.B);
		m_mb[i].zeros();
		m_vb[i].zeros();
	}
	m_init_matB = true;

	return true;
}

bool MlpOptim::pass(std::vector<mlp> &_mlp)
{
	if(_mlp.empty())
		return false;

	if(!m_init_matB){
		throw new std::invalid_argument("MlpOptim::pass: not initialized");
	}

	m_iteration++;
	double sb1 = (1. / (1. - pow(m_betha1, m_iteration)));
	double sb2 = (1. / (1. - pow(m_betha2, m_iteration)));

	for(size_t i = 0; i < _mlp.size(); ++i){
		mlp& mlpi = _mlp[i];

		gpumat::add(m_mW[i], mlpi.gW, m_betha1, (1. - m_betha1));
		gpumat::add(m_mb[i], mlpi.gB, m_betha1, (1. - m_betha1));

		gpumat::elemwiseSqr(mlpi.gW, sW[i]);
		gpumat::elemwiseSqr(mlpi.gB, sB[i]);

		gpumat::add(m_vW[i], sW[i], m_betha2, (1. - m_betha2));
		gpumat::add(m_vb[i], sB[i], m_betha2, (1. - m_betha2));

		/// W = -alpha * (sb1 * mW / (sqrt(sb2 * vW) + eps))

//		gpumat::add(W[i], m_mW[i], 1, -m_alpha);
//		gpumat::add(b[i], m_mb[i], 1, -m_alpha);
		gpumat::sub_adamGrad(mlpi.W, m_mW[i], m_vW[i], m_alpha, sb1, sb2);
		gpumat::sub_adamGrad(mlpi.B, m_mb[i], m_vb[i], m_alpha, sb1, sb2);
	}
	return true;
}

/////////////////

#include "convnn_gpu.h"

template<typename T >
void _norm(GpuMat& A, T &s)
{
	gpumat::GpuMat res;
	ct::Mat_<T> mA;

	gpumat::reduce(A, res);

	gpumat::convert_to_mat(res, mA);
	s = sqrt(mA.at(0));
	std::cout << s << std::endl;
}

template<typename T >
void _maxnorm(GpuMat& A, double c, double s)
{
	if(s > c){
		gpumat::mulval(A, c / s);
	}
}

void gpumat::maxnorm(GpuMat &A, double c)
{
	GpuMat S;
	gpumat::elemwiseSqr(A, S);

	switch (A.type) {
		case GPU_DOUBLE:
		{
			double s;
			_norm<double>(S, s);
			_maxnorm<double>(A, c, s);
			break;
		}
		case GPU_FLOAT:
		default:
		{
			float s;
			_norm<float>(S, s);
			_maxnorm<float>(A, c, s);
			break;
		}
	}
}

//////////////////////////////////////////////////////

bool MlpOptimSG::init(const std::vector<mlp> &_mlp)
{
	if(_mlp.empty())
		return false;

	return true;
}

bool MlpOptimSG::pass(std::vector<mlp> &_mlp)
{
	m_iteration++;
	for(size_t i = 0; i < _mlp.size(); ++i){
		mlp& mlpi = _mlp[i];
		gpumat::sub(mlpi.W, mlpi.gW, 1., m_alpha);
		gpumat::sub(mlpi.B, mlpi.gB, 1., m_alpha);

	}
	return true;
}

//////////////////////////////////////////////////

bool MlpOptimMoment::init(const std::vector<mlp> &_mlp)
{
	if(_mlp.empty())
		return false;

	m_iteration = 0;

	m_mW.resize(_mlp.size());
	m_mb.resize(_mlp.size());

	for(size_t i = 0; i < _mlp.size(); i++){
		const gpumat::mlp& _mlpi = _mlp[i];
		m_mW[i].resize(_mlpi.W);
		m_mW[i].zeros();

		m_mb[i].resize(_mlpi.B);
		m_mb[i].zeros();
	}

	return true;

}

bool MlpOptimMoment::pass(std::vector<mlp> &_mlp)
{
	if(_mlp.empty())
		return false;

	m_iteration++;
	for(size_t i = 0; i < _mlp.size(); ++i){
		mlp& mlpi = _mlp[i];

		gpumat::add(m_mW[i], mlpi.gW, m_betha, (1. - m_betha));
		gpumat::add(m_mb[i], mlpi.gB, m_betha, (1. - m_betha));

		gpumat::sub(mlpi.W, m_mW[i], 1., m_alpha);
		gpumat::sub(mlpi.B, m_mb[i], 1., m_alpha);
	}
	return true;

}

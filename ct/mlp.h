#ifndef MLP_H
#define MLP_H

#include "custom_types.h"
#include "common_types.h"
#include "nn.h"

namespace ct{

template< typename T >
class mlp;

template< typename T >
class MlpOptim: public AdamOptimizer<T>{
public:
	MlpOptim(): AdamOptimizer<T>(){

	}

#define AO this->

	bool init(std::vector< ct::mlp<T> >& Mlp){
		if(Mlp.empty())
			return false;

		AO m_iteration = 0;

		AO m_mW.resize(Mlp.size());
		AO m_mb.resize(Mlp.size());

		AO m_vW.resize(Mlp.size());
		AO m_vb.resize(Mlp.size());

		for(size_t i = 0; i < Mlp.size(); i++){
			ct::mlp<T>& _mlp = Mlp[i];
			AO m_mW[i].setSize(_mlp.W.size());
			AO m_vW[i].setSize(_mlp.W.size());
			AO m_mW[i].fill(0);
			AO m_vW[i].fill(0);

			AO m_mb[i].setSize(_mlp.B.size());
			AO m_vb[i].setSize(_mlp.B.size());
			AO m_mb[i].fill(0);
			AO m_vb[i].fill(0);
		}
		AO m_init = true;
		return true;
	}

	bool pass(std::vector< ct::mlp<T> >& Mlp){

		using namespace ct;

		AO m_iteration++;
		T sb1 = (T)(1. / (1. - pow(AO m_betha1, AO m_iteration)));
		T sb2 = (T)(1. / (1. - pow(AO m_betha2, AO m_iteration)));
		T eps = (T)(10e-8);

		for(size_t i = 0; i < Mlp.size(); ++i){
			ct::mlp<T>& _mlp = Mlp[i];
			AO m_mW[i] = AO m_betha1 * AO m_mW[i] + (T)(1. - AO m_betha1) * _mlp.gW;
			AO m_mb[i] = AO m_betha1 * AO m_mb[i] + (T)(1. - AO m_betha1) * _mlp.gB;

			AO m_vW[i] = AO m_betha2 * AO m_vW[i] + (T)(1. - AO m_betha2) * elemwiseSqr(_mlp.gW);
			AO m_vb[i] = AO m_betha2 * AO m_vb[i] + (T)(1. - AO m_betha2) * elemwiseSqr(_mlp.gB);

			Mat_<T> mWs = AO m_mW[i] * sb1;
			Mat_<T> mBs = AO m_mb[i] * sb1;
			Mat_<T> vWs = AO m_vW[i] * sb2;
			Mat_<T> vBs = AO m_vb[i] * sb2;

			vWs.sqrt(); vBs.sqrt();
			vWs += eps; vBs += eps;
			mWs = elemwiseDiv(mWs, vWs);
			mBs = elemwiseDiv(mBs, vBs);

			_mlp.W -= AO m_alpha * mWs;
			_mlp.B -= AO m_alpha * mBs;
		}
		return true;
	}
};

template< typename T >
class MlpOptimSG: public StohasticGradientOptimizer<T>{
public:
	MlpOptimSG(): StohasticGradientOptimizer<T>(){

	}
	bool pass(std::vector< ct::mlp<T> >& Mlp){
		if(Mlp.empty())
			return false;

		for(size_t i = 0; i < Mlp.size(); ++i){
			ct::mlp<T>& _mlp = Mlp[i];

			_mlp.W -= Optimizer<T>::m_alpha * _mlp.gW;
			_mlp.B -= Optimizer<T>::m_alpha * _mlp.gB;
		}

		return true;
	}
};

template< typename T >
class MlpOptimMoment: public MomentOptimizer<T>{
public:
	MlpOptimMoment(): MomentOptimizer<T>(){

	}
	bool init(std::vector< ct::mlp<T> >& Mlp){
		if(Mlp.empty())
			return false;

		Optimizer<T>::m_iteration = 0;

		Optimizer<T>::m_mW.resize(Mlp.size());
		Optimizer<T>::m_mb.resize(Mlp.size());

		for(size_t i = 0; i < Mlp.size(); i++){
			ct::mlp<T>& _mlp = Mlp[i];
			Optimizer<T>::m_mW[i].setSize(_mlp.W.size());
			Optimizer<T>::m_mW[i].fill(0);

			Optimizer<T>::m_mb[i].setSize(_mlp.B.size());
			Optimizer<T>::m_mb[i].fill(0);
		}
		return true;
	}

	bool pass(std::vector< ct::mlp<T> >& Mlp){
		if(Mlp.empty())
			return false;

		for(size_t i = 0; i < Mlp.size(); ++i){
			ct::mlp<T>& _mlp = Mlp[i];

			_mlp.W -= Optimizer<T>::m_alpha * _mlp.gW;
			_mlp.B -= Optimizer<T>::m_alpha * _mlp.gB;
		}

		return true;
	}
};

template< typename T >
class mlp{
public:
	Mat_<T> *pA0;
	Mat_<T> W;
	Mat_<T> B;
	Mat_<T> Z;
	Mat_<T> A1;
	Mat_<T> DA1;
	Mat_<T> D1;
	Mat_<T> DltA0;
	Mat_<T> Dropout;
	Mat_<T> XDropout;
	Mat_<T> gW;
	Mat_<T> gB;

	mlp(){
		m_func = RELU;
		m_init = false;
		m_is_dropout = false;
		m_prob = (T)0.95;
		pA0 = nullptr;
		m_lambda = 0;
	}

	void setLambda(T val){
		m_lambda = val;
	}

	void setDropout(bool val){
		m_is_dropout = val;
	}
	void setDropout(T val){
		m_prob = val;
	}

	bool isInit() const{
		return m_init;
	}

	void init(int input, int output){
		double n = 1./sqrt(input);

		W.setSize(input, output);
		W.randn(0., n);
		B.setSize(output, 1);
		B.randn(0, n);

		m_init = true;
	}

	inline void apply_func(const ct::Mat_<T>& Z, ct::Mat_<T>& A, etypefunction func){
		switch (func) {
			default:
			case RELU:
				A = relu(Z);
				break;
			case SOFTMAX:
				A = softmax(Z, 1);
				break;
			case SIGMOID:
				A = sigmoid(Z);
				break;
			case TANH:
				A = tanh(Z);
				break;
		}
	}
	inline void apply_back_func(const ct::Mat_<T>& D1, ct::Mat_<T>& D2, etypefunction func){
		switch (func) {
			default:
			case RELU:
				v_derivRelu(A1, DA1);
				break;
			case SOFTMAX:
//				A = softmax(A, 1);
				D1.copyTo(D2);
				return;
			case SIGMOID:
				v_derivSigmoid(A1, DA1);
				break;
			case TANH:
				v_derivTanh(A1, DA1);
				break;
		}
		elemwiseMult(D1, DA1, D2);
	}

	etypefunction funcType() const{
		return m_func;
	}

	void forward(const ct::Mat_<T> *mat, etypefunction func = RELU, bool save_A0 = true){
		if(!m_init || !mat)
			throw new std::invalid_argument("mlp::forward: not initialized. wrong parameters");
		pA0 = (Mat_<T>*)mat;
		m_func = func;

		if(m_is_dropout && std::abs(m_prob - 1) > 1e-6){
			ct::dropout(pA0->rows, pA0->cols, m_prob, Dropout);
			elemwiseMult(*pA0, Dropout, XDropout);
			Z = XDropout * W;
		}else{
			Z = *pA0 * W;
		}

		Z.biasPlus(B);
		apply_func(Z, A1, func);

		if(!save_A0)
			pA0 = nullptr;
	}
	void backward(const ct::Mat_<T> &Delta, bool last_layer = false){
		if(!pA0 || !m_init)
			throw new std::invalid_argument("mlp::backward: not initialized. wrong parameters");

		apply_back_func(Delta, DA1, m_func);

		T m = Delta.rows;

		if(m_is_dropout && std::abs(m_prob - 1) > 1e-6){
			matmulT1(XDropout, DA1, gW);
		}else{
			matmulT1(*pA0, DA1, gW);
		}
		gW *= (T) 1. / m;


		if(m_lambda > 0){
			gW += W * (m_lambda / m);
		}

		gB = (sumRows(DA1) * (1.f / m));
		gB.swap_dims();

		if(!last_layer){
			matmulT2(DA1, W, DltA0);
		}
	}

	void write(std::fstream& fs){
		write_fs(fs, W);
		write_fs(fs, B);
	}

	void read(std::fstream& fs){
		read_fs(fs, W);
		read_fs(fs, B);
	}

private:
	bool m_init;
	bool m_is_dropout;
	T m_prob;
	T m_lambda;
	etypefunction m_func;
};

}

#endif // MLP_H

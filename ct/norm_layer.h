#ifndef NORM_LAYER_H
#define NORM_LAYER_H

#include "custom_types.h"
#include "matops.h"

namespace ct{

/**
 * @brief The NL class
 * normalize layer
 */
template< typename T >
class NL{
public:
	std::vector<ct::Mat_<T>> *pX;
	std::vector<ct::Mat_<T>> A1;
	std::vector<ct::Mat_<T>> mX;
	std::vector<ct::Mat_<T>> stdX;
	T eps;

	NL(){
		pX = nullptr;
		eps = (T)1e-8;
	}

	void forward(const std::vector<ct::Mat_<T>>& X){
		if(X.empty())
			throw new std::invalid_argument("NL::forward: empty vector");
		pX = (std::vector<ct::Mat_<T>>*)&X;

		A1.resize(X.size());
		stdX.resize(X.size());
		mX.resize(X.size());
		for(size_t i = 0; i < mX.size(); ++i){
			ct::get_mean(X[i], mX[i], 0);
			ct::get_std(X[i], mX[i], stdX[i], 0);
			ct::get_norm(X[i], mX[i], stdX[i], A1[i], 0, eps);
		}
	}
	void backward(const std::vector<ct::Mat_<T>>& D){
		if(D.empty())
			throw new std::invalid_argument("NL::backward: empty vector");
	}
};

}

#endif // NORM_LAYER_H

#ifndef CONVNN_H
#define CONVNN_H

#include "custom_types.h"
#include "nn.h"

namespace ct{

template<typename T>
class convnn{
public:
	typedef std::vector< ct::Mat_<T> > tvmat;

	convnn(){
		stride = 1;
		weight_size = 3;
		m_init = false;
		A0 = nullptr;
		m_use_pool = true;
	}

	ct::Mat_<T> *A0;
	ct::Mat_<T> DltA0;
	tvmat A1;
	tvmat A2;
	tvmat W;
	tvmat prevW;
	std::vector< T > B;
	std::vector< T > prevB;
	tvmat Masks;
	ct::Size szA0;
	ct::Size szA1;
	ct::Size szA2;
	int stride;
	int weight_size;
	ct::AdamOptimizer<T> m_optim;

	tvmat gradW;
	std::vector< T > gradB;
	std::vector< ct::Mat_<T> > dA2, dA1;

	void setWeightSize(int ws, bool use_pool){
		weight_size = ws;
		if(W.size())
			init(W.size(), szA0, use_pool);
	}

	void init(size_t count_weight, const ct::Size& _szA0, bool use_pool){
		W.resize(count_weight);
		B.resize(count_weight);

		szA0 = _szA0;

		m_use_pool = use_pool;

		ct::get_cnv_sizes(szA0, ct::Size(weight_size, weight_size), stride, szA1, szA2);
//		if(!m_use_pool)
//			szA2 = szA1;

		update_random();

		m_init = true;
	}

	void save_weight(){
		prevW.resize(W.size());
		prevB.resize(B.size());
		for(int i = 0; i < W.size(); ++i){
			W[i].copyTo(prevW[i]);
			prevB[i] = B[i];
		}
	}
	void restore_weights(){
		if(prevW.empty() || prevB.empty())
			return;
		for(int i = 0; i < W.size(); ++i){
			prevW[i].copyTo(W[i]);
			B[i] = prevB[i];
		}
	}

	void update_random(){
		for(size_t i = 0; i < W.size(); ++i){
			W[i].setSize(weight_size, weight_size);
			W[i].randn(0, 0.1);
			B[i] = (T)0.1;
		}
	}

	void setAlpha(T alpha){
		m_optim.setAlpha(alpha);
	}

	void clear(){
		A0 = nullptr;
		A1.clear();
		A1.clear();
		Masks.clear();
	}

	bool use_pool() const{
		return m_use_pool;
	}

	ct::Size szOut() const{
		if(m_use_pool)
			return szA2;
		else
			return szA1;
	}

	bool forward(const ct::Mat_<T>* mat, ct::etypefunction func){
		if(!m_init || !mat)
			throw new std::invalid_argument("convnn::forward: not initialized");
		A0 = (ct::Mat_<T>*)mat;
		m_func = func;
		ct::conv2D(*A0, szA0, stride, W, B, A1, func);

		if(m_use_pool){
			ct::Size sztmp;
			bool res = ct::subsample(A1, szA1, A2, Masks, sztmp);
			return res;
		}else{
			return true;
		}
	}

	inline void apply_back(const ct::Mat_<T>& A1, const ct::Mat_<T>& dA2, ct::Mat_<T>& dA1, ct::etypefunction func)
	{
		switch (func) {
			case ct::LINEAR:
				ct::elemwiseMult(dA2, ct::derivRelu(A1), dA1);
				break;
			default:
			case ct::RELU:
				ct::elemwiseMult(dA2, ct::derivRelu(A1), dA1);
				break;
			case ct::SIGMOID:
				ct::elemwiseMult(dA2, ct::derivSigmoid(A1), dA1);
				break;
			case ct::TANH:
				ct::elemwiseMult(dA2, ct::derivTanh(A1), dA1);
				break;
		}
	}

	void back2conv(const tvmat& A1, const tvmat& dA2, tvmat& dA1, ct::etypefunction func){
		dA1.resize(A1.size());
		for(size_t i = 0; i < A1.size(); i++){
			apply_back(A1[i], dA2[i], dA1[i], func);
		}
	}

	void back2conv(const tvmat& A1, const tvmat& dA2, int first, tvmat& dA1, ct::etypefunction func){
		dA1.resize(A1.size());
		for(size_t i = 0; i < A1.size(); i++){
			apply_back(A1[i], dA2[i + first], dA1[i], func);
		}
	}

	void back2conv(const tvmat& A1, const std::vector< convnn >& dA2, int first, tvmat& dA1, ct::etypefunction func){
		dA1.resize(A1.size());
		for(size_t i = 0; i < A1.size(); i++){
			apply_back(A1[i], dA2[i + first].DltA0, dA1[i], func);
		}
	}

	void backward(const std::vector< ct::Mat_<T> >& Delta, int first = -1, int last = -1, bool last_layer = false){
		if(!m_init || !A0)
			throw new std::invalid_argument("convnn::backward: not initialized");

		if(m_use_pool){
			ct::upsample(Delta, szA2, szA1, Masks, dA2, first, last);
			back2conv(A1, dA2, dA1, m_func);
		}else{
			back2conv(A1, Delta, first, dA1, m_func);
		}

		ct::Size szW(weight_size, weight_size);

		ct::deriv_conv2D(*A0, dA1, szA0, szA1, szW, stride, gradW, gradB);

		if(!last_layer)
			ct::deriv_prev_cnv(dA1, W, szA1, szA0, DltA0);

		m_optim.pass(gradW, gradB, W, B);
	}

	void backward(const std::vector< convnn >& Delta, int first = -1, int last = -1, bool last_layer = false){
		if(!m_init || !A0)
			throw new std::invalid_argument("convnn::backward: not initialized");
		if(m_use_pool){
			convnn::upsample(Delta, szA2, szA1, Masks, dA2, first, last);
			back2conv(A1, dA2, dA1, m_func);
		}else{
			back2conv(A1, Delta, first, dA1, m_func);
		}

		ct::Size szW(weight_size, weight_size);

		ct::deriv_conv2D(*A0, dA1, szA0, szA1, szW, stride, gradW, gradB);

		if(!last_layer)
			ct::deriv_prev_cnv(dA1, W, szA1, szA0, DltA0);

		m_optim.pass(gradW, gradB, W, B);
	}

	static void hconcat(const std::vector< convnn<T> > &cnv, ct::Mat_<T>& _out){
		if(cnv.empty())
			return;
		std::vector< ct::Mat_<T> > slice;

		for(size_t i = 0; i < cnv.size(); ++i){
			ct::Mat_< T > res;
			ct::hconcat(cnv[i].A2, res);
			slice.push_back(res);
		}
		ct::hconcat(slice, _out);
	}

	void write(std::fstream& fs){
		for(size_t i = 0; i < W.size(); ++i){
			write_fs(fs, W[i]);
			fs.write((char*)&B[i], sizeof(T));
		}
	}

	void read(std::fstream& fs){
		for(size_t i = 0; i < W.size(); ++i){
			read_fs(fs, W[i]);
			fs.read((char*)&B[i], sizeof(T));
		}
	}

private:
	bool m_init;
	bool m_use_pool;
	ct::etypefunction m_func;

	bool upsample(const std::vector< convnn > &A1,
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
			if(!ct::upsample(A1[i].DltA0, szA1, szA0, Masks[j], A0[j]))
				return false;
		}
		return true;
	}

};

}

#endif // CONVNN_H

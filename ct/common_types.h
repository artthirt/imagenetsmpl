#ifndef COMMON_TYPES_H
#define COMMON_TYPES_H

#include <fstream>

namespace ct{

enum etypefunction{
	LINEAR,
	RELU,
	SOFTMAX,
	SIGMOID,
	TANH
};

struct Size{
	Size(){
		width = height = 0;
	}
	Size(int w, int h){
		width = w;
		height = h;
	}
	int area() const{
		return width * height;
	}
	Size t(){
		return Size(height, width);
	}

	int width;
	int height;
};

struct ParamsCommon{
	double prob;
	double lambda_l2;
	int count;

	ParamsCommon(){
		prob = 1.;
		lambda_l2 = 0;
		count = 0;
	}

	void write(std::fstream& fs) const{
		fs.write((char*)&prob, sizeof(prob));
		fs.write((char*)&lambda_l2, sizeof(lambda_l2));
		fs.write((char*)&count, sizeof(count));
	}
	void read(std::fstream& fs){
		fs.read((char*)&prob, sizeof(prob));
		fs.read((char*)&lambda_l2, sizeof(lambda_l2));
		fs.read((char*)&count, sizeof(count));
	}
};

struct ParamsMlp: public ParamsCommon{
	ParamsMlp():ParamsCommon(){
		count = 0;
		this->prob = 1;
		this->lambda_l2 = 0;
	}
	ParamsMlp(int val, double prob, double lambda_l2 = 0):ParamsCommon(){
		this->count = val;
		this->prob = prob;
		this->lambda_l2 = lambda_l2;
	}
};

struct ParamsCnv: public ParamsCommon{
	ParamsCnv():ParamsCommon(){
		size_w = 0;
		count = 0;
		pooling = true;
		prob = 1;
		lambda_l2 = 0.;
		stride = 1;
	}
	ParamsCnv(int size_w, int count_kernels, bool pooling, double prob, double lambda_l2, int stride = 1):ParamsCommon(){
		this->size_w = size_w;
		this->count = count_kernels;
		this->pooling = pooling;
		this->prob = prob;
		this->lambda_l2 = lambda_l2;
		this->stride = stride;
	}

	int size_w;
	bool pooling;
	int stride;

	void write(std::fstream& fs) const{
		ParamsCommon::write(fs);
		fs.write((char*)&size_w, sizeof(size_w));
		fs.write((char*)&pooling, sizeof(pooling));
	}
	void read(std::fstream& fs){
		ParamsCommon::read(fs);
		fs.read((char*)&size_w, sizeof(size_w));
		fs.read((char*)&pooling, sizeof(pooling));
	}
};

}

#endif // COMMON_TYPES_H

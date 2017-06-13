#ifndef HELPER_GPU_H
#define HELPER_GPU_H

#include "custom_types.h"
#include "gpumat.h"

#define PRINT_GMAT10(mat) {		\
	std::string s = mat.print(10);			\
	qDebug("%s\n", s.c_str());	\
}

namespace gpumat{

/**
 * @brief convert_to_gpu
 * @param mat
 * @param gmat
 */
void convert_to_gpu(const ct::Matf& mat, gpumat::GpuMat& gmat);
/**
 * @brief convert_to_gpu
 * @param mat
 * @param gmat
 */
void convert_to_gpu(const ct::Matd& mat, gpumat::GpuMat& gmat);
/**
 * @brief convert_to_mat
 * @param gmat
 * @param mat
 */
void convert_to_mat(const gpumat::GpuMat& gmat, ct::Matf& mat);
/**
 * @brief convert_to_mat
 * @param gmat
 * @param mat
 */
void convert_to_mat(const gpumat::GpuMat& gmat, ct::Matd& mat);

/**
 * @brief write_fs
 * write to fstream
 * @param fs
 * @param mat
 */
void write_fs(std::fstream &fs, const GpuMat &mat);

/**
 * @brief write_gmat
 * @param name
 * @param mat
 */
void write_gmat(const std::string &name, const GpuMat &mat);

/**
 * @brief read_fs
 * read from fstream
 * @param fs
 * @param mat
 */
void read_fs(std::fstream &fs, gpumat::GpuMat& mat);

/////////////////////////////////////////

class Optimizer{
public:
	Optimizer();
	virtual ~Optimizer();

	double alpha()const;

	void setAlpha(double v);

	uint32_t iteration() const;

	virtual bool init(const std::vector<GpuMat> &gradW, const std::vector<GpuMat> &gradB);
	virtual bool pass(const std::vector< gpumat::GpuMat >& gradW, const std::vector< gpumat::GpuMat >& gradB,
			  std::vector< gpumat::GpuMat >& W, std::vector< gpumat::GpuMat >& b);

protected:
	uint32_t m_iteration;
	double m_alpha;

private:
};

//////////////////////////////////////////

class StohasticGradientOptimizer: public Optimizer{
public:
	StohasticGradientOptimizer();

	virtual bool init(const std::vector<GpuMat> &gradW, const std::vector<GpuMat> &gradB);
	virtual bool pass(const std::vector< gpumat::GpuMat >& gradW, const std::vector< gpumat::GpuMat >& gradB,
			  std::vector< gpumat::GpuMat >& W, std::vector< gpumat::GpuMat >& b);

private:

};

//////////////////////////////////////////

class MomentumOptimizer: public Optimizer{
public:
	MomentumOptimizer();

	double betha() const;
	void setBetha(double b);

	virtual bool init(const std::vector<GpuMat> &gradW, const std::vector<GpuMat> &gradB);
	virtual bool pass(const std::vector< gpumat::GpuMat >& gradW, const std::vector< gpumat::GpuMat >& gradB,
			  std::vector< gpumat::GpuMat >& W, std::vector< gpumat::GpuMat >& b);

protected:
	double m_betha;
	std::vector< gpumat::GpuMat > m_mW;
	std::vector< gpumat::GpuMat > m_mb;
};

/////////////////////////////////////////

class AdamOptimizer: public Optimizer{
public:
	AdamOptimizer();


	double betha1() const;

	void setBetha1(double v);

	double betha2() const;

	void setBetha2(double v);


	bool empty() const;

	virtual bool init(const std::vector<GpuMat> &gradW, const std::vector<GpuMat> &gradB);
	void init_single(const std::vector<GpuMat> &gradW);

	virtual bool pass(const std::vector< gpumat::GpuMat >& gradW, const std::vector< gpumat::GpuMat >& gradB,
			  std::vector< gpumat::GpuMat >& W, std::vector< gpumat::GpuMat >& b);
	bool pass(const std::vector< gpumat::GpuMat >& gradW, const std::vector< float >& gradB,
			  std::vector< gpumat::GpuMat >& W, std::vector< float >& b);

protected:
	double m_betha1;
	double m_betha2;
	bool m_init_matB;
	bool m_init_singleB;

	std::vector< gpumat::GpuMat > sB, sW;

	std::vector< gpumat::GpuMat > m_mW;
	std::vector< gpumat::GpuMat > m_mb;
	std::vector< gpumat::GpuMat > m_vW;
	std::vector< gpumat::GpuMat > m_vb;

	std::vector< float > m_mb_single;
	std::vector< float > m_vb_single;
};

class SimpleAutoencoder
{
public:

	typedef void (*tfunc)(const GpuMat& _in, GpuMat& _out);

	SimpleAutoencoder();

	double m_alpha;
	int m_neurons;

	std::vector<GpuMat> W;
	std::vector<GpuMat> b;
	std::vector<GpuMat> dW;
	std::vector<GpuMat> db;

	tfunc func;
	tfunc deriv;

	void init(GpuMat& _W, GpuMat& _b, int samples, int neurons, tfunc fn, tfunc dfn);

	void pass(const GpuMat& X);
	double l2(const GpuMat& X);
private:
	AdamOptimizer adam;
	GpuMat a[3], tw1;
	GpuMat z[2], d, di, sz;
};

/**
 * @brief save_gmat
 * @param mat
 * @param fn
 */
void save_gmat(const GpuMat &mat, const std::string &fn);
/**
 * @brief save_gmat10
 * @param mat
 * @param fn
 */
void save_gmat10(const GpuMat& mat, const std::string& fn);

}

#endif // HELPER_GPU_H

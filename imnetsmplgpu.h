#ifndef IMNETSMPLGPU_H
#define IMNETSMPLGPU_H

#include "custom_types.h"
#include "gpumat.h"
#include "convnn2_gpu.h"
#include "gpu_mlp.h"

#include "imreader.h"

class ImNetSmplGpu
{
public:
	ImNetSmplGpu();

	void setReader(ImReader* ir);
	void setLearningRate(double lr);
	void setLayerFrom(int val);

	void init();
	void doPass(int pass, int batch);

	void forward(const std::vector< gpumat::GpuMat >& X, gpumat::GpuMat **pyOut, bool dropout = false);
	void backward(const gpumat::GpuMat& Delta);
	ct::Matf predict(gpumat::GpuMat &y);
	ct::Matf predict(const QString& name, bool show_debug = false);
	void predicts(const QString& sdir);
	float loss(const gpumat::GpuMat &y, const gpumat::GpuMat &y_);

	/**
	 * @brief setSaveModelName
	 * name for saved model
	 * @param name
	 */
	void setSaveModelName(const QString name);

	/**
	 * @brief save_net
	 * save only matrices of weights
	 * @param name
	 */
	void save_net(const QString& name);
	/**
	 * @brief load_net
	 * load only matrices of weights
	 * @param name
	 */
	void load_net(const QString& name);

	/**
	 * @brief save_net2
	 * save matrices with information about count layers and size of matrices
	 * @param name
	 */
	void save_net2(const QString& name);
	/**
	 * @brief load_net2
	 * load matrices with information about count layers and size of matrices
	 * @param name
	 */
	void load_net2(const QString& name);

	void setModelName(const QString& name);

	void setUseBackConv(bool val);

	void set_train(bool val);

	void check_delta(const gpumat::GpuMat& g_D, const Batch& btch);

private:
	ImReader *m_reader;
	double m_learningRate;
	int m_layer_from;

	bool m_useBackConv;

	std::vector< gpumat::convnn_gpu > m_conv;
	std::vector< gpumat::mlp > m_mlp;
	int m_classes;
	gpumat::GpuMat m_A1;
	gpumat::MlpOptimMoment m_optim;
	gpumat::CnvMomentumOptimizer m_cnv_optim;
	std::vector< gpumat::GpuMat > m_deltas;

	int m_check_count;
	QString m_model;
	QString m_save_model;

	bool m_init;

};

#endif // IMNETSMPLGPU_H

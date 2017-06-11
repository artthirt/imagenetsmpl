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

	void init();
	void doPass(int pass, int batch);

	void forward(const std::vector< gpumat::GpuMat >& X, gpumat::GpuMat **pyOut);
	void backward(const gpumat::GpuMat& Delta);
	ct::Matf predict(gpumat::GpuMat &y);
	ct::Matf predict(const QString& name, bool show_debug = false);
	void predicts(const QString& sdir);
	float loss(const gpumat::GpuMat &y, const gpumat::GpuMat &y_);

	void save_net(const QString& name);
	void load_net(const QString& name);

	void setModelName(const QString& name);

private:
	ImReader *m_reader;

	std::vector< gpumat::conv2::convnn_gpu > m_conv;
	std::vector< gpumat::mlp > m_mlp;
	int m_classes;
	gpumat::GpuMat m_A1;
	gpumat::MlpOptim m_optim;
	std::vector< gpumat::GpuMat > m_deltas;

	int m_check_count;
	QString m_model;

	bool m_init;

};

#endif // IMNETSMPLGPU_H

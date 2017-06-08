#include "imnetsmplgpu.h"

ImNetSmplGpu::ImNetSmplGpu()
{
	m_init = false;
}

void ImNetSmplGpu::setReader(ImReader *ir)
{
	m_reader = ir;
}

void ImNetSmplGpu::init()
{
	int W = ImReader::IM_WIDTH, H = ImReader::IM_HEIGHT;

	m_conv.resize(5);

	m_conv[0].init(ct::Size(W, H), 3, 2, 64, ct::Size(5, 5), false);
	m_conv[1].init(m_conv[0].szOut(), 64, 2, 128, ct::Size(5, 5), false);
	m_conv[2].init(m_conv[1].szOut(), 128, 2, 128, ct::Size(5, 5), false);
	m_conv[3].init(m_conv[2].szOut(), 128, 1, 256, ct::Size(5, 5));
	m_conv[4].init(m_conv[3].szOut(), 256, 1, 256, ct::Size(5, 5));

	qDebug("Out=[%dx%dx%d]", m_conv[4].szOut().width, m_conv[4].szOut().height, m_conv[4].K);

	int outFeatures = m_conv[4].outputFeatures();

	m_mlp.resize(2);

	m_mlp[0].init(outFeatures, 4096, gpumat::GPU_FLOAT);
	m_mlp[1].init(4096, m_classes, gpumat::GPU_FLOAT);

	m_optim.init(m_mlp);
	m_optim.setAlpha(0.001f);

	for(int i = 0; i < m_conv.size(); ++i){
		m_conv[i].setAlpha(0.001f);
	}

	m_init = true;
}

void get_gX(const std::vector< ct::Matf>& X, std::vector< gpumat::GpuMat >& gX)
{
	if(gX.size() != X.size())
		gX.resize(X.size());
	for(size_t i = 0; i < X.size(); ++i){
		gpumat::convert_to_gpu(X[i], gX[i]);
	}
}

void ImNetSmplGpu::doPass(int pass, int batch)
{
	if(!m_reader)
		return;

	if(!m_init)
		init();

	std::vector< gpumat::GpuMat > gX;
	gpumat::GpuMat gy, *gy_, gD;

	for(int i = 0; i < pass; ++i){
		std::vector< ct::Matf > X;
		ct::Matf y, y_;

		m_reader->get_batch(X, y, batch);

		get_gX(X, gX);
		gpumat::convert_to_gpu(y, gy);

		qDebug("--> pass %d", i);
		forward(gX, gy_);

		gpumat::subIndOne(*gy_, gy, gD);

		printf("--> backward\r");
		backward(gD);

		if((i % 5) == 0){
			std::vector< ct::Matf > X;
			ct::Matf y, y_, p;
			m_reader->get_batch(X, y, batch * 3);

			get_gX(X, gX);
			gpumat::convert_to_gpu(y, gy);

			forward(gX, gy_);

			float l = loss(gy, *gy_);
			p = predict(*gy_);
			double pr = check(y, p);
			qDebug("loss=%f;\tpred=%f", l, pr);
		}
		if((i % 20) == 0){
			save_net("model.bin");
		}
	}
}

void ImNetSmplGpu::forward(const std::vector<gpumat::GpuMat> &X, gpumat::GpuMat *yOut)
{
	m_conv[0].forward(&X, gpumat::RELU);
	m_conv[1].forward(&m_conv[0].XOut(), gpumat::RELU);
	m_conv[2].forward(&m_conv[1].XOut(), gpumat::RELU);
	m_conv[3].forward(&m_conv[2].XOut(), gpumat::RELU);
	m_conv[4].forward(&m_conv[3].XOut(), gpumat::RELU);

	gpumat::conv2::vec2mat(m_conv[4].XOut(), m_A1);

	m_mlp[0].forward(&m_A1);
	m_mlp[1].forward(&m_mlp[0].A1, gpumat::SOFTMAX);

	yOut = &m_mlp[1].A1;

}

void ImNetSmplGpu::backward(const gpumat::GpuMat &Delta)
{
	if(m_mlp.empty() || m_mlp[1].A1.empty())
		return;

	m_mlp[1].backward(Delta);
	m_mlp[0].backward(m_mlp[1].DltA0);

	gpumat::conv2::mat2vec(m_mlp[0].DltA0, m_conv[4].szK, m_deltas);

	printf("-cnv4        \r");
	m_conv[4].backward(m_deltas);
	printf("-cnv3        \r");
	m_conv[3].backward(m_conv[4].Dlt);
	printf("-cnv2        \r");
	m_conv[2].backward(m_conv[3].Dlt);
	printf("-cnv1        \r");
	m_conv[1].backward(m_conv[2].Dlt);
	printf("-cnv0        \r\n");
	m_conv[0].backward(m_conv[1].Dlt, true);

	m_optim.pass(m_mlp);
}

ct::Matf ImNetSmplGpu::predict(gpumat::GpuMat &y)
{
	return ct::Matf();
}

ct::Matf ImNetSmplGpu::predict(const QString &name, bool show_debug)
{
	return ct::Matf();
}

float ImNetSmplGpu::loss(const gpumat::GpuMat &y, const gpumat::GpuMat &y_)
{
	return -1;
}

void ImNetSmplGpu::save_net(const QString &name)
{

}

void ImNetSmplGpu::load_net(const QString &name)
{

}

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

void ImNetSmplGpu::doPass(int pass, int batch)
{

}

void ImNetSmplGpu::forward(const std::vector<ct::Matf> &X, ct::Matf &yOut)
{

}

void ImNetSmplGpu::backward(const ct::Matf &Delta)
{

}

ct::Matf ImNetSmplGpu::predict(ct::Matf &y)
{
	return ct::Matf();
}

ct::Matf ImNetSmplGpu::predict(const QString &name, bool show_debug)
{
	return ct::Matf();
}

float ImNetSmplGpu::loss(const ct::Matf &y, ct::Matf &y_)
{
	return -1;
}

void ImNetSmplGpu::save_net(const QString &name)
{

}

void ImNetSmplGpu::load_net(const QString &name)
{

}

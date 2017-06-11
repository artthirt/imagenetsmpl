#include "imnetsmplgpu.h"

#include <QDir>
#include <QFile>
#include <QDebug>

ImNetSmplGpu::ImNetSmplGpu()
{
	m_init = false;
	m_check_count = 200;
	m_classes = 1000;
	m_model = "model.bin";
}

void ImNetSmplGpu::setReader(ImReader *ir)
{
	m_reader = ir;
}

void ImNetSmplGpu::init()
{
	int W = ImReader::IM_WIDTH, H = ImReader::IM_HEIGHT;

	m_conv.resize(5);

	m_conv[0].init(ct::Size(W, H), 3, 4, 64, ct::Size(7, 7), false);
	m_conv[1].init(m_conv[0].szOut(), 64, 1, 128, ct::Size(3, 3));
	m_conv[2].init(m_conv[1].szOut(), 128, 1, 256, ct::Size(3, 3));
	m_conv[3].init(m_conv[2].szOut(), 256, 1, 512, ct::Size(3, 3));
	m_conv[4].init(m_conv[3].szOut(), 512, 1, 1024, ct::Size(3, 3), false);

	qDebug("Out=[%dx%dx%d]", m_conv[4].szOut().width, m_conv[4].szOut().height, m_conv[4].K);

	int outFeatures = m_conv[4].outputFeatures();

	m_mlp.resize(3);

	m_mlp[0].init(outFeatures, 4096, gpumat::GPU_FLOAT);
	m_mlp[1].init(4096, 2048, gpumat::GPU_FLOAT);
	m_mlp[2].init(2048, m_classes, gpumat::GPU_FLOAT);

	m_optim.init(m_mlp);
	m_optim.setAlpha(0.0001f);

	for(int i = 0; i < m_conv.size(); ++i){
		m_conv[i].setAlpha(0.0001);
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
		ct::Matf y;

		m_reader->get_batch(X, y, batch, true);

		get_gX(X, gX);
		gpumat::convert_to_gpu(y, gy);

//		qDebug("--> pass %d", i);
		forward(gX, &gy_);

//		gpumat::save_gmat(*gy_, "tmp1.txt");

		gpumat::subIndOne(*gy_, gy, gD);

//		printf("--> backward\r");
		backward(gD);

		if((i % 40) == 0){
			std::vector< ct::Matf > X;
			ct::Matf y, p;

			int idx = 0;
			double ls = 0, pr = 0;
			for(int i = 0; i < m_check_count; i += batch, idx++){
				m_reader->get_batch(X, y, batch);

				get_gX(X, gX);
				gpumat::convert_to_gpu(y, gy);
//				gpumat::save_gmat(gy, "tmp1.txt");
//				ct::save_mat(y, "tmp2.txt");

				forward(gX, &gy_);

				ls += loss(gy, *gy_);
				p = predict(*gy_);
				pr += check(y, p);
			}
			if(!idx)idx = 1;
			qDebug("pass %d: loss=%f;\tpred=%f", i, ls / idx, pr / idx);
		}
		if((i % 40) == 0){
			save_net(m_model);
		}
	}
}

void ImNetSmplGpu::forward(const std::vector<gpumat::GpuMat> &X, gpumat::GpuMat **pyOut)
{
	m_conv[0].forward(&X, gpumat::RELU);
	m_conv[1].forward(&m_conv[0].XOut(), gpumat::RELU);
	m_conv[2].forward(&m_conv[1].XOut(), gpumat::RELU);
	m_conv[3].forward(&m_conv[2].XOut(), gpumat::RELU);
	m_conv[4].forward(&m_conv[3].XOut(), gpumat::RELU);

	gpumat::conv2::vec2mat(m_conv[4].XOut(), m_A1);

	m_mlp[0].forward(&m_A1);
	m_mlp[1].forward(&m_mlp[0].A1);
	m_mlp[2].forward(&m_mlp[1].A1, gpumat::SOFTMAX);

	*pyOut = &m_mlp[2].A1;

}

void ImNetSmplGpu::backward(const gpumat::GpuMat &Delta)
{
	if(m_mlp.empty() || m_mlp[1].A1.empty())
		return;

	m_mlp[2].backward(Delta);
	m_mlp[1].backward(m_mlp[2].DltA0);
	m_mlp[0].backward(m_mlp[1].DltA0);

	gpumat::conv2::mat2vec(m_mlp[0].DltA0, m_conv[4].szK, m_deltas);

//	printf("-cnv4        \r");
	m_conv[4].backward(m_deltas);
//	printf("-cnv3        \r");
	m_conv[3].backward(m_conv[4].Dlt);
//	printf("-cnv2        \r");
	m_conv[2].backward(m_conv[3].Dlt);
//	printf("-cnv1        \r");
	m_conv[1].backward(m_conv[2].Dlt);
//	printf("-cnv0        \r\n");
	m_conv[0].backward(m_conv[1].Dlt, true);

	m_optim.pass(m_mlp);
}

ct::Matf ImNetSmplGpu::predict(gpumat::GpuMat &gy)
{
	ct::Matf res, y;
	gpumat::convert_to_mat(gy, y);

//	gpumat::save_gmat(gy, "tmp.txt");

	res.setSize(y.rows, 1);

	for(int i = 0; i < y.rows; ++i){
		res.ptr()[i] = y.argmax(i, 1);
	}
	return res;
}

ct::Matf ImNetSmplGpu::predict(const QString &name, bool show_debug)
{
	QString n = QDir::fromNativeSeparators(name);
//	qDebug() << n;

	if(!QFile::exists(n) || !m_reader)
		return ct::Matf();

	ct::Matf Xi = m_reader->get_image(n.toStdString()), y;
	std::vector< ct::Matf> X;
	X.push_back(Xi);

	std::vector< gpumat::GpuMat > gX;
	gpumat::GpuMat gy, *gy_, gD;

	get_gX(X, gX);

	forward(gX, &gy_);
	gpumat::convert_to_mat(*gy_, y);

	if(show_debug){
		int cls = y.argmax(0, 1);
		printf("--> predicted class %d\n", cls);
	}

	return y;
}

void ImNetSmplGpu::predicts(const QString &sdir)
{
	QString n = QDir::fromNativeSeparators(sdir);
	qDebug() << n;

	QDir dir(n);
	QStringList sl;
	sl << "*.jpg" << "*.jpeg" << "*.bmp" << "*.png" << "*.tiff";
	dir.setNameFilters(sl);

	printf("Start predicting. Count files %d\n", dir.count());

	std::cout << "predicted classes: ";

	for(int i= 0; i < dir.count(); ++i){
		QString s = dir.path() + "/" + dir[i];
		QFileInfo f(s);
		if(f.isFile()){
			ct::Matf y = predict(s, false);
			int cls = y.argmax(0, 1);
			std::cout << cls << ", ";
		}
	}
	std::cout << std::endl;
	printf("Stop predicting\n");
}


float ImNetSmplGpu::loss(const gpumat::GpuMat &y, const gpumat::GpuMat &y_)
{
	gpumat::GpuMat gr;
	gpumat::subIndOne(y_, y, gr);
	gpumat::elemwiseSqr(gr, gr);
	ct::Matf r;

	gpumat::convert_to_mat(gr, r);

	float f = r.sum() / r.rows;

	return f;
}

void ImNetSmplGpu::save_net(const QString &name)
{
	QString n = QDir::fromNativeSeparators(name);

	std::fstream fs;
	fs.open(n.toStdString(), std::ios_base::out | std::ios_base::binary);

	if(!fs.is_open()){
		qDebug("File %s not open", n.toLatin1().data());
		return;
	}

//	write_vector(fs, m_cnvlayers);
//	write_vector(fs, m_layers);

//	fs.write((char*)&m_szA0, sizeof(m_szA0));

	for(size_t i = 0; i < m_conv.size(); ++i){
		gpumat::conv2::convnn_gpu &cnv = m_conv[i];
		cnv.write(fs);
	}

	for(size_t i = 0; i < m_mlp.size(); ++i){
		m_mlp[i].write(fs);
	}

	printf("model saved.\n");

}

void ImNetSmplGpu::load_net(const QString &name)
{
	QString n = QDir::fromNativeSeparators(name);

	std::fstream fs;
	fs.open(n.toStdString(), std::ios_base::in | std::ios_base::binary);

	if(!fs.is_open()){
		qDebug("File %s not open", n.toLatin1().data());
		return;
	}

	m_model = n;

//	read_vector(fs, m_cnvlayers);
//	read_vector(fs, m_layers);

//	fs.read((char*)&m_szA0, sizeof(m_szA0));

//	setConvLayers(m_cnvlayers, m_szA0);

	init();

	for(size_t i = 0; i < m_conv.size(); ++i){
		gpumat::conv2::convnn_gpu &cnv = m_conv[i];
		cnv.read(fs);
	}

	for(size_t i = 0; i < m_mlp.size(); ++i){
		m_mlp[i].read(fs);
	}

	printf("model loaded.\n");

}

void ImNetSmplGpu::setModelName(const QString &name)
{
	if(!name.isEmpty())
		m_model = name;
}

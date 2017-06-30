#include "imnetsmplgpu.h"

#include <QDir>
#include <QFile>

const int cnv_size = 4;
const int mlp_size = 3;

ImNetSmplGpu::ImNetSmplGpu()
{
	m_learningRate =0.001;
	m_init = false;
	m_check_count = 600;
	m_classes = 1000;
	m_model = "model.bin";
}

void ImNetSmplGpu::setReader(ImReader *ir)
{
	m_reader = ir;
}

void ImNetSmplGpu::setLearningRate(double lr)
{
	m_learningRate = lr;
}

void ImNetSmplGpu::init()
{
	int W = ImReader::IM_WIDTH, H = ImReader::IM_HEIGHT;

	m_conv.resize(cnv_size);
	m_sg.resize(m_conv.size());

//	for(size_t i = 0; i < m_conv.size(); ++i){
//		m_conv[i].setOptimizer(&m_sg[i]);
//	}

	m_conv[0].init(ct::Size(W, H), 3, 4, 64, ct::Size(7, 7), true, false);
	m_conv[1].init(m_conv[0].szOut(), 64, 1, 256, ct::Size(5, 5), true);
	m_conv[2].init(m_conv[1].szOut(), 256, 1, 512, ct::Size(3, 3), true);
	m_conv[3].init(m_conv[2].szOut(), 512, 1, 1024, ct::Size(3, 3), false);
//	m_conv[4].init(m_conv[3].szOut(), 512, 1, 1024, ct::Size(3, 3), false);
//	m_conv[5].init(m_conv[4].szOut(), 1024, 1, 1024, ct::Size(3, 3));
//	m_conv[6].init(m_conv[5].szOut(), 512, 1, 1024, ct::Size(3, 3), false);

//	printf("Out=[%dx%dx%d]\n", m_conv.back().szOut().width, m_conv.back().szOut().height, m_conv.back().K);

	int outFeatures = m_conv.back().outputFeatures();

	m_mlp.resize(mlp_size);

	m_mlp[0].init(outFeatures, 4096, gpumat::GPU_FLOAT);
	m_mlp[1].init(4096, 2048, gpumat::GPU_FLOAT);
	m_mlp[2].init(2048, m_classes, gpumat::GPU_FLOAT);

	m_optim.init(m_mlp);
	m_optim.setAlpha(m_learningRate);

	for(int i = 0; i < m_conv.size(); ++i){
		m_conv[i].setAlpha(m_learningRate);
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
		std::cout << "pass " << i << "\r" << std::flush;

		std::vector< ct::Matf > X;
		ct::Matf y;

		m_reader->get_batch(X, y, batch, true);

		get_gX(X, gX);
		gpumat::convert_to_gpu(y, gy);

//		std::cout << "pass " << i << "\r";
//		qDebug("--> pass %d", i);
		forward(gX, &gy_);

//		gpumat::save_gmat(*gy_, "tmp1.txt");

		gpumat::subIndOne(*gy_, gy, gD);

//		printf("--> backward\r");
		backward(gD);

		if((i % 80) == 0){
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
			printf("pass %d: loss=%f;\tpred=%f\n", i, ls / idx, pr / idx);
		}
		if((i % 80) == 0){
			save_net(m_model);
			save_net2(m_model + "_ext");
		}
	}
}

void ImNetSmplGpu::forward(const std::vector<gpumat::GpuMat> &X, gpumat::GpuMat **pyOut)
{
	m_conv[0].forward(&X, gpumat::RELU);
	for(size_t i = 1; i < m_conv.size(); ++i){
		m_conv[i].forward(&m_conv[i - 1].XOut(), gpumat::RELU);
	}

	gpumat::conv2::vec2mat(m_conv.back().XOut(), m_A1);

	m_mlp[0].forward(&m_A1);
	m_mlp[1].forward(&m_mlp[0].A1);
	m_mlp[2].forward(&m_mlp[1].A1, gpumat::SOFTMAX);

	*pyOut = &m_mlp.back().A1;

}

void ImNetSmplGpu::backward(const gpumat::GpuMat &Delta)
{
	if(m_mlp.empty() || m_mlp.back().A1.empty())
		return;

	m_mlp.back().backward(Delta);
	m_mlp[1].backward(m_mlp[2].DltA0);
	m_mlp[0].backward(m_mlp[1].DltA0);

	gpumat::conv2::mat2vec(m_mlp[0].DltA0, m_conv.back().szK, m_deltas);

//	printf("-cnv4        \r");
	m_conv.back().backward(m_deltas);

	for(int i = m_conv.size() - 2; i >= 0; i--){
	//	printf("-cnv3        \r");
		m_conv[i].backward(m_conv[i + 1].Dlt, i == 0);
	//	printf("-cnv2        \r");
//		m_conv[2].backward(m_conv[3].Dlt);
//	//	printf("-cnv1        \r");
//		m_conv[1].backward(m_conv[2].Dlt);
//	//	printf("-cnv0        \r\n");
//		m_conv[0].backward(m_conv[1].Dlt, true);
	}

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
		for(int i = 0; i < y.cols; ++i){
			if(y.ptr()[i] < 0.1)
				y.ptr()[i] = 0;
		}

		int cls = y.argmax(0, 1);
		printf("--> predicted class %d\n", cls);
	}

	return y;
}

void ImNetSmplGpu::predicts(const QString &sdir)
{
	QString n = QDir::fromNativeSeparators(sdir);
	std::cout << n.toLatin1().data() << std::endl;

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
		printf("File %s not open\n", n.toLatin1().data());
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
		printf("File %s not open\n", n.toLatin1().data());
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

//////////////////////////

void ImNetSmplGpu::save_net2(const QString &name)
{
	QString n = QDir::fromNativeSeparators(name);

	std::fstream fs;
	fs.open(n.toStdString(), std::ios_base::out | std::ios_base::binary);

	if(!fs.is_open()){
		printf("File %s not open\n", n.toLatin1().data());
		return;
	}

//	write_vector(fs, m_cnvlayers);
//	write_vector(fs, m_layers);

//	fs.write((char*)&m_szA0, sizeof(m_szA0));

	int cnvs = m_conv.size(), mlps = m_mlp.size();

	/// size of convolution array
	fs.write((char*)&cnvs, sizeof(cnvs));
	/// size of mlp array
	fs.write((char*)&mlps, sizeof(mlps));

	for(size_t i = 0; i < m_conv.size(); ++i){
		gpumat::conv2::convnn_gpu &cnv = m_conv[i];
		cnv.write2(fs);
	}

	for(size_t i = 0; i < m_mlp.size(); ++i){
		m_mlp[i].write2(fs);
	}

	printf("model saved.\n");

}

void ImNetSmplGpu::load_net2(const QString &name)
{
	QString n = QDir::fromNativeSeparators(name);

	std::fstream fs;
	fs.open(n.toStdString(), std::ios_base::in | std::ios_base::binary);

	if(!fs.is_open()){
		printf("File %s not open\n", n.toLatin1().data());
		return;
	}

	m_model = n;

//	read_vector(fs, m_cnvlayers);
//	read_vector(fs, m_layers);

//	fs.read((char*)&m_szA0, sizeof(m_szA0));

//	setConvLayers(m_cnvlayers, m_szA0);

	init();

	int cnvs, mlps;

	/// size of convolution array
	fs.read((char*)&cnvs, sizeof(cnvs));
	/// size of mlp array
	fs.read((char*)&mlps, sizeof(mlps));

	printf("Load model: conv size %d, mlp size %d\n", cnvs, mlps);

	m_conv.resize(cnvs);
	m_mlp.resize(mlps);

	printf("conv\n");
	for(size_t i = 0; i < m_conv.size(); ++i){
		gpumat::conv2::convnn_gpu &cnv = m_conv[i];
		cnv.read2(fs);
		printf("layer %d: rows %d, cols %d\n", i, cnv.W[0].rows, cnv.W[0].cols);
	}

	printf("mlp\n");
	for(size_t i = 0; i < m_mlp.size(); ++i){
		gpumat::mlp &mlp = m_mlp[i];
		mlp.read2(fs);
		printf("layer %d: rows %d, cols %d\n", i, mlp.W.rows, mlp.W.cols);
	}

	printf("model loaded.\n");

}

//////////////////////////

void ImNetSmplGpu::setModelName(const QString &name)
{
	if(!name.isEmpty())
		m_model = name;
}

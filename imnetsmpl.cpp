#include "imnetsmpl.h"

#include <QDebug>
#include <QFile>
#include <QDir>
#include <QFileInfo>

#include "nn.h"
#include "convnn2.h"
#include "mlp.h"

#include <chrono>

const int cnv_size = 7;
const int mlp_size = 2;

ImNetSmpl::ImNetSmpl()
{
	m_check_count = 600;
	m_check_pass = 100;
	m_useBackConv = true;
	m_learningRate = 0.0001;
	m_reader = 0;
	m_classes = 1000;
	m_init = false;
	m_model = "model.bin";
	m_save_model = "model.bin_ext";
}

void ImNetSmpl::setReader(ImReader *ir)
{
	m_reader = ir;
}

void ImNetSmpl::setLearningRate(double lr)
{
	m_learningRate = lr;
}

void ImNetSmpl::init()
{
	int W = ImReader::IM_WIDTH, H = ImReader::IM_HEIGHT;

	m_conv.resize(cnv_size);
	m_mg.resize(cnv_size);

//	for(size_t i = 0; i < m_conv.size(); ++i){
//		m_conv[i].setOptimizer(&m_mg[i]);
//	}

	m_conv[0].init(ct::Size(W, H), 3, 2, 64, ct::Size(3, 3), ct::LEAKYRELU, false, false);
	m_conv[1].init(m_conv[0].szOut(), 64, 2, 64, ct::Size(3, 3), ct::LEAKYRELU, false);
	m_conv[2].init(m_conv[1].szOut(), 64, 2, 128, ct::Size(3, 3), ct::LEAKYRELU, false);
	m_conv[3].init(m_conv[2].szOut(), 128, 2, 128, ct::Size(3, 3), ct::LEAKYRELU, false);
	m_conv[4].init(m_conv[3].szOut(), 128, 1, 256, ct::Size(3, 3), ct::LEAKYRELU, false);
	m_conv[5].init(m_conv[4].szOut(), 256, 1, 512, ct::Size(3, 3), ct::LEAKYRELU, false);
	m_conv[6].init(m_conv[5].szOut(), 512, 1, 1024, ct::Size(3, 3), ct::LEAKYRELU, true);

//	printf("Out=[%dx%dx%d]\n", m_conv.back().szOut().width, m_conv.back().szOut().height, m_conv.back().K);

	int outFeatures = m_conv.back().outputFeatures();

	m_mlp.resize(mlp_size);

	m_mlp[0].init(outFeatures, 4096, ct::LEAKYRELU);
//	m_mlp[1].init(4096, 2048);
	m_mlp[1].init(4096, m_classes, ct::SOFTMAX);

	m_optim.init(m_mlp);
	m_optim.setAlpha(m_learningRate);

	for(int i = 0; i < m_conv.size(); ++i){
		m_conv[i].setAlpha(m_learningRate);
	}

	for(int i = 0; i < m_mlp.size(); ++i){
		m_mlp[i].setDropout(0.93f);
	}

	m_init = true;
}

void ImNetSmpl::doPass(int passes, int batch)
{
	if(!m_reader)
		return;

	if(!m_init)
		init();

	for(int pass = 0; pass < passes; ++pass){
		std::cout << "pass " << pass << "\r" << std::flush;

		std::vector< ct::Matf > X;
		ct::Matf y, y_;

		m_reader->get_batch(X, y, batch, true, true);

//		qDebug("--> pass %d", i);
		forward(X, y_, true);

		ct::Matf Dlt = ct::subIndOne(y_, y);

//		ct::save_mat(Dlt, "dlt.txt");

//		printf("--> backward\r");
		backward(Dlt);

		if((pass % m_check_pass) == 0 && pass > 0 || pass == 30){
			std::vector< ct::Matf > X;
			ct::Matf y, y_, p;

			int idx = 0;
			double ls = 0, pr = 0;
			for(int i = 0; i < m_check_count; i += batch, idx++){
				m_reader->get_batch(X, y, batch);

//				gpumat::save_gmat(gy, "tmp1.txt");
//				ct::save_mat(y, "tmp2.txt");

				forward(X, y_);

				ls += loss(y, y_);
				p = predict(y_);
				pr += check(y, p);

				printf("test: cur %d, all %d    \r", i, m_check_count);
				std::cout << std::flush;
			}
			if(!idx)idx = 1;
			printf("pass %d: loss=%f;\tpred=%f\n", pass, ls / idx, pr / idx);
		}
		if((pass % m_check_pass) == 0 && pass > 0){
			save_net2(m_save_model);
		}
	}
}

void ImNetSmpl::forward(const std::vector<ct::Matf> &X, ct::Matf &yOut, bool dropout)
{
	for(int i = 0; i < m_mlp.size(); ++i){
		m_mlp[i].setDropout(dropout);
	}

	m_conv[0].forward(&X);
	for(size_t i = 1; i < m_conv.size(); ++i){
		m_conv[i].forward(m_conv[i - 1]);
	}

	conv2::vec2mat(m_conv.back().XOut(), m_A1);

	ct::Matf *pX = &m_A1;
	for(size_t i = 0; i < m_mlp.size(); ++i){
		ct::mlp_mixed& mlp = m_mlp[i];
		mlp.forward(pX);
		pX = &mlp.A1;
//		m_mlp[0].forward(&m_A1);
//		m_mlp[1].forward(&m_mlp[0].A1);
//		m_mlp[2].forward(&m_mlp[1].A1, gpumat::SOFTMAX);
	}

	yOut = m_mlp.back().A1;
}

void ImNetSmpl::backward(const ct::Matf &Delta)
{
	if(m_mlp.empty() || m_mlp.back().A1.empty())
		return;

	ct::Matf *pX = (ct::Matf*)&Delta;
	for(int i = m_mlp.size() - 1; i >= 0; i--){
		ct::mlp_mixed& mlp = m_mlp[i];
		mlp.backward(*pX);
		pX = &mlp.DltA0;
//	m_mlp.back().backward(Delta);
//	m_mlp[1].backward(m_mlp[2].DltA0);
//	m_mlp[0].backward(m_mlp[1].DltA0);
	}

	if(m_useBackConv){
		conv2::mat2vec(m_mlp[0].DltA0, m_conv.back().szK, deltas1);
	//	conv2::mat2vec(D2, m_pool_1.szK, deltas2);

		m_conv.back().backward(deltas1);
		for(int i = m_conv.size() - 2; i >= 0; i--){
			m_conv[i].backward(m_conv[i + 1].Dlt, i == 0);
		}
	}

	m_optim.pass(m_mlp);
}

ct::Matf ImNetSmpl::predict(ct::Matf &y)
{
	ct::Matf res;

//	ct::save_mat(y, "tmp.txt");

	res.setSize(y.rows, 1);

	for(int i = 0; i < y.rows; ++i){
		res.ptr()[i] = y.argmax(i, 1);
	}
	return res;
}

ct::Matf ImNetSmpl::predict(const QString &name, bool show_debug)
{
	QString n = QDir::fromNativeSeparators(name);

	if(!QFile::exists(n) || !m_reader)
		return ct::Matf();

	ct::Matf Xi = m_reader->get_image(n.toStdString()), y;
	std::vector< ct::Matf> X;
	X.push_back(Xi);
	forward(X, y);

	if(show_debug){
		qDebug() << n;
		int cls = y.argmax(0, 1);
		QFileInfo f(n);
		printf("--> predicted class %d\n; file: %s", cls, f.fileName().toLatin1().data());
	}

	return y;
}

void ImNetSmpl::predicts(const QString &sdir)
{
	QString n = QDir::fromNativeSeparators(sdir);
	qDebug() << n;

	QDir dir(n);
	QStringList sl;
	sl << "*.jpg" << "*.jpeg" << "*.bmp" << "*.png" << "*.tiff";
	dir.setNameFilters(sl);

	printf("Start predicting. Count files %d\n", dir.count());

	std::cout << "predicted classes: ";

	std::map<int, int> stat;

	using namespace std::chrono;
	milliseconds ms = duration_cast< milliseconds >(
		system_clock::now().time_since_epoch()
	);

	for(int i= 0; i < dir.count(); ++i){
		QString s = dir.path() + "/" + dir[i];
		QFileInfo f(s);
		if(f.isFile()){
			ct::Matf y = predict(s, false);
			int cls = y.argmax(0, 1);
			int cnt = stat[cls];
			stat[cls] = cnt + 1;
			std::cout << cls << ", ";
		}
	}
	std::cout << std::endl;

	ms = duration_cast< milliseconds >(
		system_clock::now().time_since_epoch()
	) - ms;
	std::cout << std::endl;

	int all = dir.count();
	printf("\nStatisctics\n");
	for(std::map<int, int>::iterator it = stat.begin(); it != stat.end(); it++){
		int key = it->first;
		int val = it->second;
		printf("Pclass[%d]=%f\n", key, (double)val/all);
	}

	printf("Stop predicting. time=%f, fps=%f\n", (double)ms.count() / 1000., (double)all / ms.count() * 1000.);
}

float ImNetSmpl::loss(const ct::Matf &y, ct::Matf &y_)
{
	ct::Matf r = ct::subIndOne(y_, y);
	ct::v_elemwiseSqr(r);
	float f = r.sum() / r.rows;

	return f;
}

void ImNetSmpl::setSaveModelName(const QString name)
{
	m_save_model = name;
}

void ImNetSmpl::save_net(const QString &name)
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
		conv2::convnn2_mixed &cnv = m_conv[i];
		cnv.write(fs);
	}

	for(size_t i = 0; i < m_mlp.size(); ++i){
		m_mlp[i].write(fs);
	}

	printf("model saved.\n");
}

void ImNetSmpl::load_net(const QString &name)
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
		conv2::convnn2_mixed &cnv = m_conv[i];
		cnv.read(fs);
	}

	for(size_t i = 0; i < m_mlp.size(); ++i){
		m_mlp[i].read(fs);
	}

	printf("model loaded.\n");
}


//////////////////////////

void ImNetSmpl::save_net2(const QString &name)
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
		conv2::convnn2_mixed &cnv = m_conv[i];
		cnv.write2(fs);
	}

	for(size_t i = 0; i < m_mlp.size(); ++i){
		m_mlp[i].write2(fs);
	}

	printf("model saved.\n");

}

void ImNetSmpl::load_net2(const QString &name)
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

	printf("Load model: conv size %d, mlp size %d", cnvs, mlps);

	m_conv.resize(cnvs);
	m_mlp.resize(mlps);

	printf("conv\n");
	for(size_t i = 0; i < m_conv.size(); ++i){
		conv2::convnn2_mixed &cnv = m_conv[i];
		cnv.read2(fs);
		printf("layer %d: rows %d, cols %d\n", i, cnv.W[0].rows, cnv.W[0].cols);
	}

	printf("mlp\n");
	for(size_t i = 0; i < m_mlp.size(); ++i){
		ct::mlp_mixed &mlp = m_mlp[i];
		mlp.read2(fs);
		printf("layer %d: rows %d, cols %d\n", i, mlp.W.rows, mlp.W.cols);
	}

	printf("model loaded.\n");

}

////////////////////////////////

void ImNetSmpl::setModelName(const QString &name)
{
	if(!name.isEmpty())
		m_model = name;
}

void ImNetSmpl::setUseBackConv(bool val)
{
	m_useBackConv = val;
}

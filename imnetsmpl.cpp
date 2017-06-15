#include "imnetsmpl.h"

#include <QDebug>
#include <QFile>
#include <QDir>
#include <QFileInfo>

#include "nn.h"
#include "convnn2.h"
#include "mlp.h"

const int cnv_size = 4;
const int mlp_size = 3;

ImNetSmpl::ImNetSmpl()
{
	m_check_count = 50;
	m_learningRate = 0.0001;
	m_reader = 0;
	m_classes = 1000;
	m_init = false;
	m_model = "model.bin";
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

	for(size_t i = 0; i < m_conv.size(); ++i){
		m_conv[i].setOptimizer(&m_mg[i]);
	}

	m_conv[0].init(ct::Size(W, H), 3, 4, 64, ct::Size(7, 7), true, false);
	m_conv[1].init(m_conv[0].szOut(), 64, 1, 128, ct::Size(3, 3), true);
	m_conv[2].init(m_conv[1].szOut(), 128, 1, 256, ct::Size(3, 3), true);
	m_conv[3].init(m_conv[2].szOut(), 256, 1, 512, ct::Size(3, 3), false);

	m_pool_1.init(m_conv[1]);

//	printf("Out=[%dx%dx%d]\n", m_conv.back().szOut().width, m_conv.back().szOut().height, m_conv.back().K);

	int outFeatures = m_conv.back().outputFeatures() + m_pool_1.outputFeatures();

	m_mlp.resize(mlp_size);

	m_mlp[0].init(outFeatures, 2048);
	m_mlp[1].init(2048, 2048);
	m_mlp[2].init(2048, m_classes);

	m_optim.init(m_mlp);
	m_optim.setAlpha(m_learningRate);

	for(int i = 0; i < m_conv.size(); ++i){
		m_conv[i].setAlpha(m_learningRate);
	}

	m_init = true;
}

void ImNetSmpl::doPass(int pass, int batch)
{
	if(!m_reader)
		return;

	if(!m_init)
		init();

	for(int i = 0; i < pass; ++i){
		std::cout << "pass " << i << "\r" << std::flush;

		std::vector< ct::Matf > X;
		ct::Matf y, y_;

		m_reader->get_batch(X, y, batch);

//		qDebug("--> pass %d", i);
		forward(X, y_);

		ct::Matf Dlt = ct::subIndOne(y_, y);

//		printf("--> backward\r");
		backward(Dlt);

		if((i % 20) == 0){
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
			}
			if(!idx)idx = 1;
			printf("pass %d: loss=%f;\tpred=%f\n", i, ls / idx, pr / idx);
		}
		if((i % 40) == 0){
			save_net(m_model);
		}
	}
}

void ImNetSmpl::forward(const std::vector<ct::Matf> &X, ct::Matf &yOut)
{
//	m_conv[0].forward(&X, ct::RELU);
//	m_conv[1].forward(&m_conv[0].XOut(), ct::RELU);
//	m_conv[2].forward(&m_conv[1].XOut(), ct::RELU);
//	m_conv[3].forward(&m_conv[2].XOut(), ct::RELU);
//	m_conv[4].forward(&m_conv[3].XOut(), ct::RELU);
	m_conv[0].forward(&X, ct::RELU);
	for(size_t i = 1; i < m_conv.size(); ++i){
		m_conv[i].forward(m_conv[i - 1], ct::RELU);
	}

	m_pool_1.forward(m_conv[1]);
//	printf("pool out [%dx%dx%dx%d]\n", m_pool_1.szOut().width, m_pool_1.szOut().height, m_pool_1.szK.width, m_pool_1.szK.height);

//	conv2::vec2mat(m_conv.back().XOut(), m_A1);
//	conv2::vec2mat(m_pool_1.XOut(), m_A2);

//	std::vector< ct::Matf* > concat;

//	concat.push_back(&m_A1);
//	concat.push_back(&m_A2);

//	ct::hconcat(concat, m_Aout);

	m_concat.forward(&m_conv.back(), &m_pool_1);

//	m_mlp[0].forward(&m_A1);
//	m_mlp[1].forward(&m_mlp[0].A1, ct::SOFTMAX);
	m_mlp[0].forward(&m_concat.Y);
	m_mlp[1].forward(&m_mlp[0].A1);
	m_mlp[2].forward(&m_mlp[1].A1, ct::SOFTMAX);

	yOut = m_mlp.back().A1;
}

void ImNetSmpl::backward(const ct::Matf &Delta)
{
	if(m_mlp.empty() || m_mlp[2].A1.empty())
		return;

	m_mlp.back().backward(Delta);
	m_mlp[1].backward(m_mlp[2].DltA0);
	m_mlp[0].backward(m_mlp[1].DltA0);

//	std::vector< int > cols;
//	std::vector< ct::Matf* > mats;
//	cols.push_back(m_conv.back().outputFeatures());
//	cols.push_back(m_pool_1.outputFeatures());
//	mats.push_back(&D1);
//	mats.push_back(&D2);
//	ct::hsplit(m_mlp[0].DltA0, cols, mats);

//	conv2::mat2vec(D1, m_conv.back().szK, deltas1);
//	conv2::mat2vec(D2, m_pool_1.szK, deltas2);

	m_concat.backward(m_mlp[0].DltA0);

	m_pool_1.backward(m_concat.Dlt2);

	m_conv.back().backward(m_concat.Dlt1);
	for(int i = m_conv.size() - 2; i >= 0; i--){
		if(i == 1){
			conv2::convnn<float>& conv = m_conv[i + 1];
			if(m_pool_1.Dlt.size()){
				for(size_t i = 0; i < m_pool_1.Dlt.size(); ++i){
					conv.Dlt[i] += m_pool_1.Dlt[i];
				}
			}
		}
		m_conv[i].backward(m_conv[i + 1].Dlt, i == 0);
	}

//	printf("-cnv4        \r");
//	m_conv[4].backward(deltas);
//	printf("-cnv3        \r");
//	m_conv[3].backward(m_conv[4].Dlt);
//	printf("-cnv2        \r");
//	m_conv[2].backward(m_conv[3].Dlt);
//	printf("-cnv1        \r");
//	m_conv[1].backward(m_conv[2].Dlt);
//	printf("-cnv0        \r\n");
//	m_conv[0].backward(m_conv[1].Dlt, true);

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

float ImNetSmpl::loss(const ct::Matf &y, ct::Matf &y_)
{
	ct::Matf r = ct::subIndOne(y_, y);
	r = ct::elemwiseSqr(r);
	float f = r.sum() / r.rows;

	return f;
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
		conv2::convnn<float> &cnv = m_conv[i];
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
		conv2::convnn<float> &cnv = m_conv[i];
		cnv.read(fs);
	}

	for(size_t i = 0; i < m_mlp.size(); ++i){
		m_mlp[i].read(fs);
	}

	printf("model loaded.\n");
}

void ImNetSmpl::setModelName(const QString &name)
{
	if(!name.isEmpty())
		m_model = name;
}

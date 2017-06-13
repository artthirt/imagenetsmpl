#include "imnetsmpl.h"

#include <QDebug>
#include <QFile>
#include <QDir>
#include <QFileInfo>

#include "nn.h"
#include "convnn2.h"
#include "mlp.h"

ImNetSmpl::ImNetSmpl()
{
	m_learningRate =0.001;
	m_reader = 0;
	m_classes = 200;
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

	m_conv.resize(5);

	m_conv[0].init(ct::Size(W, H), 3, 2, 64, ct::Size(5, 5), false);
	m_conv[1].init(m_conv[0].szOut(), 64, 2, 128, ct::Size(5, 5), false);
	m_conv[2].init(m_conv[1].szOut(), 128, 2, 128, ct::Size(5, 5), false);
	m_conv[3].init(m_conv[2].szOut(), 128, 1, 256, ct::Size(5, 5));
	m_conv[4].init(m_conv[3].szOut(), 256, 1, 256, ct::Size(5, 5));

	qDebug("Out=[%dx%dx%d]", m_conv[4].szOut().width, m_conv[4].szOut().height, m_conv[4].K);

	int outFeatures = m_conv[4].outputFeatures();

	m_mlp.resize(2);

	m_mlp[0].init(outFeatures, 4096);
	m_mlp[1].init(4096, m_classes);

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
		std::vector< ct::Matf > X;
		ct::Matf y, y_;

		m_reader->get_batch(X, y, batch);

		qDebug("--> pass %d", i);
		forward(X, y_);

		ct::Matf Dlt = ct::subIndOne(y_, y);

		printf("--> backward\r");
		backward(Dlt);

		if((i % 5) == 0){
			std::vector< ct::Matf > X;
			ct::Matf y, y_, p;
			m_reader->get_batch(X, y, batch * 3);

			forward(X, y_);

			float l = loss(y, y_);
			p = predict(y_);
			double pr = check(y, p);
			qDebug("loss=%f;\tpred=%f", l, pr);
		}
		if((i % 20) == 0){
			save_net(m_model);
		}
	}
}

void ImNetSmpl::forward(const std::vector<ct::Matf> &X, ct::Matf &yOut)
{
	m_conv[0].forward(&X, ct::RELU);
	m_conv[1].forward(&m_conv[0].XOut(), ct::RELU);
	m_conv[2].forward(&m_conv[1].XOut(), ct::RELU);
	m_conv[3].forward(&m_conv[2].XOut(), ct::RELU);
	m_conv[4].forward(&m_conv[3].XOut(), ct::RELU);

	conv2::vec2mat(m_conv[4].XOut(), m_A1);

	m_mlp[0].forward(&m_A1);
	m_mlp[1].forward(&m_mlp[0].A1, ct::SOFTMAX);

	yOut =m_mlp[1].A1;
}

void ImNetSmpl::backward(const ct::Matf &Delta)
{
	if(m_mlp.empty() || m_mlp[1].A1.empty())
		return;

	m_mlp[1].backward(Delta);
	m_mlp[0].backward(m_mlp[1].DltA0);

	std::vector< ct::Matf > deltas;
	conv2::mat2vec(m_mlp[0].DltA0, m_conv[4].szK, deltas);

	printf("-cnv4        \r");
	m_conv[4].backward(deltas);
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

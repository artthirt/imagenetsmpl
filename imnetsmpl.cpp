#include "imnetsmpl.h"

#include <QDebug>
#include <QFile>
#include <QDir>

#include "nn.h"
#include "convnn2.h"
#include "mlp.h"

ImNetSmpl::ImNetSmpl()
{
	m_reader = 0;
	m_classes = 200;
	m_init = false;
}

void ImNetSmpl::setReader(ImReader *ir)
{
	m_reader = ir;
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
	m_optim.setAlpha(0.001f);

	for(int i = 0; i < m_conv.size(); ++i){
		m_conv[i].setAlpha(0.001f);
	}

	m_init = true;
}

double check(const ct::Matf& i1, const ct::Matf& i2)
{
	if(i1.empty() || i1.rows != i2.rows || i1.cols != 1 || i2.cols != 1)
		return -1.;

	int idx = 0;
	for(int i = 0; i < i1.rows; ++i){
		if(i1.ptr()[i] == i2.ptr()[i])
			idx++;
	}
	double pred = (double)idx / i1.rows;

	return pred;
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
			m_reader->get_batch(X, y, batch * 2);

			forward(X, y_);

			float l = loss(y, y_);
			p = predict(y_);
			double pr = check(y, p);
			qDebug("loss=%f;\tpred=%f", l, pr);
		}
		if((i % 100) == 0){
			save_net("model.bin");
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

	res.setSize(y.rows, 1);

	for(int i = 0; i < y.rows; ++i){
		res.ptr()[i] = y.argmax(i, 1);
	}
	return res;
}

ct::Matf ImNetSmpl::predict(const QString &name, bool show_debug)
{
	QString n = QDir::fromNativeSeparators(name);
	qDebug() << n;

	if(!QFile::exists(n) || !m_reader)
		return ct::Matf();

	ct::Matf Xi = m_reader->get_image(n.toStdString()), y;
	std::vector< ct::Matf> X;
	X.push_back(Xi);
	forward(X, y);

	if(show_debug){
		int cls = y.argmax(0, 1);
		printf("--> predicted class %d\n", cls);
	}

	return y;
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
}

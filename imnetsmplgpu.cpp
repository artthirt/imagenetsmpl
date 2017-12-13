#include "imnetsmplgpu.h"

#include <chrono>

#include <QDir>
#include <QFile>

const int cnv_size = 8;
const int mlp_size = 3;

//const int stop_cnv_layer = 6;

///////////////////////

template <typename T>
std::vector<size_t> sort_indexes(const std::vector<T> &v, int ascend = 0) {

  // initialize original index locations
  std::vector<size_t> idx(v.size());
  std::iota(idx.begin(), idx.end(), 0);

  if(ascend == 0){
  // sort indexes based on comparing values in v
	  std::sort(idx.begin(), idx.end(),
		   [&v](size_t i1, size_t i2) {return v[i1] < v[i2];});
  }else{
	  std::sort(idx.begin(), idx.end(),
		   [&v](size_t i1, size_t i2) {return v[i1] > v[i2];});
  }

  return idx;
}

struct SortC{
	SortC(){}
	SortC(float p, int index): p(p), index(index){}
	float p;
	int index;
};

std::vector< SortC > sort_column(const ct::Matf& mat, int row)
{
	std::vector< SortC > res;
	std::vector< SortC > sc;

	float *dF = mat.ptr(row);
	for(int i = 0; i < mat.cols; ++i){
		sc.push_back(SortC(dF[i], i));
	}
	std::sort(sc.begin(), sc.end(), [](const SortC& s1, const SortC& s2){return s1.p > s2.p;});

	res.resize(5);
	std::copy(sc.begin(), sc.begin() + 5, res.begin());
	return res;
}

double check2(const gpumat::GpuMat& prob, const ct::Matf& classes)
{
	if(classes.empty() || classes.rows != prob.rows || classes.cols != 1)
		return -1.;

	ct::Matf mp;
	gpumat::convert_to_mat(prob, mp);

	int idx = 0;
	for(int i = 0; i < classes.rows; ++i){
		std::vector< SortC > preds;
		preds = sort_column(mp, i);
		for(const SortC& s: preds){
			if(s.p > 0.1 && s.index == classes.ptr(i)[0]){
				idx++;
				break;
			}
		}
	}
	double pred = (double)idx / classes.rows;

//	std::cout << "predicted: " << ss.str() << std::endl;

	return pred;
}


///////////////////////

ImNetSmplGpu::ImNetSmplGpu()
{
	m_layer_from = 0;
	m_learningRate = 0.001;
	m_useBackConv = true;
	m_init = false;
	m_check_count = 600;
	m_classes = 1000;
	m_model = "model.bin";
	m_save_model = "model.bin_ext";
}

void ImNetSmplGpu::setReader(ImReader *ir)
{
	m_reader = ir;
}

void ImNetSmplGpu::setLearningRate(double lr)
{
	m_learningRate = lr;
	m_optim.setAlpha(m_learningRate);
	m_cnv_optim.setAlpha(m_learningRate);
}

void ImNetSmplGpu::setLayerFrom(int val)
{
	m_layer_from = val;
	m_cnv_optim.stop_layer = m_layer_from;
}

void ImNetSmplGpu::init()
{
	int W = ImReader::IM_WIDTH, H = ImReader::IM_HEIGHT;

	m_conv.resize(cnv_size);

	m_conv[0].init(ct::Size(W, H), 3, 4, 96, ct::Size(7, 7), gpumat::LEAKYRELU, false, true, false);
	m_conv[1].init(m_conv[0].szOut(), 96, 1, 128, ct::Size(3, 3), gpumat::LEAKYRELU, false, true, true);
	m_conv[2].init(m_conv[1].szOut(), 128, 2, 256, ct::Size(3, 3), gpumat::LEAKYRELU, false, true, true);
	m_conv[3].init(m_conv[2].szOut(), 256, 1, 256, ct::Size(3, 3), gpumat::LEAKYRELU, false, true, true);
	m_conv[4].init(m_conv[3].szOut(), 256, 2, 512, ct::Size(3, 3), gpumat::LEAKYRELU, false, true, true);
	m_conv[5].init(m_conv[4].szOut(), 512, 1, 512, ct::Size(3, 3), gpumat::LEAKYRELU, false, true, true);
	m_conv[6].init(m_conv[5].szOut(), 512, 1, 512, ct::Size(3, 3), gpumat::LEAKYRELU, false, true, true);
	m_conv[7].init(m_conv[6].szOut(), 512, 1, 512, ct::Size(3, 3), gpumat::LEAKYRELU, false, true, true);
//	m_conv[8].init(m_conv[7].szOut(), 512, 1, 512, ct::Size(3, 3), gpumat::LEAKYRELU, false, true, true);
//	m_conv[9].init(m_conv[8].szOut(), 512, 1, 512, ct::Size(3, 3), gpumat::LEAKYRELU, false, true, true);
//	m_conv[10].init(m_conv[9].szOut(), 512, 1, 512, ct::Size(3, 3), gpumat::LEAKYRELU, false, true, true);

//	printf("Out=[%dx%dx%d]\n", m_conv.back().szOut().width, m_conv.back().szOut().height, m_conv.back().K);

	int outFeatures = m_conv.back().outputFeatures();

	m_mlp.resize(mlp_size);

	m_mlp[0].init(outFeatures,	4096,		gpumat::GPU_FLOAT, gpumat::LEAKYRELU);
	m_mlp[1].init(4096,			4096,		gpumat::GPU_FLOAT, gpumat::LEAKYRELU);
	m_mlp[2].init(4096,			m_classes,	gpumat::GPU_FLOAT, gpumat::SOFTMAX);

	m_optim.init(m_mlp);
	m_optim.setAlpha(m_learningRate);

	m_cnv_optim.init(m_conv);
	m_cnv_optim.setAlpha(m_learningRate);

	m_cnv_optim.stop_layer = m_layer_from;

//	m_optim.setDelimiterIteration(16);
//	m_cnv_optim.setDelimiterIteration(16);

	for(int i = 0; i < m_conv.size(); ++i){
		m_conv[i].setDropout(0.7);
	}

	m_mlp[0].setDropout(0.7);
	m_mlp[1].setDropout(0.7);
	m_mlp[2].setDropout(1.);

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

	m_reader->set_params_batch(batch, true);
	m_reader->start();

	for(int i = 0; i < pass; ++i){
		std::cout << "pass " << i << "; batches in mem: " << m_reader->batches() << "     \r" << std::flush;

#if 1
		while(!m_reader->is_batch_exist()){
			std::this_thread::sleep_for(std::chrono::milliseconds(100));
		}
		Batch& btch = m_reader->front();
//		std::vector< ct::Matf >& X	= btch.X;
//		ct::Matf& y					= btch.y;

		get_gX(btch.X, gX);
		gpumat::convert_to_gpu(btch.y, gy);

#else
		std::vector< ct::Matf > X;
		ct::Matf y;
		m_reader->get_batch(X, y, batch, true, true);

		get_gX(X, gX);
		gpumat::convert_to_gpu(y, gy);
#endif

//		std::cout << "pass " << i << "\r";
//		qDebug("--> pass %d", i);
		forward(gX, &gy_, true);

//		gpumat::save_gmat(*gy_, "tmp1.txt");

		gpumat::subIndOne(*gy_, gy, gD);

		check_delta(gD, btch);

		m_reader->pop_front();
//		printf("--> backward\r");
		backward(gD);

		if((i % 200) == 0/* && i > 0*/ || i == 30){
			std::vector< ct::Matf > X;
			ct::Matf y, p;

			int idx = 0;
			double ls = 0, pr = 0, pr2 = 0;
			for(int i = 0; i <= m_check_count; i += batch, idx++){
				m_reader->get_batch(X, y, batch, false, false);

				get_gX(X, gX);
				gpumat::convert_to_gpu(y, gy);
//				gpumat::save_gmat(gy, "tmp1.txt");
//				ct::save_mat(y, "tmp2.txt");

				forward(gX, &gy_);

				ls += loss(gy, *gy_);
				p = predict(*gy_);
				pr += check(y, p);
				pr2 += check2(*gy_, y);

				printf("test: cur %d, all %d            \r", i, m_check_count);
				std::cout << std::flush;
			}
			if(!idx)idx = 1;
			printf("pass %d: loss=%f;\tpred=%f;\tpred2=%f           \n", i, ls/idx, pr/idx, pr2/idx);
		}
		if((i % 200) == 0 && i > 0){
//			save_net(m_model);
			save_net2(m_save_model);
		}
	}
}

void ImNetSmplGpu::forward(const std::vector<gpumat::GpuMat> &X, gpumat::GpuMat **pyOut, bool dropout)
{
//	for(int i = 0; i < m_conv.size(); ++i){
//		m_conv[i].setDropout(dropout);
//	}
//	m_conv[0].setDropout(dropout);
//	m_conv[1].setDropout(dropout);

//	set_train(dropout);

	m_mlp[0].setDropout(dropout);
	m_mlp[1].setDropout(dropout);
	m_mlp[2].setDropout(dropout);

	m_conv[0].forward(&X);
	for(size_t i = 1; i < m_conv.size(); ++i){
		m_conv[i].forward(&m_conv[i - 1].XOut());
	}

	gpumat::vec2mat(m_conv.back().XOut(), m_A1);

	gpumat::GpuMat *pX = &m_A1;
	for(size_t i = 0; i < m_mlp.size(); ++i){
		gpumat::mlp& mlp = m_mlp[i];
		mlp.forward(pX);
		pX = &mlp.A1;
//		m_mlp[0].forward(&m_A1);
//		m_mlp[1].forward(&m_mlp[0].A1);
//		m_mlp[2].forward(&m_mlp[1].A1, gpumat::SOFTMAX);
	}

	*pyOut = &m_mlp.back().A1;

}

void ImNetSmplGpu::backward(const gpumat::GpuMat &Delta)
{
	if(m_mlp.empty() || m_mlp.back().A1.empty())
		return;

	gpumat::GpuMat *pX = (gpumat::GpuMat*)&Delta;
	for(int i = m_mlp.size() - 1; i >= 0; i--){
		gpumat::mlp& mlp = m_mlp[i];
		mlp.backward(*pX);
		pX = &mlp.DltA0;
//	m_mlp.back().backward(Delta);
//	m_mlp[1].backward(m_mlp[2].DltA0);
//	m_mlp[0].backward(m_mlp[1].DltA0);
	}

	if(m_useBackConv){
		gpumat::mat2vec(m_mlp[0].DltA0, m_conv.back().szK, m_deltas);

	//	printf("-cnv4        \r");
		//m_conv.back().backward(m_deltas);

		std::vector< gpumat::GpuMat > *pD = &m_deltas;

		for(int i = m_conv.size() - 1; i >= m_layer_from; i--){
		//	printf("-cnv3        \r");
			m_conv[i].backward(*pD, i == m_layer_from);
			pD = &m_conv[i].Dlt;
		//	printf("-cnv2        \r");
	//		m_conv[2].backward(m_conv[3].Dlt);
	//	//	printf("-cnv1        \r");
	//		m_conv[1].backward(m_conv[2].Dlt);
	//	//	printf("-cnv0        \r\n");
	//		m_conv[0].backward(m_conv[1].Dlt, true);
		}
		m_cnv_optim.pass(m_conv);
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
		std::vector< int > numbers;
		std::vector< float > prob;
		for(int i = 0; i < y.cols; ++i){
			if(y.ptr()[i] < 0.1)
				y.ptr()[i] = 0;
			else{
				numbers.push_back(i);
				prob.push_back(y.ptr()[i]);
			}
		}

		int cls = y.argmax(0, 1);
		printf("--> predicted class %d\n", cls);

		printf("probs:\n");
		for(size_t i = 0; i < numbers.size(); ++i){
			std::cout << "cls=" << numbers[i] << "(" << prob[i] << "); ";
		}
		std::cout << std::endl;
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

void ImNetSmplGpu::setSaveModelName(const QString name)
{
	m_save_model = name;
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
		gpumat::convnn_gpu &cnv = m_conv[i];
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
		gpumat::convnn_gpu &cnv = m_conv[i];
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
		gpumat::convnn_gpu &cnv = m_conv[i];
		cnv.write2(fs);
	}

	for(size_t i = 0; i < m_mlp.size(); ++i){
		m_mlp[i].write2(fs);
	}

	int use_bn = 0, layers = 0;
	for(gpumat::convnn_gpu& item: m_conv){
		if(item.use_bn()){
			use_bn = 1;
			layers++;
		}
	}

	fs.write((char*)&use_bn, sizeof(use_bn));
	fs.write((char*)&layers, sizeof(layers));
	if(use_bn > 0){
		for(size_t i = 0; i < m_conv.size(); ++i){
			if(m_conv[i].use_bn()){
				fs.write((char*)&i, sizeof(i));
				m_conv[i].bn.write(fs);
			}
		}
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

#define USE_MLP 1

	if(m_conv.size() < cnvs)
		m_conv.resize(cnvs);
#if USE_MLP
	m_mlp.resize(mlps);
#endif
	printf("conv\n");
	for(size_t i = 0; i < cnvs; ++i){
		gpumat::convnn_gpu &cnv = m_conv[i];
		cnv.read2(fs);
		printf("layer %d: rows %d, cols %d\n", i, cnv.W.rows, cnv.W.cols);
	}

	printf("mlp\n");
	for(size_t i = 0; i < mlps; ++i){
#if USE_MLP
		gpumat::mlp &mlp = m_mlp[i];
		mlp.read2(fs);
		printf("layer %d: rows %d, cols %d\n", i, mlp.W.rows, mlp.W.cols);
#else
		gpumat::GpuMat W, B;
		gpumat::read_fs2(fs, W);
		gpumat::read_fs2(fs, B);
		printf("layer %d: rows %d, cols %d\n", i, W.rows, W.cols);
#endif
	}

	int use_bn = 0, layers = 0;
	fs.read((char*)&use_bn, sizeof(use_bn));
	fs.read((char*)&layers, sizeof(layers));
	if(use_bn > 0){
		for(int i = 0; i < layers; ++i){
			int64_t layer = -1;
			fs.read((char*)&layer, sizeof(layer));
			if(layer >=0 && layer < 10000){
				m_conv[layer].bn.read(fs);
//				gpumat::save_gmat(m_conv[layer].bn.gamma, "g" + std::to_string(layer) +".txt");
//				gpumat::save_gmat(m_conv[layer].bn.betha, "b" + std::to_string(layer) +".txt");
			}
		}
	}

	printf("model loaded.\n");

}

//////////////////////////

void ImNetSmplGpu::setModelName(const QString &name)
{
	if(!name.isEmpty())
		m_model = name;
}

void ImNetSmplGpu::setUseBackConv(bool val)
{
	m_useBackConv = val;
}

void ImNetSmplGpu::set_train(bool val)
{
	for(gpumat::convnn_gpu& item: m_conv){
		item.setTrainMode(val);
	}
}

void ImNetSmplGpu::check_delta(const gpumat::GpuMat &g_D, const Batch &btch)
{
	gpumat::GpuMat g_Out, g_R;
	gpumat::elemwiseSqr(g_D, g_Out);
	gpumat::sumCols(g_Out, g_R, 1);
	ct::Matf d;

//	gpumat::save_gmat(g_Out, "out.txt");
//	gpumat::save_gmat(g_R, "out1.txt");

	gpumat::convert_to_mat(g_R, d);

	std::vector< float > df;
	std::vector< size_t > idx;
	for(int i = 0; i < d.rows; ++i){
		float *dF = d.ptr(i);
		float F = dF[0];
		df.push_back(F);
	}
	idx = sort_indexes(df, 1);

	for(size_t i = 0; i < std::min((size_t)FOR_REPEAT_BATCH, idx.size()); ++i){
		int id = idx[i];
		float f = df[id];
        if(f > 0.5){
			m_reader->push_to_saved(btch.X[id], btch.y.ptr(id)[0]);
		}
    }
}

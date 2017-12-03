#include "imreader.h"

#include <QDir>
#include <QDebug>
#include <QFileInfo>

#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

#include "imnet_list.h"

#include <random>
#include <chrono>

static std::mt19937 _rnd;

///////////////////////

double check(const ct::Matf& classes, const ct::Matf& predicted)
{
	if(classes.empty() || classes.rows != predicted.rows || classes.cols != 1 || predicted.cols != 1)
		return -1.;

	std::stringstream ss;

	int idx = 0;
	for(int i = 0; i < classes.rows; ++i){
		ss << predicted.ptr()[i] << ", ";
		if(classes.ptr()[i] == predicted.ptr()[i])
			idx++;
	}
	double pred = (double)idx / classes.rows;

//	std::cout << "predicted: " << ss.str() << std::endl;

	return pred;
}

cv::Mat GetSquareImage( const cv::Mat& img, int target_width = 500 )
{
	int width = img.cols,
	   height = img.rows;

	cv::Mat square = cv::Mat::zeros( target_width, target_width, img.type() );

	int max_dim = ( width >= height ) ? width : height;
	float scale = ( ( float ) target_width ) / max_dim;
	cv::Rect roi;
	if ( width >= height )
	{
		roi.width = target_width;
		roi.x = 0;
		roi.height = height * scale;
		roi.y = ( target_width - roi.height ) / 2;
	}
	else
	{
		roi.y = 0;
		roi.height = target_width;
		roi.width = width * scale;
		roi.x = ( target_width - roi.width ) / 2;
	}

	cv::resize( img, square( roi ), roi.size() );

	return square;
}

////////////////////////
////////////////////////

//const QString ImNetPath("../../../data/imagenet/");

ImReader::ImReader()
{
	m_batch = 10;
	m_aug = true;
	m_thread = 0;
	m_done = false;
}

ImReader::ImReader(const QString& pathToImages)
{
	m_batch = 10;
	m_aug = true;
	m_thread = 0;
	m_done = false;

	m_image_path = pathToImages;
	init();
}

ImReader::~ImReader()
{
	m_done = true;
	if(m_thread){
		delete m_thread;
	}
}

void ImReader::init()
{
	QDir dir(m_image_path);

	if(dir.count() == 0){
		qDebug() << "ERROR: dir is empty";
		return;
	}

	m_all_count = 0;

	int numb = 0;
	for(uint i = 0; i < dir.count(); ++i){
		QFileInfo fi(dir.path() + "/" + dir[i]);
		if(!fi.isDir() || dir[i] == "." || dir[i] == "..")
			continue;

		QDir inDir(dir.path() + "/" + dir[i]);

		std::vector< std::string > files;
		for(uint j = 0; j < inDir.count(); ++j){
			QString sfile = inDir[j];
			if(sfile == "." || sfile == "..")
				continue;
			files.push_back(QString(dir[i] + "/" + sfile).toStdString());
		}
		m_all_count += files.size();
		qDebug() << numb++ << ": FILES[" << dir[i] << ", " << imnet::getNumberOfList(dir[i].toStdString()) << "]=" << files.size();
		m_files.push_back(files);
		m_dirs.push_back(dir[i].toStdString());
	}
	qDebug() << "DIRS" << m_dirs.size();
}

void ImReader::get_batch(std::vector<ct::Matf> &X, ct::Matf &y, int batch, bool aug, bool train)
{
	if(m_files.empty())
		return;

	X.resize(batch);
	y = ct::Matf::zeros(batch, 1);

//	std::vector< int > shuffle;
//	shuffle.resize(batch);
//	cv::randu(shuffle, 0, m_files.size());

//	std::stringstream ss, ss2;

	std::binomial_distribution<int> bn(1, 0.5);
	std::normal_distribution<float> nl(1, 0.1);

	std::uniform_int_distribution<int> ui(0, m_files.size() - 1);

	uint off = 0;

	if(train){
		if(m_saved.size() && aug){
			int cnt = std::min(batch/3, FOR_REPEAT_BATCH);
			if(!cnt) cnt = std::max(1, batch/2);
			for(int off = 0; off < cnt && !m_saved.empty(); ++off){
				X[off] = m_saved.front().X;
				y.ptr(off)[0] = m_saved.front().id;
				m_saved.pop_front();
			}
		}
	}

//#pragma omp parallel for
	for(uint i = off; i < batch; ++i){
//		int train_edge = TRAIN_EDGE;

//		if(len < TRAIN_EDGE){
//			if(len < TRAIN_EDGE2)
//				train_edge = len * 0.8;
//			else
//				train_edge = TRAIN_EDGE2;
//		}

//		std::uniform_int_distribution<int> un(0, train_edge);

//		if(!train){
//			un = std::uniform_int_distribution<int>(train_edge + 1, len - 1);
//		}

		if(train || m_val_files.empty()){
			int id1 = ui(_rnd);//shuffle[i];

			int len = m_files[id1].size();

			std::uniform_int_distribution<int> un(0, len - 1);

			int id2 = un(_rnd);

			Aug _aug;
			if(aug)
				_aug.gen(m_gt);


			ct::Matf Xi = get_image(m_image_path.toStdString() + "/" + m_files[id1][id2], _aug);
			X[i] = Xi;
			std::string n = m_dirs[id1];
			int idy = imnet::getNumberOfList(n);
			y.ptr()[i] = idy;
		}else{
//			std::cout << "val" << std::endl << std::flush;
			std::uniform_int_distribution<int> un(0, m_val_files.size() - 1);

			int id1 = un(_rnd);
			Aug _aug;
			ct::Matf Xi = get_image(m_val_files[id1], _aug);

//			cv::Mat m;
//			getMat(Xi, &m, ct::Size(IM_WIDTH, IM_HEIGHT));
//			cv::imwrite("tmp" + std::to_string(i) + ".jpg", m);

			X[i] = Xi;
			y.ptr()[i] = m_val_gt[id1];
		}
	}
//	std::cout << std::endl;
//	std::cout << "classes: " << ss.str() << std::endl;
//	std::cout << "indexes: " << ss2.str() << std::endl;
}

void offsetImage(cv::Mat &image, cv::Scalar bordercolour, int xoffset, int yoffset, float angle)
{
	using namespace cv;
	float mdata[] = {
		cos(angle), sin(angle), xoffset,
		-sin(angle), cos(angle), yoffset
	};

	Mat M(2, 3, CV_32F, mdata);
	warpAffine(image, image, M, image.size());
}

bool is_file_exists(const std::string& fn)
{
	if (FILE *file = fopen(fn.c_str(), "r")) {
		fclose(file);
		return true;
	} else {
		return false;
	}
}

ct::Matf ImReader::get_image(const std::string &name, const Aug &aug)
{
	ct::Matf res;

	if(!is_file_exists(name))
		return res;

	cv::Mat m = cv::imread(name);
	if(m.empty())
		return res;
	cv::resize(m, m, cv::Size(IM_WIDTH, IM_HEIGHT));
//	m = GetSquareImage(m, ImReader::IM_WIDTH);
	int W = IM_WIDTH;

	if(aug.zoomx != 1.f || aug.zoomy != 1.f){
		int _Wx = (float)W * aug.zoomx;
		int _Wy = (float)W * aug.zoomy;
		cv::resize(m, m, cv::Size(_Wx, _Wy));
		if(aug.zoomx < 1 && aug.zoomy < 1){
			cv::Mat r = cv::Mat::zeros(cv::Size(W, W), CV_8UC3);
			m.copyTo(r(cv::Rect(W/2 - m.cols/2, W/2 - m.rows/2, m.cols, m.rows)));
			r.copyTo(m);
		}else if(aug.zoomx < 1 && aug.zoomy > 1){
			cv::Mat r = cv::Mat::zeros(cv::Size(W, W), CV_8UC3);
			m(cv::Rect(0, _Wy/2 - W/2, _Wx, W )).copyTo(m);
			m.copyTo(r(cv::Rect(W/2 - m.cols/2, W/2 - m.rows/2, m.cols, m.rows)));
			r.copyTo(m);
		}else if(aug.zoomx > 1 && aug.zoomy < 1){
			cv::Mat r = cv::Mat::zeros(cv::Size(W, W), CV_8UC3);
			m(cv::Rect(_Wx/2 - W/2, 0 , W, _Wy)).copyTo(m);
			m.copyTo(r(cv::Rect(m.cols/2 - W/2, W/2 - m.rows/2, W, m.rows)));
			r.copyTo(m);
		}else{
			m(cv::Rect(_Wx/2 - W/2, _Wy/2 - W/2, W, W)).copyTo(m);
		}
	}

	if(aug.vflip || aug.hflip){
		if(aug.hflip && !aug.vflip){
			cv::flip(m, m, 1);
//			std::cout << "1\n";
		}else
		if(aug.vflip && !aug.hflip){
			cv::flip(m, m, 0);
//			std::cout << "2\n";
		}else{
			cv::flip(m, m, -1);
//			std::cout << "3\n";
		}
	}

	if(aug.augmentation && (aug.xoff != 0 || aug.yoff != 0)){
		offsetImage(m, cv::Scalar(0), aug.xoff, aug.yoff, aug.angle);
	}
	if(aug.inv){
		cv::bitwise_not(m, m);
	}else{
		if(aug.gray){
			cv::cvtColor(m, m, CV_RGB2GRAY);
			cv::cvtColor(m, m, CV_GRAY2RGB);
		}
	}
//	cv::imwrite("ss.bmp", m);

#if 1
	if(!aug.augmentation){
		m.convertTo(m, CV_32F, 1./255., 0);
	}else{
		m.convertTo(m, CV_32F, 1./255., aug.contrast);
	}
#else
	m.convertTo(m, CV_32F, 1./255., 0);
#endif

	res.setSize(m.channels(), m.cols * m.rows);

#pragma omp parallel for
	for(int y = 0; y < m.rows; ++y){
		float *v = m.ptr<float>(y);
		float* dX1 = res.ptr(0);
		float* dX2 = res.ptr(1);
		float* dX3 = res.ptr(2);
		for(int x = 0; x < m.cols; ++x){
			int off = y * m.cols + x;
			dX1[off] = aug.kr * v[x * m.channels() + 0];
			dX2[off] = aug.kg * v[x * m.channels() + 1];
			dX3[off] = aug.kb * v[x * m.channels() + 2];
		}
	}

	res.clipRange(0, 1);

#ifdef _DEBUG
	cv::Mat out;
	getMat(res, &out, ct::Size(IM_WIDTH, IM_HEIGHT));
	cv::imwrite("tmp.jpg", out);
#endif
	return res;
}

void ImReader::getMat(const ct::Matf &in, cv::Mat *out, const ct::Size sz)
{
	if(in.empty() || !out)
		return;

	int channels = in.total() / (sz.area());
	if(channels != 3)
		return;

	*out = cv::Mat(sz.height, sz.width, CV_32FC3);

	float* dX1 = in.ptr() + 0 * out->rows * out->cols;
	float* dX2 = in.ptr() + 1 * out->rows * out->cols;
	float* dX3 = in.ptr() + 2 * out->rows * out->cols;

	for(int y = 0; y < out->rows; ++y){
		float *v = out->ptr<float>(y);
		for(int x = 0; x < out->cols; ++x){
			int off = y * out->cols + x;
			v[x * out->channels() + 0] = dX1[off];
			v[x * out->channels() + 1] = dX2[off];
			v[x * out->channels() + 2] = dX3[off];
		}
	}
	out->convertTo(*out, CV_8UC3, 255.);
}

void ImReader::setImagePath(const QString &path)
{
	m_image_path = path;
}

void ImReader::setValidation(const std::string &folder, const std::string groundtruth_file)
{
	QDir dir;
	if(!dir.exists(QString::fromStdString(folder))
			|| !QFile::exists(QString::fromStdString(groundtruth_file))){
		printf("validation folder or validation groundtruth not set\n");
		return;
	}

	m_val_gt_file = groundtruth_file;

	QFile file(QString::fromStdString(m_val_gt_file));
	if(!file.open(QIODevice::ReadOnly)){
		printf("groundtruth file not open\n");
		return;
	}

	QTextStream tstream(&file);

	std::map< QString, int > values;

	while(!tstream.atEnd()){
		QString sid;
		sid = tstream.readLine();
		QStringList sl = sid.split(' ');
		values[sl[0]] = sl[1].toInt();
//		m_val_gt.push_back(sl[1].toInt());
	}
	file.close();

	m_val_gt.clear();
	m_val_files.clear();

//	int index = 0;
	dir.setPath(QString::fromStdString(folder));
	printf("folder %s %d\n", folder.c_str(), dir.count());
	for(int i = 0; i < dir.count(); ++i){
		QFileInfo fi(dir.path() + "/" + dir[i]);
		if(fi.isDir() || dir[i] == "." || dir[i] == "..")
			continue;
//		printf("VAL FILE %d: %s\n", index++, dir[i].toStdString().c_str());
		m_val_files.push_back(QString(dir.path() + "/" + dir[i]).toStdString());
		m_val_gt.push_back(values[dir[i]]);
	}

	std::cout << "validation loaded\n" << std::flush;

	if(m_val_files.size() != m_val_gt.size()){
		m_val_files.clear();
		m_val_gt.clear();
		printf("count of files and groundtruth  not equal\n");
	}

	if(!m_dirs.empty()){
		std::vector< std::string > new_files;
		std::vector< int > new_id;
		std::map< int, bool > cat_ids;
		std::map<int, int> cat_files;

		for(std::string n: m_dirs){
			//std::string n = m_dirs[id1];
			int idy = imnet::getNumberOfList(n);
			cat_ids[idy] = true;
		}
		for(int i = 0; i < m_val_files.size(); ++i){
			if(contain(cat_ids, m_val_gt[i])){
				new_files.push_back(m_val_files[i]);
				new_id.push_back(m_val_gt[i]);
				if(contain(cat_files, m_val_gt[i])){
					cat_files[m_val_gt[i]]++;
				}else{
					cat_files[m_val_gt[i]] = 0;
				}
			}
		}
		m_val_files = new_files;
		m_val_gt = new_id;

//		for(auto& kv: cat_files){
//			std::cout << kv.first << ": " << kv.second << std::endl;
//		}

		printf("real validation files %d\n", m_val_gt.size());
	}
}

Batch &ImReader::front()
{
	m_mutex.lock();
	Batch& bt = m_batches.front();
	m_mutex.unlock();
	return bt;
}

void ImReader::pop_front()
{
	if(is_batch_exist()){
		m_mutex.lock();
		m_batches.pop_front();
		m_mutex.unlock();
	}
}

bool ImReader::is_batch_exist() const
{
	return !m_batches.empty();
}

void ImReader::set_params_batch(int batch, bool aug)
{
	m_batch = batch;
	m_aug = aug;
}

int ImReader::batches() const
{
	return m_batches.size();
}

void ImReader::start()
{
	if(m_thread)
		return;

	m_thread = new std::thread(&ImReader::run, this);
}

void ImReader::run()
{
#define MAX_BATCHES		4

	while(!m_done){
		std::vector<ct::Matf> X;
		ct::Matf y;

		if(m_batches.size() < MAX_BATCHES){
			get_batch(X, y, m_batch, m_aug, true);
			m_mutex.lock();
			m_batches.push_back(Batch(X, y));
			m_mutex.unlock();
		}else{
			std::this_thread::sleep_for(std::chrono::milliseconds(100));
		}
	}
}

void ImReader::setSeed(int seed)
{
#ifndef CV_MAJOR_VERSION == 3 && CV_MINOR_VERSION == 1
	cv::setRNGSeed(seed);
#else
	cv::theRNG().state = seed;
#endif
	_rnd.seed(seed);

	m_gt.seed(seed);
}

void ImReader::push_to_saved(const ct::Matf &X, float id)
{
	if(m_saved.size() > MAX_SAVED){
		return;
	}
	m_saved.push_back(Saved(X, id));
}

///////////////////////////////////

///////////////////////////

inline float a2r(float angle){
	return angle * CV_PI / 180.;
}

Aug::Aug()
{
	augmentation = false;
	vflip = hflip = false;
	xoff = yoff = contrast = 0;
	kr = kb = kg = 1.;
	zoomx = 1;
	zoomy = 1;
	angle = 0;
	inv = false;
	gray = false;
}

void Aug::gen(std::mt19937 &gn)
{
	std::uniform_real_distribution<float> distr(-1., 1.);

	augmentation = true;
//	xoff = (float)ImReader::IM_WIDTH * 0.01 * distr(gn);
//	yoff = (float)ImReader::IM_HEIGHT * 0.01 * distr(gn);
	contrast = 0.03 * distr(gn);
	kr = 0.98 + 0.05 * distr(gn);
	kg = 0.98 + 0.05 * distr(gn);
	kb = 0.98 + 0.05 * distr(gn);
	zoomx = 0.95 + 0.1 * distr(gn);
	zoomy = 0.95 + 0.1 * distr(gn);
	//angle = a2r(5. * distr(gn));
	std::binomial_distribution<int> bd(1, 0.5);
	//vflip = bd(gn);
	hflip = bd(gn);
//	inv = bd(gn);
//	gray = bd(gn);
}

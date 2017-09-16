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

ImReader::ImReader(int seed)
{
	m_batch = 10;
	m_flip = true;
	m_aug = true;
	m_thread = 0;
	m_done = false;

	cv::setRNGSeed(seed);
	_rnd.seed(seed);
}

ImReader::ImReader(const QString& pathToImages, int seed)
{
	m_batch = 10;
	m_flip = true;
	m_aug = true;
	m_thread = 0;
	m_done = false;

	cv::setRNGSeed(seed);
	_rnd.seed(seed);
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
			files.push_back(QString(dir[i] + "/" + inDir[j]).toStdString());
		}
		m_all_count += files.size();
		qDebug() << numb++ << ": FILES[" << dir[i] << ", " << imnet::getNumberOfList(dir[i].toStdString()) << "]=" << files.size();
		m_files.push_back(files);
		m_dirs.push_back(dir[i].toStdString());
	}
	qDebug() << "DIRS" << m_dirs.size();
}

void ImReader::get_batch(std::vector<ct::Matf> &X, ct::Matf &y, int batch, bool flip, bool aug)
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
	std::uniform_int_distribution<int> ui(0, m_files.size() - 1);
	std::normal_distribution<float> nl(1, 0.1);

	std::vector < bool > bflip;
	bflip.resize(batch);

	if(flip){
		for(uint i = 0; i < bflip.size(); ++i){
			bflip[i] = bn(_rnd);
		}
	}else{
		std::fill(bflip.begin(), bflip.end(), false);
	}

//#pragma omp parallel for
	for(uint i = 0; i < batch; ++i){
		int id1 = ui(_rnd);//shuffle[i];

		int len = m_files[id1].size();

		std::uniform_int_distribution<int> un(0, len - 1);
		int id2 = un(_rnd);

		int xoff = 0, yoff = 0;

		if(aug){
			std::normal_distribution<float> nd(0, IM_WIDTH * 0.05);
			xoff = nd(m_gt);
			yoff = nd(m_gt);
		}

		bool is_gray = false;
		ct::Vec3f lvls = ct::Vec3f(1, 1, 1);
		if(aug){
			is_gray = bn(_rnd);
			lvls[0] = nl(_rnd);
			lvls[1] = nl(_rnd);
			lvls[2] = nl(_rnd);
		}

		ct::Matf Xi = get_image(m_image_path.toStdString() + "/" + m_files[id1][id2], bflip[i], aug, is_gray, lvls, Point(xoff, yoff));
		if(!Xi.empty()){
			X[i] = Xi;
			std::string n = m_dirs[id1];
			int idy = imnet::getNumberOfList(n);
			if(idy >= 0)
				y.ptr()[i] = idy;
			else
				printf("Oops. index not found for '%s'", n.c_str());
		}else{
			X[i] = ct::Matf::zeros(1, IM_HEIGHT * IM_WIDTH * 3);
		}
	}
//	std::cout << std::endl;
//	std::cout << "classes: " << ss.str() << std::endl;
//	std::cout << "indexes: " << ss2.str() << std::endl;
}

void offsetImage(cv::Mat &image, cv::Scalar bordercolour, int xoffset, int yoffset)
{
	using namespace cv;
	float mdata[] = {
		1, 0, xoffset,
		0, 1, yoffset
	};

	Mat M(2, 3, CV_32F, mdata);
	warpAffine(image, image, M, image.size());
}


ct::Matf ImReader::get_image(const std::string &name, bool flip, bool aug, bool gray,
							 const ct::Vec3f& lvls, const Point &off)
{
	ct::Matf res;

	cv::Mat m = cv::imread(name);
	if(m.empty())
		return res;
	cv::resize(m, m, cv::Size(IM_WIDTH, IM_HEIGHT));
//	m = GetSquareImage(m, ImReader::IM_WIDTH);

	if(flip){
		cv::flip(m, m, 1);
	}

//	if(aug && off.x != 0 && off.y != 0){
//		offsetImage(m, cv::Scalar(0), off.x, off.y);
//	}
	if(gray){
		cv::cvtColor(m, m, cv::COLOR_BGR2GRAY);
		cv::cvtColor(m, m, cv::COLOR_GRAY2BGR);
	}

//	cv::imwrite("ss.bmp", m);

#if 0
	if(!aug){
		m.convertTo(m, CV_32F, 1./255., 0);
	}else{
		std::normal_distribution<float> nd(0, 0.1);
		float br = nd(m_gt);
		float cntr = 0.3 * nd(m_gt);
		m.convertTo(m, CV_32F, (0.95 + br)/255., 0);
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
			dX1[off] = /*lvls[0] * */v[x * m.channels() + 0];
			dX2[off] = /*lvls[1] * */v[x * m.channels() + 1];
			dX3[off] = /*lvls[1] * */v[x * m.channels() + 2];
		}
	}

	res.clipRange(0, 1);

	cv::Mat out;
	getMat(res, &out, ct::Size(IM_WIDTH, IM_HEIGHT));
	cv::imwrite("tmp.jpg", out);

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

void ImReader::set_params_batch(int batch, bool flip, bool aug)
{
	m_batch = batch;
	m_flip = flip;
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
			get_batch(X, y, m_batch, m_flip, m_aug);
			m_mutex.lock();
			m_batches.push_back(Batch(X, y));
			m_mutex.unlock();
		}else{
			std::this_thread::sleep_for(std::chrono::milliseconds(100));
		}
	}
}

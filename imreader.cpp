#include "imreader.h"

#include <QDir>
#include <QDebug>
#include <QFileInfo>

#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

#include <random>
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
	cv::setRNGSeed(seed);
}

ImReader::ImReader(const QString& pathToImages, int seed)
{
	cv::setRNGSeed(seed);
	m_image_path = pathToImages;
	init();
}

void ImReader::init()
{
	QDir dir(m_image_path);

	if(dir.count() == 0){
		qDebug() << "ERROR: dir is empty";
		return;
	}

	m_all_count = 0;

	for(int i = 0; i < dir.count(); ++i){
		QFileInfo fi(dir.path() + "/" + dir[i]);
		if(!fi.isDir() || dir[i] == "." || dir[i] == "..")
			continue;

		QDir inDir(dir.path() + "/" + dir[i]);

		std::vector< std::string > files;
		for(int j = 0; j < inDir.count(); ++j){
			files.push_back(QString(dir[i] + "/" + inDir[j]).toStdString());
		}
		m_all_count += files.size();
		qDebug() << "FILES[" << dir[i] << "]=" << files.size();
		m_files.push_back(files);
		m_dirs.push_back(dir[i].toStdString());
	}
	qDebug() << "DIRS" << m_dirs.size();
}

static std::mt19937 _rnd;

void ImReader::get_batch(std::vector<ct::Matf> &X, ct::Matf &y, int batch, bool flip)
{
	if(m_files.empty())
		return;

	X.resize(batch);
	y = ct::Matf::zeros(batch, 1);

	std::vector< int > shuffle;
	shuffle.resize(batch);
	cv::randu(shuffle, 0, m_all_count);

	std::stringstream ss, ss2;

	std::binomial_distribution<int> bn(1, 0.5);

	for(int i = 0; i < shuffle.size(); ++i){
		int id = shuffle[i];

		int id1 = 0;
		int id2 = 0;

		int cnt = 0;
		for(int j = 0; j < m_files.size(); ++j){
			if(cnt + m_files[j].size() > id){
				id1 = j;
				id2 = id - cnt;
				break;
			}
			cnt += m_files[j].size();
		}

		ss << id1 << ", ";
		ss2 << id2 << ", ";

		bool fl = false;
		if(flip){
			fl = bn(_rnd);
		}

		ct::Matf Xi = get_image(m_image_path.toStdString() + "/" + m_files[id1][id2], fl);
		if(!Xi.empty()){
			X[i] = Xi;
			y.ptr()[i] = id1;
		}else{
			X[i] = ct::Matf::zeros(1, IM_HEIGHT * IM_WIDTH * 3);
		}
	}
//	std::cout << "classes: " << ss.str() << std::endl;
//	std::cout << "indexes: " << ss2.str() << std::endl;
}

ct::Matf ImReader::get_image(const std::string &name, bool flip)
{
	ct::Matf res;

	cv::Mat m = cv::imread(name);
	if(m.empty())
		return res;
	//cv::resize(m, m, cv::Size(IM_WIDTH, IM_HEIGHT));
	m = GetSquareImage(m, ImReader::IM_WIDTH);

	if(flip){
		cv::flip(m, m, 1);
	}

//	cv::imwrite("ss.bmp", m);

	m.convertTo(m, CV_32F, 1./255., 0);

	res.setSize(1, m.cols * m.rows * m.channels());

	int idx = 0;
	float* dX1 = res.ptr() + 0 * m.rows * m.cols;
	float* dX2 = res.ptr() + 1 * m.rows * m.cols;
	float* dX3 = res.ptr() + 2 * m.rows * m.cols;

	for(int y = 0; y < m.rows; ++y){
		float *v = m.ptr<float>(y);
		for(int x = 0; x < m.cols; ++x, ++idx){
			dX1[idx] = v[x * m.channels() + 0];
			dX2[idx] = v[x * m.channels() + 1];
			dX3[idx] = v[x * m.channels() + 2];
		}
	}
	return res;
}

void ImReader::setImagePath(const QString &path)
{
	m_image_path = path;
}

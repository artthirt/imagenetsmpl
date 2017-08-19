#ifndef IMREADER_H
#define IMREADER_H

#include "custom_types.h"
#include <vector>
#include <QString>

#include <thread>
#include <list>

struct Point{
	Point(){
		x = 0; y = 0;
	}
	Point(int a1, int a2){
		x = a1; y = a2;
	}

	int x, y;
};

namespace cv{
	class Mat;
}

struct Batch{
	std::vector< ct::Matf > X;
	ct::Matf y;

	Batch(){}
	Batch(std::vector< ct::Matf >& X, ct::Matf& y){
		this->X = X;
		this->y = y;
	}
};

/**
 * @brief check
 * @param i1
 * @param i2
 * @return
 */
double check(const ct::Matf& classes, const ct::Matf& predicted);

class ImReader
{
public:
	enum {
		IM_WIDTH=224, IM_HEIGHT=224
	};

	ImReader(int seed = 11);
	ImReader(const QString &pathToImages, int seed = 11);
	~ImReader();

	void init();

	void get_batch(std::vector< ct::Matf >& X, ct::Matf& y, int batch, bool flip = false, bool aug = false);

	ct::Matf get_image(const std::string& name, bool flip = false, bool aug = false, const Point& off = Point());

	void getMat(const ct::Matf &in, cv::Mat *out, const ct::Size sz);

	void setImagePath(const QString& path);

	Batch &front();
	void pop_front();
	bool is_batch_exist() const;
	void set_params_batch(int batch, bool flip, bool aug);

	void start();
	void run();

private:
	std::mt19937 m_gt;

	std::list<Batch> m_batches;
	std::thread *m_thread;
	int m_batch;
	bool m_flip;
	bool m_aug;
	bool m_done;

	std::vector< std::string > m_dirs;
	std::vector< std::vector< std::string > > m_files;
	int m_all_count;
	QString m_image_path;
};

#endif // IMREADER_H

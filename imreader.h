#ifndef IMREADER_H
#define IMREADER_H

#include "custom_types.h"
#include <vector>
#include <QString>

#include <thread>
#include <list>
#include <mutex>

#define MAX_SAVED			50
#define FOR_REPEAT_BATCH	20

struct Aug{
	Aug();
	bool augmentation;
	bool vflip;
	bool hflip;
	int xoff;
	int yoff;
	float kr;
	float kg;
	float kb;
	float contrast;
	float zoomx;
	float zoomy;
	bool inv;
	bool gray;

	void gen(std::mt19937& gn);
};

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

struct Saved{
	Saved(): id(0){}
	Saved(const ct::Matf& _X, float _id){
		_X.copyTo(X);
		id = _id;
	}
	ct::Matf X;
	float id;
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

	void get_batch(std::vector< ct::Matf >& X, ct::Matf& y, int batch, bool aug = false, bool train = true);

	ct::Matf get_image(const std::string& name, const Aug &aug = Aug());

	void getMat(const ct::Matf &in, cv::Mat *out, const ct::Size sz);

	void setImagePath(const QString& path);

	Batch &front();
	void pop_front();
	bool is_batch_exist() const;
	void set_params_batch(int batch, bool aug);
	int batches() const;

	void start();
	void run();

	void push_to_saved(const ct::Matf& X, float id);

private:
	std::mt19937 m_gt;

	std::list<Batch> m_batches;
	std::thread *m_thread;
	int m_batch;
	bool m_aug;
	bool m_done;
	std::mutex m_mutex;

	std::vector< std::string > m_dirs;
	std::vector< std::vector< std::string > > m_files;
	int m_all_count;
	QString m_image_path;

	std::list< Saved > m_saved;
};

#endif // IMREADER_H

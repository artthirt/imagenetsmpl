#ifndef IMREADER_H
#define IMREADER_H

#include "custom_types.h"
#include <vector>
#include <QString>

#include <thread>
#include <list>
#include <mutex>
#include <map>

#define MAX_SAVED			100
#define FOR_REPEAT_BATCH	100

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
	float angle;
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

void clear_predicted();
void save_predicted();

class ImReader
{
public:
	enum {
		IM_WIDTH=224, IM_HEIGHT=224, TRAIN_EDGE = 1100, TRAIN_EDGE2 = 900
	};

	ImReader();
	ImReader(const QString &pathToImages);
	~ImReader();

	void init();

	void get_batch(std::vector< ct::Matf >& X, ct::Matf& y, int batch, bool aug = false, bool train = true);

	ct::Matf get_image(const std::string& name, const Aug &aug = Aug());

	void getMat(const ct::Matf &in, cv::Mat *out, const ct::Size sz);

	void setImagePath(const QString& path);

	void setValidation(const std::string &folder, const std::string groundtruth_file);

	Batch &front();
	void pop_front();
	bool is_batch_exist() const;
	void set_params_batch(int batch, bool aug);
	int batches() const;

	void start();
	void run();

	void setSeed(int seed);

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

	std::vector< std::string > m_val_files;
	std::string m_val_gt_file;
	std::vector< int > m_val_gt;

	std::list< Saved > m_saved;
};

template< typename T1, typename T2>
bool contain(const std::map<T1, T2>& mp, const T1& key)
{
	return mp.find(key) != mp.end();
}

#endif // IMREADER_H

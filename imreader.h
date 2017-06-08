#ifndef IMREADER_H
#define IMREADER_H

#include "custom_types.h"
#include <vector>
#include <QString>

class ImReader
{
public:
	enum {
		IM_WIDTH=224, IM_HEIGHT=224
	};

	ImReader();
	ImReader(const QString &pathToImages);

	void init();

	void get_batch(std::vector< ct::Matf >& X, ct::Matf& y, int batch);

	ct::Matf get_image(const std::string& name);

	void setImagePath(const QString& path);

private:
	std::vector< std::string > m_dirs;
	std::vector< std::vector< std::string > > m_files;
	int m_all_count;
	QString m_image_path;
};

#endif // IMREADER_H

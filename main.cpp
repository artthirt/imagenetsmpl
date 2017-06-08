#include <QCoreApplication>

#include "imnetsmpl.h"
#include "imreader.h"
#include <map>

std::map<std::string, std::string> parseArgs(int argc, char *argv[])
{
	std::map<std::string, std::string> res;
	for(int i = 0; i < argc; ++i){
		std::string str = argv[i];
		if(str == "-f" && i < argc){
			res["imnet"] = argv[i + 1];
		}
	}
	return res;
}

int main(int argc, char *argv[])
{
	std::map<std::string, std::string> res = parseArgs(argc, argv);

	if(res.empty()){
		printf("Usage: prog -f Path/To/ImageNet/Folder");
		return 1;
	}

	QCoreApplication a(argc, argv);

	ImReader ir(QString(res["imnet"].c_str()));
	//ImReader ir(QString("d:/Down/smpl/data/imagenet"));

	ImNetSmpl imnetSmpl;

//	std::vector< ct::Matf > X;
//	ct::Matf y;

	imnetSmpl.setReader(&ir);

	imnetSmpl.doPass(1000, 10);

//	ir.get_batch(X, y, 10);

	return a.exec();
}

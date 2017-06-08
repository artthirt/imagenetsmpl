#include <QCoreApplication>
#include <QDebug>

#include "imnetsmpl.h"
#include "imreader.h"
#include <map>

bool contain(const std::map<std::string, std::string>& mp, const std::string& key)
{
	return mp.find(key) != mp.end();
}

std::map<std::string, std::string> parseArgs(int argc, char *argv[])
{
	std::map<std::string, std::string> res;
	for(int i = 0; i < argc; ++i){
		std::string str = argv[i];
		if(str == "-f" && i < argc){
			res["imnet"] = argv[i + 1];
		}
		if(str == "-load" && i < argc){
			res["load"] = argv[i + 1];
		}
		if(str == "-image" && i < argc){
			res["image"] = argv[i + 1];
		}
	}
	return res;
}

int main(int argc, char *argv[])
{
	std::map<std::string, std::string> res = parseArgs(argc, argv);

	if(res.empty()){
		printf("Usage: app [OPTIONS]\n"
			   "-f Path/To/ImageNet/Folder \n"
			   "-load path/to/model \n"
			   "-image path/to/image \n");
		return 1;
	}

	QCoreApplication a(argc, argv);

	ImReader ir(QString(res["imnet"].c_str()));
	//ImReader ir(QString("d:/Down/smpl/data/imagenet"));

	ImNetSmpl imnetSmpl;

	imnetSmpl.setReader(&ir);

	if(contain(res, "load")){
		imnetSmpl.load_net(res["load"].c_str());
	}

	if(contain(res, "image")){
		imnetSmpl.predict(res["image"].c_str(), true);
	}

	printf("1\n");

	if(!contain(res, "imnet") || res["imnet"].empty()){
		printf("path to imagenet not specified. exit\n");
		return 1;
	}

	printf("2\n");
//	std::vector< ct::Matf > X;
//	ct::Matf y;

	imnetSmpl.doPass(1000, 10);

//	ir.get_batch(X, y, 10);

	return a.exec();
}

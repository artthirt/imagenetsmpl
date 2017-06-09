#include <QCoreApplication>
#include <QDebug>

#include "imnetsmpl.h"
#include "imnetsmplgpu.h"
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
		if(str == "-gpu"){
			res["gpu"] = "1";
		}
		if(str == "-pass" && i < argc){
			res["pass"] = argv[i + 1];
		}
		if(str == "-batch" && i < argc){
			res["batch"] = argv[i + 1];
		}
		if(str == "-images" && i < argc){
			res["images"] = argv[i + 1];
		}
	}
	return res;
}

int main(int argc, char *argv[])
{
	std::map<std::string, std::string> res = parseArgs(argc, argv);

	if(res.empty()){
		printf("Usage: app [OPTIONS]\n"
			   "-f Path/To/ImageNet/Folder     - path to directory with ImageNet data\n"
			   "-load path/to/model            - load model for network\n"
			   "-image path/to/image           - predict one image\n"
			   "-gpu                           - use gpu\n"
			   "-pass [numbers pass]            - size of pass\n"
			   "-batch [number batch for one]   - size of batch for pass\n"
			   "-images path/to/dir/with/images - check all images in directory\n");
		return 1;
	}

	QCoreApplication a(argc, argv);

	ImReader ir(QString(res["imnet"].c_str()));
	//ImReader ir(QString("d:/Down/smpl/data/imagenet"));


	int batch = 10;
	int pass = 1000;

	if(contain(res, "pass")){
		pass = std::stoi(res["pass"]);
	}

	if(contain(res, "batch")){
		batch = std::stoi(res["batch"]);
	}

	printf("Parameters startup: pass=%d, batch=%d\n", pass, batch);

	if(contain(res, "gpu")){
		ImNetSmplGpu imnetSmpl;
		imnetSmpl.setReader(&ir);

		if(contain(res, "load")){
			imnetSmpl.load_net(res["load"].c_str());
		}

		if(contain(res, "image")){
			imnetSmpl.predict(res["image"].c_str(), true);
		}

		if(!contain(res, "imnet") || res["imnet"].empty()){
			printf("path to imagenet not specified. exit\n");
			return 1;
		}

		imnetSmpl.doPass(pass, batch);
	}else{
		ImNetSmpl imnetSmpl;

		imnetSmpl.setReader(&ir);

		if(contain(res, "load")){
			imnetSmpl.load_net(res["load"].c_str());
		}

		if(contain(res, "image")){
			imnetSmpl.predict(res["image"].c_str(), true);
		}

		if(contain(res, "images")){
			imnetSmpl.predicts(res["images"].c_str());
		}

		if(!contain(res, "imnet") || res["imnet"].empty()){
			printf("path to imagenet not specified. exit\n");
			return 1;
		}

//	std::vector< ct::Matf > X;
//	ct::Matf y;

		imnetSmpl.doPass(pass, batch);
	}

//	ir.get_batch(X, y, 10);

	return a.exec();
}

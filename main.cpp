#include <QCoreApplication>

#include "imnetsmpl.h"
#include "imnetsmplgpu.h"
#include "imreader.h"
#include <map>

#include "nn.h"

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
		if(str == "-load2" && i < argc){
			res["load2"] = argv[i + 1];
		}
		if(str == "-save" && i < argc){
			res["save"] = argv[i + 1];
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
		if(str == "-lr" && i < argc){
			res["lr"] = argv[i + 1];
		}
		if(str == "-images" && i < argc){
			res["images"] = argv[i + 1];
		}
		if(str == "-seed" && i < argc){
			res["seed"] = argv[i + 1];
		}
		if(str == "-backconv" && i < argc){
			res["backconv"] = argv[i + 1];
		}
	}
	return res;
}

void test()
{
	ct::Matf res, a1, a2, a3, b1, b2, b3;

	a1.setSize(10, 13);
	a1.fill(1);
	a2.setSize(10, 15);
	a2.fill(3);
	a3 = ct::Matf::eye(10, 14);

	std::vector< ct::Matf > mats, mats1;
	mats.push_back(a1);
	mats.push_back(a2);
	mats.push_back(a3);
	ct::hconcat(mats, res);

	std::cout << res.print() << std::endl;

	std::vector< int > cols;
	cols.push_back(13);
	cols.push_back(15);
	cols.push_back(14);

	mats1.push_back(b1);
	mats1.push_back(b2);
	mats1.push_back(b3);

	ct::hsplit(res, cols, mats1);

	std::cout << "b1\n" << b1.print() << std::endl;
	std::cout << "b2\n" << b2.print() << std::endl;
	std::cout << "b3\n" << b3.print() << std::endl;
}

void test2()
{
	ct::Matf A(5, 10), B = ct::Matf::ones(3, 3), C, D;

	for(int i = 0, k = 1; i < A.rows; ++i){
		float *dA = A.ptr(i);
		for(int j = 0; j < A.cols; ++j, ++k){
			dA[j] = k;
		}
	}
//	A = A.t();

	std::cout << A.print() << std::endl;

	ct::Size szOut;

	conv2::conv2(A, ct::Size(10, 5), 1, 1, B, ct::Size(3, 3), C, szOut, conv2::SAME);

	int rows = C.rows;
	int cols = C.cols;

	C.rows = szOut.height;
	C.cols = szOut.width;

	std::cout << C.print() << std::endl;

	C.rows = rows;
	C.cols = cols;

	conv2::conv2_transpose(C, ct::Size(10, 5), 1, 1, B, ct::Size(3, 3), szOut, D, conv2::SAME);

	D.rows = szOut.height;
	D.cols = szOut.width;

	std::cout << D.print() << std::endl;

	gpumat::GpuMat g_A, g_B, g_C;
	gpumat::convert_to_gpu(A, g_A);
	gpumat::convert_to_gpu(B, g_B);

	gpumat::conv2(g_A, ct::Size(10, 5), 1, 1, g_B, ct::Size(3, 3), g_C, szOut, gpumat::SAME);

	g_C.rows = szOut.height;
	g_C.cols = szOut.width;

	std::cout << g_C.print() << std::endl;
}

int main(int argc, char *argv[])
{
	std::map<std::string, std::string> res = parseArgs(argc, argv);

//	test2();

	if(res.empty()){
		printf("Usage: app [OPTIONS]\n"
			   "-f Path/To/ImageNet/Folder		- path to directory with ImageNet data\n"
			   "-load path/to/model				- load model for network\n"
			   "-image path/to/image			- predict one image\n"
			   "-gpu							- use gpu\n"
			   "-pass [numbers pass]            - size of pass *default: 1000\n"
			   "-batch [number batch for one]   - size of batch for pass (default: 10)\n"
			   "-images path/to/dir/with/images - check all images in directory\n"
			   "-lr learing_rate				- learing rate (default: 0.001"
			   "-save path/to/model				- name for saved train model"
			   "-load2 path/to/model			- load extension model with information about sizes of layers and matrices"
			   "-seed number					- seed for set random seed for train"
			   "-backconv bool					- use or not bakward pass for convolution");
		return 1;
	}

	//QCoreApplication a(argc, argv);

	int seed = 1;
	if(contain(res, "seed")){
		seed = std::stoi(res["seed"]);
	}

	ImReader ir(QString(res["imnet"].c_str()), seed);
	//ImReader ir(QString("d:/Down/smpl/data/imagenet"));


//	ir.get_image("d:/Down/smpl/data/imnet/n01818515/n01818515_583.JPEG", true);
//	return 1;

	int batch = 10;
	int pass = 1000;
	bool backconv = true;
	double lr = 0.001;

	printf("Startup\n");

	if(contain(res, "pass")){
		pass = std::stoi(res["pass"]);
	}

	if(contain(res, "batch")){
		batch = std::stoi(res["batch"]);
	}

	if(contain(res, "lr")){
		lr = std::stod(res["lr"]);
	}

	if(contain(res, "backconv")){
		backconv = std::stoi(res["backconv"]);
	}

	printf("pass %d\n", pass);
	printf("batch %d\n", batch);
	printf("learning rate %f\n", lr);
	printf("seed %d\n", seed);
	printf("use backward convolution %d\n", backconv);

	//printf("Parameters startup: pass=%d, batch=%d, lr=%f\n", pass, batch, lr);

	if(contain(res, "gpu")){
		ImNetSmplGpu imnetSmpl;
		imnetSmpl.setReader(&ir);
		imnetSmpl.setLearningRate(lr);
		imnetSmpl.setUseBackConv(backconv);

		if(contain(res, "load")){
			printf("load simple model '%s'\n", res["load"].c_str());
			imnetSmpl.load_net(res["load"].c_str());
		}

		if(contain(res, "save")){
			printf("save model to '%s'\n", res["save"].c_str());
			imnetSmpl.setSaveModelName(res["save"].c_str());
		}

		if(contain(res, "load2")){
			printf("load model '%s'\n", res["load2"].c_str());
			imnetSmpl.load_net2(res["load2"].c_str());
		}

		if(contain(res, "image")){
			printf("predict image '%s'\n", res["image"].c_str());
			imnetSmpl.predict(res["image"].c_str(), true);
			return 0;
		}

		if(contain(res, "images")){
			printf("predict images '%s'\n", res["images"].c_str());
			imnetSmpl.predicts(res["images"].c_str());
			return 0;
		}

		if(!contain(res, "imnet") || res["imnet"].empty()){
			printf("path to imagenet not specified. exit\n");
			return 1;
		}

		imnetSmpl.doPass(pass, batch);
	}else{
		ImNetSmpl imnetSmpl;
		imnetSmpl.setReader(&ir);
		imnetSmpl.setLearningRate(lr);
		imnetSmpl.setUseBackConv(backconv);

		if(contain(res, "load")){
			printf("load simple model '%s'\n", res["load"].c_str());
			imnetSmpl.load_net(res["load"].c_str());
		}

		if(contain(res, "save")){
			printf("save model to '%s'\n", res["load2"].c_str());
			imnetSmpl.setSaveModelName(res["save"].c_str());
		}

		if(contain(res, "load2")){
			printf("load model '%s'\n", res["load2"].c_str());
			imnetSmpl.load_net2(res["load2"].c_str());
		}

		if(contain(res, "image")){
			printf("predict image '%s'\n", res["image"].c_str());
			imnetSmpl.predict(res["image"].c_str(), true);
			return 0;
		}

		if(contain(res, "images")){
			printf("predict images '%s'\n", res["images"].c_str());
			imnetSmpl.predicts(res["images"].c_str());
			return 0;
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

	return 0;//a.exec();
}

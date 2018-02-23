#include <QCoreApplication>

#include "imnetsmpl.h"
#include "imreader.h"
#include <map>

#ifdef _USE_GPU
#include "imnetsmplgpu.h"
#endif

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
        if(str == "-files" && i < argc){
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
		if(str == "-train_layer_from" && i < argc){
			res["train_layer_from"] = argv[i + 1];
		}
//		if(str == "-validation_groundtruth" && i < argc){
//			res["validation_groundtruth"] = argv[i + 1];
//		}
        if(str == "-val" && i < argc){
            res["val"] = argv[i + 1];
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

#ifdef _USE_GPU

void test2()
{
	int c_rows = 35;
	int c_cols = 10;
	int c_channels = 2;

	ct::Matf A(c_rows, c_cols * c_channels), B = ct::Matf::ones(c_channels * 3 * 3, 2), C, D;

	for(int i = 0, k = 1; i < c_rows; ++i){
		float *dA = A.ptr(i);
		for(int j = 0; j < c_cols; ++j, ++k){
			dA[c_channels * j + 0] = k;
			dA[c_channels * j + 1] = k;
		}
	}
//	A = A.t();

	std::cout << A.print() << std::endl;

	ct::Size szOut;

	conv2::conv2(A, ct::Size(c_cols, c_rows), c_channels, 1, B, ct::Size(3, 3), C, szOut, conv2::SAME, true);

	int rows = C.rows;
	int cols = C.cols;

	C.rows = szOut.height;
	C.cols = szOut.width * c_channels;

	std::cout << C.print() << std::endl;
	ct::save_mat(C, "C.txt");

	C.rows = rows;
	C.cols = cols;

	conv2::conv2_transpose(C, ct::Size(c_cols, c_rows), c_channels, 1, B, ct::Size(3, 3), szOut, D, conv2::SAME, true);

	D.rows = c_rows;
	D.cols = c_cols;

	std::cout << D.print() << std::endl;

	/////////////

	gpumat::GpuMat g_A, g_B, g_C, g_D;
	gpumat::convert_to_gpu(A, g_A);
	gpumat::convert_to_gpu(B, g_B);

	gpumat::conv2(g_A, ct::Size(c_cols, c_rows), c_channels, 1, g_B, ct::Size(3, 3), g_C, szOut, gpumat::SAME, true);

	g_C.rows = szOut.height;
	g_C.cols = szOut.width * c_channels;

	std::cout << g_C.print() << std::endl;
	gpumat::save_gmat(g_C, "g_C.txt");

	g_C.rows = rows;
	g_C.cols = cols;

	gpumat::conv2_transpose(g_C, ct::Size(c_cols, c_rows), c_channels, 1, g_B, ct::Size(3, 3), szOut, g_D, gpumat::SAME, true);

	g_D.rows = c_rows;
	g_D.cols = c_cols;

	std::cout << g_D.print() << std::endl;

}

void test3()
{
	ct::Matf W(20, 10), B(1, 10), A(13, 20);

	W.randn(0, 1);
	B.randn(0, 1);
	A.randn(0, 1);

	gpumat::GpuMat gW, gA, gB, gC, gD;
	gpumat::convert_to_gpu(A, gA);
	gpumat::convert_to_gpu(B, gB);
	gpumat::convert_to_gpu(W, gW);

	gpumat::matmul(gA, gW, gC);
	gpumat::biasPlus(gC, gB);
	gpumat::leakyReLu(gC, 0.1);
	gpumat::save_gmat(gC, "C1.txt");

	gpumat::m2mpbaf(gA, gW, gB, gpumat::LEAKYRELU, gD, 0.1);
	gpumat::save_gmat(gD, "C2.txt");
}

#endif

int main(int argc, char *argv[])
{
	std::map<std::string, std::string> res = parseArgs(argc, argv);

//	test2();
//	test3();

	if(res.empty()){
		printf("Usage: app [OPTIONS]\n"
               "-files Path/To/ImageNet/Folder  - path to directory with ImageNet data\n"
               "-load path/to/model             - load model for network\n"
               "-image path/to/image            - predict one image\n"
               "-gpu                            - use gpu\n"
               "-pass [numbers pass]            - size of pass *default: 1000\n"
               "-batch [number batch for one]   - size of batch for pass (default: 10)\n"
			   "-images path/to/dir/with/images - check all images in directory\n"
               "-lr learing_rate                - learing rate (default: 0.001"
               "-save path/to/model             - name for saved train model"
               "-load2 path/to/model            - load extension model with information about sizes of layers and matrices"
               "-seed number                    - seed for set random seed for train"
               "-backconv 0/1                   - use or not bakward pass for convolution"
               "-train_layer_from number        - train conv layers from this to end");
		return 1;
	}

	//QCoreApplication a(argc, argv);

	int seed = 1;
	if(contain(res, std::string("seed"))){
		seed = std::stoi(res["seed"]);
	}

    ImReader ir(res["imnet"]);
	//ImReader ir(QString("d:/Down/smpl/data/imagenet"));


//	ir.get_image("d:/Down/smpl/data/imnet/n01818515/n01818515_583.JPEG", true);
//	return 1;

	int batch = 10;
	int pass = 1000;
	bool backconv = true;
	double lr = 0.001;
	int train_layer_from = 0;

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
	if(contain(res, "train_layer_from")){
		train_layer_from = std::stoi(res["train_layer_from"]);
	}

    std::string val_folder/*, val_gt*/;
    if(contain(res, "val")){
        val_folder = res["val"];
	}

//	if(contain(res, "validation_groundtruth")){
//		val_gt = res["validation_groundtruth"];
//	}

    ir.setValidation(val_folder/*, val_gt*/);

	printf("pass %d\n", pass);
	printf("batch %d\n", batch);
	printf("learning rate %f\n", lr);
	printf("seed %d\n", seed);
	printf("use backward convolution %d\n", backconv);
	printf("train_layer_from %d\n", train_layer_from);

	//printf("Parameters startup: pass=%d, batch=%d, lr=%f\n", pass, batch, lr);

	ir.setSeed(seed);

	if(contain(res, "gpu")){
#ifdef _USE_GPU
		ImNetSmplGpu imnetSmpl;
		imnetSmpl.setReader(&ir);
		imnetSmpl.setLearningRate(lr);
		imnetSmpl.setUseBackConv(backconv);
		imnetSmpl.setLayerFrom(train_layer_from);

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
#endif
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

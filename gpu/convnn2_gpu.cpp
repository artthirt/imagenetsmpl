#include "convnn2_gpu.h"
#include "nn.h"

#include "qt_work_mat.h"

using namespace gpumat::conv2;

//////////////////////////

void save_vec(const std::vector< gpumat::GpuMat >& Dlt)
{
	for(size_t i = 0; i < Dlt.size(); ++i){
		std::stringstream ss;
		ss << "data/" << i << ".txt";
		gpumat::save_gmat(Dlt[i], ss.str());
	}
}

//////////////////////////

#include "convnn2.h"

template< typename T>
void check(const ct::Mat_<T> &c1, const ct::Mat_<T>& c2)
{
	ct::Mat_<T> c3 = c2 - c1;
	ct::v_elemwiseSqr(c3);
	float s = c3.sum();
	ct::save_mat(c3, "c3.txt");
	assert(s < 1e-6);
}

void check_deriv(const std::vector< gpumat::GpuMat >& Delta,
				 const ct::Size& szOut,
				 const ct::Size& szA0,
				 int channels,
				 const ct::Size& szW,
				 int stride,
				 std::vector< gpumat::GpuMat >& X)
{
	gpumat::GpuMat Dlt;

	back_derivT(Delta[0], szOut, szA0, channels, szW, stride, Dlt);

	ct::Matf c1, c2, c3, c4;

	gpumat::convert_to_mat(X[0], c1);
//	gpumat::convert_to_mat(Dlt, c2);

//	check(c1, c2);

	gpumat::convert_to_mat(Delta[0], c4);

	conv2::back_derivT(c4, szOut, szA0, channels, szW, stride, c3);

	check(c3, c1);
}


//////////////////////////

convnn_gpu::convnn_gpu()
{
	m_use_pool = false;
	pX = nullptr;
	stride = 1;
	m_use_transpose = true;
	m_pool_dropout = false;
	m_prob_dropout = 0.9;
	m_lambda = 0;
	m_optim = &m_innet_optim;
}

void convnn_gpu::setOptimizer(gpumat::Optimizer *optim)
{
	if(optim)
		m_optim = optim;
}

void convnn_gpu::setAlpha(double val)
{
	m_optim->setAlpha(val);
}

void convnn_gpu::setLambda(double val)
{
	m_lambda = val;
}

void convnn_gpu::setDropout(bool val)
{
	m_pool_dropout = val;
}

void convnn_gpu::setDropout(double val)
{
	m_prob_dropout = val;
}

std::vector<gpumat::GpuMat> &convnn_gpu::XOut()
{
	if(m_use_pool)
		return A2;
	return A1;
}

std::vector<gpumat::GpuMat> &convnn_gpu::XOut1()
{
	return A1;
}

std::vector<gpumat::GpuMat> &convnn_gpu::XOut2()
{
	return A2;
}

bool convnn_gpu::use_pool() const
{
	return m_use_pool;
}

int convnn_gpu::outputFeatures() const
{
	if(m_use_pool){
		int val = szA2.area() * K;
		return val;
	}else{
		int val= szA1.area() * K;
		return val;
	}
}

ct::Size convnn_gpu::szOut() const
{
	if(m_use_pool)
		return szA2;
	else
		return szA1;
}

void convnn_gpu::init(const ct::Size &_szA0, int _channels, int stride, int _K,
					  const ct::Size &_szW, bool use_pool, bool use_transpose)
{
	szW = _szW;
	K = _K;
	channels = _channels;
	m_use_pool = use_pool;
	m_use_transpose = use_transpose;
	szA0 = _szA0;
	this->stride = stride;

	int rows = szW.area() * channels;
	int cols = K;

	ct::get_cnv_sizes(szA0, szW, stride, szA1, szA2);

	W.resize(1);
	B.resize(1);

	gW.resize(1);
	gB.resize(1);

	float n = (float)1./szW.area();

	for(size_t i = 0; i < W.size(); ++i){
		ct::Matf Wi(rows, cols), Bi(K, 1);
		Wi.randn(0, n);
		gpumat::convert_to_gpu(Wi, W[i]);
		Bi.randn(0, n);
		gpumat::convert_to_gpu(Bi, B[i]);
	}

	gW[0].resize(W[0]);
	gB[0].resize(B[0]);

	printf("Out=[%dx%dx%d]\n", szOut().width, szOut().height, K);

	m_optim->init(W, B);
}

template< typename T >
void dropout_to_gpu(gpumat::GpuMat& Dropout, const ct::Size& sz, double prob)
{
	ct::Mat_<T> d;
	ct::dropout(sz.height, sz.width, (T)prob, d);
	gpumat::convert_to_gpu(d, Dropout);
}

void get_dropout(double prob, std::vector<gpumat::GpuMat>& X, gpumat::GpuMat& Dropout)
{
	if(X.empty() || std::abs(prob - 1.) < 1e-6)
		return;

	switch (X[0].type) {
		case gpumat::GPU_DOUBLE:
			dropout_to_gpu<double>(Dropout, X[0].sz(), prob);
			break;
		default:
		case gpumat::GPU_FLOAT:
			dropout_to_gpu<float>(Dropout, X[0].sz(), prob);
			break;
	}
	for(size_t i = 0; i < X.size(); ++i){
		gpumat::elemwiseMult(X[i], Dropout);
	}
//	qDebug("get_dropout: pool dropout generated and applied. prob=%f", prob);
}

void set_dropout(std::vector<gpumat::GpuMat>& X, const gpumat::GpuMat& Dropout)
{
	if(X.empty() || Dropout.empty())
		return;

	for(size_t i = 0; i < X.size(); ++i){
		gpumat::elemwiseMult(X[i], Dropout);
	}
//	qDebug("set_dropout: pool dropout applied");
}

void convnn_gpu::forward(const std::vector<gpumat::GpuMat> *_pX, gpumat::etypefunction func)
{
	if(!_pX)
		return;
	pX = (std::vector< gpumat::GpuMat >*)_pX;
	m_func = func;

	Xc.resize(pX->size());
	A1.resize(pX->size());

	ct::Size szOut;

	if(m_use_transpose){
		gpumat::conv2::im2colsT(*pX, szA0, channels, szW, stride, Xc, szOut);
	}else{
		gpumat::conv2::im2cols(*pX, szA0, channels, szW, stride, Xc, szOut);
	}

	for(size_t i = 0; i < Xc.size(); ++i){
		gpumat::GpuMat& Xi = Xc[i];
		gpumat::GpuMat& A1i = A1[i];
		gpumat::matmul(Xi, W[0], A1i);
		gpumat::biasPlus(A1i, B[0]);
	}

	for(size_t i = 0; i < A1.size(); ++i){
		gpumat::GpuMat& Ao = A1[i];
		switch (m_func) {
			case gpumat::RELU:
				gpumat::reLu(Ao);
				break;
			case gpumat::SIGMOID:
				gpumat::sigmoid(Ao);
				break;
			case gpumat::TANH:
				gpumat::tanh(Ao);
				break;
			default:
				break;
		}
	}

	if(m_pool_dropout){
		get_dropout(m_prob_dropout, A1, m_Dropout);
	}

	if(m_use_pool){
		Mask.resize(Xc.size());
		A2.resize(A1.size());
		ct::Size szOut;
		conv2::subsample(A1, szA1, A2, Mask, szOut);
		szK = A2[0].sz();
	}else{
		szK = A1[0].sz();
	}

#if 0
	if(channels == 3){
		qt_work_mat::q_save_mat((*pX)[26], "testPx26.txt");
		qt_work_mat::q_save_mat(Xc[26], "testXc26.txt");
		qt_work_mat::q_save_mat(A1[26], "testA126.txt");
		qt_work_mat::q_save_mat(A2[26], "testA226.txt");
		qt_work_mat::q_save_mat(W[0], "testW.txt");
		qt_work_mat::q_save_mat(B[0], "testB.txt");
		qt_work_mat::q_save_mat(Mask[26], "testMask.txt");
	}
#endif

#if 0
	{
		QString pref = QString::number(channels) + "_" + QString::number(K);
		qt_work_mat::q_save_mat((*pX)[0], "Px_" + pref + ".txt");
		qt_work_mat::q_save_mat(Xc[0], "Xc_" + pref + ".txt");
		qt_work_mat::q_save_mat(A1[0], "A1_" + pref + ".txt");
		if(!A2.empty()){
			qt_work_mat::q_save_mat(Mask[0], "M_" + pref + ".txt");
			qt_work_mat::q_save_mat(A2[0], "A2_" + pref + ".txt");
		}
		qt_work_mat::q_save_mat(W[0], "W_" + pref + ".txt");
		qt_work_mat::q_save_mat(B[0], "B_" + pref + ".txt");
	}
#endif
}

void convnn_gpu::backcnv(const std::vector<gpumat::GpuMat> &D, std::vector<gpumat::GpuMat> &DS)
{
//	DA1.resize(A1.size());
	/// A1 -> DA1
	for(size_t i = 0; i < D.size(); ++i){
		switch (m_func) {
			case ct::LINEAR:
				D[i].copyTo(DS[i]);
				break;
			case ct::RELU:
				gpumat::deriv_reLu(A1[i]/*, DA1[i]*/);
				break;
			case ct::SIGMOID:
				gpumat::deriv_sigmoid(A1[i]/*, DA1[i]*/);
				break;
			case ct::TANH:
				gpumat::deriv_tanh(A1[i]/*, DA1[i]*/);
				break;
			default:
				break;
		}
	}

	if(m_func == gpumat::LINEAR)
		return;

	/// D * DA1
	if(D.data() != DS.data()){
		for(size_t i = 0; i < D.size(); ++i){
			gpumat::elemwiseMult(D[i], A1[i], DS[i]);
		}
	}else{
		for(size_t i = 0; i < D.size(); ++i){
			gpumat::elemwiseMult(DS[i], A1[i]);
		}
	}
}

void convnn_gpu::backward(const std::vector<gpumat::GpuMat> &D, bool last_level)
{
	if(D.empty() || D.size() != A1.size()){
		throw new std::invalid_argument("vector D not complies saved parameters");
	}

//	qDebug("<<<<< backward(channels=%d, kernels=%d, Delta[%dx%d], W[%dx%d], szA0[%dx%d], szA1[%dx%d], szA2[%dx%d]) >>>>>>",
//		   channels, K, D[0].rows, D[0].cols, W[0].rows, W[0].cols,
//			szA0.width, szA0.height, szA1.width, szA1.height, szA2.width, szA2.height);

	dSub2.resize(D.size());

	if(m_use_pool){
//		dSub.resize(D.size());
//		qDebug("backward: upsample(D[%dx%d])", D[0].rows, D[0].cols);

		gpumat::conv2::upsample(D, K, Mask, szA2, szA1, dSub2);

//		qDebug("backward: derivative(D[%dx%d])", dSub2[0].rows, dSub2[0].cols);

		backcnv(dSub2, dSub2);

//		save_vec(dSub);

	}else{
//		qDebug("backward: derivative(D[%dx%d])", D[0].rows, D[0].cols);
		backcnv(D, dSub2);
	}

	if(m_pool_dropout){
		set_dropout(dSub2, m_Dropout);
	}
//	qt_work_mat::q_save_mat(dSub2[0], "testMask.txt");

#if 0
	if(channels == 3){
		qt_work_mat::q_save_mat(D[26], "testD26.txt");
//		qt_work_mat::q_save_mat(dSub[26], "testDSub26.txt");
		qt_work_mat::q_save_mat(dSub2[26], "testDSub2_26.txt");
		//save_vec(dSub2);
	}
#endif

//	qDebug("backward: Xc[%dx%d]' x D[%dx%d]", Xc[0].rows, Xc[0].cols, dSub2[0].rows, dSub2[0].cols);

	vgW.resize(D.size());
	vgB.resize(D.size());
	for(size_t i = 0; i < D.size(); ++i){
		gpumat::GpuMat& Xci		= Xc[i];
		gpumat::GpuMat& dSubi	= dSub2[i];
		gpumat::GpuMat& Wi		= vgW[i];
		gpumat::GpuMat& vgBi	= vgB[i];
		gpumat::matmulT1(Xci, dSubi, Wi);

//		gpumat::mulval(Wi, (double)1. / (Xci.total()));
//		gpumat::save_gmat(Xci, "Xgi.txt");
//		gpumat::save_gmat(dSubi, "Dgi.txt");
//		gpumat::save_gmat(Wi, "Wgi.txt");
		vgBi.swap_dims();
		sumRows(dSubi, vgBi /*, (double)1. / (Xci.total())*/);
		vgBi.swap_dims();
	}
//	gpumat::save_gmat(vgW[0], "Wg1.txt");
//	gpumat::save_gmat(vgW[1], "Wg2.txt");
//	gpumat::save_gmat(vgW[2], "Wg3.txt");

	gW[0].zeros();
	gB[0].zeros();
	for(size_t i = 0; i < D.size(); ++i){
		gpumat::add(gW[0], vgW[i]);
		gpumat::add(gB[0], vgB[i]);
	}
	gpumat::mulval(gW[0], (double)1./(D.size() * channels));
	gpumat::mulval(gB[0], (double)1./(D.size() * channels));

	if(m_lambda > 0){
		gpumat::add(gW[0], W[0], 1, (double)m_lambda / K);
	}

#if 0
	if(1/*channels == 128*/){
		save_vec(vgW);
		gpumat::save_gmat(gW[0], "Wg.txt");
	}
#endif
	if(!last_level){
		Dlt.resize(D.size());

		Dc.resize(D.size());
		for(size_t i = 0; i < D.size(); ++i){
			gpumat::GpuMat& Dci = Dc[i];
			gpumat::matmulT2(dSub2[i], W[0], Dci);
		}

//		gpumat::write_gmat("Dc5.bin", Dc[0]);
		back_derivT(Dc, szA1, szA0, channels, szW, stride, Dlt);

#if 0
		check_deriv(Dc, szA1, szA0, channels, szW, stride, Dlt);
#endif

//		gpumat::write_gmat("Dlt5.bin", Dlt[0]);
		//gpumat::save_gmat(dSub[0], "dSub.txt");
//		gpumat::save_gmat(Dlt[0], "Dltgi.txt");
		//gpumat::save_gmat(Dc[0], "Dc.txt");
	}

	m_optim->pass(gW, gB, W, B);

//	qDebug("<<<<< end backward >>>>>>");
}

void convnn_gpu::write(std::fstream &fs)
{
	gpumat::write_fs(fs, W[0]);
	gpumat::write_fs(fs, B[0]);
}

void convnn_gpu::read(std::fstream &fs)
{
	gpumat::read_fs(fs, W[0]);
	gpumat::read_fs(fs, B[0]);
}

///////////////////////////////
///////////////////////////////
///////////////////////////////

extern "C"
void cuda_im2cols(const gpumat::GpuMat &X,
				  const ct::Size &szA0,
				  int channels,
				  const ct::Size &szW,
				  int stride,
				  gpumat::GpuMat &Res,
				  ct::Size &szOut);

extern "C"
void cuda_im2cols_vec(const std::vector< gpumat::GpuMat > &X,
				  const ct::Size &szA0,
				  int channels,
				  const ct::Size &szW,
				  int stride,
				  std::vector< gpumat::GpuMat > &Res,
				  ct::Size &szOut);

extern "C"
void cuda_im2colsT(const gpumat::GpuMat &X,
				  const ct::Size &szA0,
				  int channels,
				  const ct::Size &szW,
				  int stride,
				  gpumat::GpuMat &Res,
				  ct::Size &szOut);

extern "C"
void cuda_im2colsT_vec(const std::vector< gpumat::GpuMat > &X,
				  const ct::Size &szA0,
				  int channels,
				  const ct::Size &szW,
				  int stride,
				  std::vector< gpumat::GpuMat > &Res,
				  ct::Size &szOut);

extern "C"
void cuda_back_deriv(const gpumat::GpuMat &Delta,
				const ct::Size &szOut,
				const ct::Size &szA0,
				int channels,
				const ct::Size &szW,
				int stride,
				gpumat::GpuMat &X);

extern "C"
void cuda_back_deriv_vec(const std::vector< gpumat::GpuMat > &Delta,
				const ct::Size &szOut,
				const ct::Size &szA0,
				int channels,
				const ct::Size &szW,
				int stride,
				std::vector< gpumat::GpuMat > &X);

extern "C"
void cuda_back_derivT(const gpumat::GpuMat &Delta,
				const ct::Size &szOut,
				const ct::Size &szA0,
				int channels,
				const ct::Size &szW,
				int stride,
				gpumat::GpuMat &X);

extern "C"
void cuda_back_derivT_vec(const std::vector< gpumat::GpuMat > &Delta,
				const ct::Size &szOut,
				const ct::Size &szA0,
				int channels,
				const ct::Size &szW,
				int stride,
				std::vector< gpumat::GpuMat > &X);

extern "C"
void cuda_subsample2(const gpumat::GpuMat &X,
					const ct::Size &szA,
					gpumat::GpuMat &Y,
					gpumat::GpuMat &Mask,
					ct::Size &szO);

extern "C"
void cuda_subsample2_vec(const std::vector< gpumat::GpuMat > &X,
					const ct::Size &szA,
					std::vector< gpumat::GpuMat > &Y,
					std::vector< gpumat::GpuMat > &Mask,
					ct::Size &szO);

extern "C"
void cuda_vec2mat(const std::vector< gpumat::GpuMat >& vec, gpumat::GpuMat& mat);

extern "C"
void cuda_mat2vec(const gpumat::GpuMat& mat, const ct::Size& sz, std::vector< gpumat::GpuMat >& vec);

extern "C"
void cuda_upsample2(const gpumat::GpuMat &Y, const gpumat::GpuMat &Mask, const ct::Size &szO,
			  const ct::Size &szA, gpumat::GpuMat &X);

extern "C"
void cuda_upsample2vec(const std::vector<gpumat::GpuMat> &Y, const std::vector<gpumat::GpuMat> &Mask,
			  const ct::Size &szO, const ct::Size &szA, std::vector<gpumat::GpuMat> &X);

///////////////////////////////

void gpumat::conv2::im2cols(const gpumat::GpuMat &X, const ct::Size &szA0,
							int channels, const ct::Size &szW,
			 int stride, gpumat::GpuMat &Res, ct::Size &szOut)
{
	if(X.empty() || ! channels || !szA0.area() || !szW.area() || !stride)
		throw new std::invalid_argument("im2cols: empty parameters");

	szOut.width = (szA0.width - szW.width)/stride + 1;
	szOut.height = (szA0.height - szW.height)/stride + 1;

	int rows = szOut.area();
	int cols = szW.area() * channels;

	Res.resize(rows, cols, X.type);

	cuda_im2cols(X, szA0, channels, szW, stride, Res, szOut);
}

void gpumat::conv2::im2colsT(const gpumat::GpuMat &X, const ct::Size &szA0,
							 int channels, const ct::Size &szW,
			 int stride, gpumat::GpuMat &Res, ct::Size &szOut)
{
	if(X.empty() || ! channels || !szA0.area() || !szW.area() || !stride)
		throw new std::invalid_argument("im2cols: empty parameters");

	szOut.width = (szA0.width - szW.width)/stride + 1;
	szOut.height = (szA0.height - szW.height)/stride + 1;

	int rows = szOut.area();
	int cols = szW.area() * channels;

	Res.resize(rows, cols, X.type);

	cuda_im2colsT(X, szA0, channels, szW, stride, Res, szOut);
}


void gpumat::conv2::im2cols(const std::vector<gpumat::GpuMat> &X, const ct::Size &szA0,
							int channels, const ct::Size &szW, int stride,
							std::vector<gpumat::GpuMat> &Res, ct::Size &szOut)
{
	if(X.empty() || X[0].empty() || ! channels || !szA0.area() || !szW.area() || !stride)
		throw new std::invalid_argument("im2cols: empty parameters");
	szOut.width = (szA0.width - szW.width)/stride + 1;

	szOut.height = (szA0.height - szW.height)/stride + 1;

	int rows = szOut.area();
	int cols = szW.area() * channels;
	int type = X[0].type;

	Res.resize(X.size());

	for(size_t i = 0; i < Res.size(); ++i){
		Res[i].resize(rows, cols, type);
	}

	cuda_im2cols_vec(X, szA0, channels, szW, stride, Res, szOut);
}

void gpumat::conv2::im2colsT(const std::vector<gpumat::GpuMat> &X, const ct::Size &szA0,
							 int channels, const ct::Size &szW, int stride,
							 std::vector<gpumat::GpuMat> &Res, ct::Size &szOut)
{
	if(X.empty() || X[0].empty() || ! channels || !szA0.area() || !szW.area() || !stride)
		throw new std::invalid_argument("im2colsT: empty parameters");
	szOut.width = (szA0.width - szW.width)/stride + 1;

	szOut.height = (szA0.height - szW.height)/stride + 1;

	int rows = szOut.area();
	int cols = szW.area() * channels;
	int type = X[0].type;

	Res.resize(X.size());

	for(size_t i = 0; i < Res.size(); ++i){
		Res[i].resize(rows, cols, type);
	}

	cuda_im2colsT_vec(X, szA0, channels, szW, stride, Res, szOut);
}

void gpumat::conv2::back_deriv(const gpumat::GpuMat &Delta,
				const ct::Size &szOut,
				const ct::Size &szA0,
				int channels,
				const ct::Size &szW,
				int stride,
				gpumat::GpuMat &X)
{
	if(Delta.empty() || ! channels || !szA0.area() || !szW.area() || !stride)
		throw new std::invalid_argument("back_deriv: empty parameters");

	X.resize(channels, szA0.area(), Delta.type);
	X.zeros();

	cuda_back_deriv(Delta, szOut, szA0, channels, szW, stride, X);
}

void gpumat::conv2::back_deriv(const std::vector<gpumat::GpuMat> &Delta,
							   const ct::Size &szOut, const ct::Size &szA0,
							   int channels, const ct::Size &szW, int stride,
							   std::vector<gpumat::GpuMat> &X)
{
	if(Delta.empty() || ! channels || !szA0.area() || !szW.area() || !stride)
		throw new std::invalid_argument("back_deriv: empty parameters");

	X.resize(Delta.size());

	int type = Delta[0].type;

	for(size_t i = 0; i < X.size(); ++i){
		X[i].resize(szA0.area(), channels, type);
		X[i].zeros();
	}

	cuda_back_deriv_vec(Delta, szOut, szA0, channels, szW, stride, X);

}

void gpumat::conv2::back_derivT(const gpumat::GpuMat &Delta,
				const ct::Size &szOut,
				const ct::Size &szA0,
				int channels,
				const ct::Size &szW,
				int stride,
				gpumat::GpuMat &X)
{
	if(Delta.empty() || ! channels || !szA0.area() || !szW.area() || !stride)
		throw new std::invalid_argument("back_derivT: empty parameters");

	X.resize(szA0.area(), channels, Delta.type);
	X.zeros();

	cuda_back_derivT(Delta, szOut, szA0, channels, szW, stride, X);
}

void gpumat::conv2::back_derivT(const std::vector<gpumat::GpuMat> &Delta,
								const ct::Size &szOut, const ct::Size &szA0,
								int channels, const ct::Size &szW, int stride,
								std::vector<gpumat::GpuMat> &X)
{
	if(Delta.empty() || ! channels || !szA0.area() || !szW.area() || !stride)
		throw new std::invalid_argument("back_derivT: empty parameters");

	X.resize(Delta.size());

	int type = Delta[0].type;

	for(size_t i = 0; i < X.size(); ++i){
		X[i].resize(szA0.area(), channels, type);
		X[i].zeros();
	}

	cuda_back_derivT_vec(Delta, szOut, szA0, channels, szW, stride, X);

}

void gpumat::conv2::subsample(const gpumat::GpuMat &X,
							  const ct::Size &szA,
							  gpumat::GpuMat &Y,
							  gpumat::GpuMat &Mask,
							  ct::Size &szO)
{
	if(X.empty() || X.rows != szA.area())
		throw new std::invalid_argument("subsample: empty parameters");

	szO.width = szA.width / 2;
	szO.height = szA.height / 2;
	int K = X.cols;

	Y.resize(szO.area(), K, X.type);
	Mask.resize(X.rows, X.cols, X.type);
	Mask.zeros();

	cuda_subsample2(X, szA, Y, Mask, szO);
}

void gpumat::conv2::subsample(const std::vector<gpumat::GpuMat> &X,
							  const ct::Size &szA,
							  std::vector<gpumat::GpuMat> &Y,
							  std::vector<gpumat::GpuMat> &Mask,
							  ct::Size &szO)
{
	if(X.empty() || X[0].rows != szA.area())
		throw new std::invalid_argument("subsample: empty parameters");

	szO.width = szA.width / 2;
	szO.height = szA.height / 2;

	int K = X[0].cols;

	Y.resize(X.size());
	Mask.resize(X.size());

	for(size_t i = 0; i < X.size(); ++i){
		Y[i].resize(szO.area(), K, X[i].type);
		Y[i].zeros();
		Mask[i].resize(X[i].rows, X[i].cols, X[i].type);
		Mask[i].zeros();
	}

	cuda_subsample2_vec(X, szA, Y, Mask, szO);
}

void gpumat::conv2::vec2mat(const std::vector<gpumat::GpuMat> &vec, gpumat::GpuMat &mat)
{
	if(vec.empty() || vec[0].empty())
		throw new std::invalid_argument("vec2mat: empty parameters");

	int rows = (int)vec.size();
	int cols = vec[0].total();

	mat.resize(rows, cols, vec[0].type);

	cuda_vec2mat(vec, mat);
}

void gpumat::conv2::mat2vec(const gpumat::GpuMat &mat, const ct::Size &szOut,
							std::vector<gpumat::GpuMat> &vec)
{
	if(mat.empty())
		throw new std::invalid_argument("mat2vec: empty parameters");

	int rows = mat.rows;

	vec.resize(rows);

	for(size_t i = 0; i < vec.size(); ++i){
		vec[i].resize(szOut.height, szOut.width, mat.type);
	}

	cuda_mat2vec(mat, szOut, vec);
}

void gpumat::conv2::upsample(const gpumat::GpuMat &Y, int K,
							 const gpumat::GpuMat &Mask, const ct::Size &szO,
			  const ct::Size &szA, gpumat::GpuMat &X)
{
	if(Y.empty() || Mask.empty() || Y.total() != szO.area() * K)
		throw new std::invalid_argument("upsample: empty parameters");

	X.resize(szA.area(), K, Y.type);

	cuda_upsample2(Y, Mask, szO, szA, X);
}

void gpumat::conv2::upsample(const std::vector<gpumat::GpuMat> &Y,
							 int K, const std::vector<gpumat::GpuMat> &Mask,
			  const ct::Size &szO, const ct::Size &szA, std::vector<gpumat::GpuMat> &X)
{
	if(Y.empty() || Y[0].empty() || Mask.empty() || Mask[0].empty() || Y[0].total() != szO.area() * K)
		throw new std::invalid_argument("upsample: empty parameters");

	X.resize(Y.size());

	int type = Y[0].type;
	int area = szA.area();

	for(size_t i = 0; i < X.size(); ++i){
		X[i].resize(area, K, type);
		X[i].zeros();
	}

	cuda_upsample2vec(Y, Mask, szO, szA, X);

}

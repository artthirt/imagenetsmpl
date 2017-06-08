#include "convnn_gpu.h"
#include "cuda_common.h"

#include "exception"

#include "qt_work_mat.h"

using namespace gpumat;

convnn::convnn()
{
	stride = 1;
	weight_size = 3;
	m_init = false;
	pA0 = nullptr;
	m_use_pool = true;
}

void convnn::setWeightSize(int ws, bool use_pool)
{
	weight_size = ws;
	if(W.size())
		init((int)W.size(), szA0, use_pool);
}

void convnn::init(int count_weight, const ct::Size &_szA0, int use_pool)
{
	W.resize(count_weight);
	B.resize(count_weight);

	szA0 = _szA0;

	m_use_pool = use_pool;

	ct::get_cnv_sizes(szA0, ct::Size(weight_size, weight_size), stride, szA1, szA2);
//	if(!m_use_pool)
//		szA2 = szA1;

	update_random();

	m_init = true;
}

void convnn::update_random()
{
	for(size_t i = 0; i < W.size(); ++i){
		ct::Matf Wi(weight_size, weight_size), Bi(1, 1);
		Wi.randn(0, 0.1);
		gpumat::convert_to_gpu(Wi, W[i]);
		Bi.randn(0, 0.1);
		gpumat::convert_to_gpu(Bi, B[i]);
	}
}

void convnn::clear()
{
	pA0 = nullptr;
//	A0.free();
	A1.clear();
	A1.clear();
	Masks.clear();
}

bool convnn::use_pool() const
{
	return m_use_pool;
}

ct::Size convnn::szOut() const{
	if(m_use_pool)
		return szA2;
	else
		return szA1;
}

void convnn::forward(const GpuMat *mat, etypefunction func)
{
	if(!m_init || !mat)
		throw new std::invalid_argument("convnn::forward: not initialized. wrong parameters");
	pA0 = (GpuMat*)mat;//mat.copyTo(A0);
	gpumat::conv2D(*pA0, szA0, stride, W, B, A1, func);

	if(m_use_pool){
		ct::Size sztmp;
		gpumat::subsample(A1, szA1, A2, Masks, sztmp);
	}
}

void convnn::apply_func(const GpuMat &A, GpuMat &B, etypefunction func)
{
	switch (func) {
		default:
		case RELU:
			gpumat::deriv_reLu(A, B);
			break;
	}
}

void convnn::back2conv(const tvmat &A1, const tvmat &dA2, tvmat &dA1, etypefunction func)
{
	dA1.resize(A1.size());
//#pragma omp parallel for
	for(int i = 0; i < (int)A1.size(); i++){
		apply_func(A1[i], dA1[i], func);
		gpumat::elemwiseMult(dA1[i], dA2[i]);
	}
}

void convnn::back2conv(const tvmat &A1, const tvmat &dA2, int first, int last, tvmat &dA1, etypefunction func)
{
	dA1.resize(last - first);
//#pragma omp parallel for
	for(int i = 0; i < (int)dA1.size(); i++){
		apply_func(A1[i], dA1[i], func);
		gpumat::elemwiseMult(dA1[i], dA2[i +first]);
	}
}

void convnn::back2conv(const tvmat &A1, const std::vector<convnn> &dA2, int first, int last, tvmat &dA1, etypefunction func)
{
	dA1.resize(last - first);
//#pragma omp parallel for
	for(int i = 0; i < (int)dA1.size(); i++){
		apply_func(A1[i], dA1[i], func);
		gpumat::elemwiseMult(dA1[i], dA2[i + first].DltA0);
	}
}

void convnn::backward(const std::vector<GpuMat> &Delta, etypefunction func, int first, int last, bool last_layer)
{
	if(!m_init || !pA0)
		throw new std::invalid_argument("convnn::backward: not initialized");

	if(m_use_pool){
		gpumat::upsample(Delta, szA2, szA1, Masks, dA2, first, last);
		back2conv(A1, dA2, dA1, func);
	}else{
		back2conv(A1, Delta, first, last, dA1, func);
	}

//	for(int i = 0; i < dA2.size(); ++i){
//		qt_work_mat::q_save_mat(Masks[i], QString("_Masks_%1.txt").arg(i));
//		qt_work_mat::q_save_mat(dA2[i], QString("_dA2_%1.txt").arg(i));
//	}

	ct::Size szW(weight_size, weight_size);

	gpumat::deriv_conv2D(*pA0, dA1, szA0, szA1, szW, stride, gradW, gradB, &m_tmp1);

	if(!last_layer)
		gpumat::deriv_prev_cnv(dA1, W, szA1, szA0, stride, DltA0);

//	for(int i = 0; i < gradW.size(); ++i){
//		std::stringstream ss;
//		ss << "_gW_" << i << ".txt";
//		qt_work_mat::q_save_mat(gradW[i], ss.str().c_str());
//		ss.str("");
//		ss << "_dA1_" << i << ".txt";
//		qt_work_mat::q_save_mat(dA1[i], ss.str().c_str());
//		ss.str("");
//		ss << "_W_" << i << ".txt";
//		qt_work_mat::q_save_mat(W[i], ss.str().c_str());
//	}
//	qt_work_mat::q_save_mat(A0, "_A0.txt");
//	qt_work_mat::q_save_mat(DltA0, "_DltA0.txt");

//	m_optim.pass(gradW, gradB, W, B);
}

void convnn::backward(const std::vector<convnn> &Delta, etypefunction func, int first, int last, bool last_layer)
{
	if(!m_init || !pA0)
		throw new std::invalid_argument("convnn::backward: not initialized");

	if(m_use_pool){
		convnn::upsample(Delta, szA2, szA1, Masks, dA2, first, last);
		back2conv(A1, dA2, dA1, func);
	}else{
		back2conv(A1, Delta, first, last, dA1, func);
	}

	ct::Size szW(weight_size, weight_size);

	gpumat::deriv_conv2D(*pA0, dA1, szA0, szA1, szW, stride, gradW, gradB, &m_tmp1);

	if(!last_layer)
		gpumat::deriv_prev_cnv(dA1, W, szA1, szA0, stride, DltA0);
//	m_optim.pass(gradW, gradB, W, B);
}

void convnn::hconcat(const std::vector<convnn> &cnv, GpuMat &_out)
{
	if(cnv.empty())
		return;

	slice.resize(cnv.size());
	for(size_t i = 0; i < cnv.size(); ++i){
		gpumat::hconcat(cnv[i].A2, slice[i]);
	}
	gpumat::hconcat(slice, _out);
}

void convnn::upsample(const std::vector<convnn> &A1, ct::Size &szA1, const ct::Size &szA0, const std::vector<GpuMat> &Masks, std::vector<GpuMat> &A0, int first, int last)
{
	if(A1.empty() || Masks.empty())
		throw new std::invalid_argument("gpumat::upsample: invalid parameters");

	if(first >= 0 && last > first){
		A0.resize(last - first);
	}else{
		A0.resize(A1.size());
		first = 0;
		last = (int)A1.size();
	}

	for(int i = first, j = 0; i < last; ++i, ++j){
		gpumat::upsample(A1[i].DltA0, szA1, szA0, Masks[j], A0[j]);
	}
}

void convnn::write(std::fstream &fs)
{
	if(!fs.is_open() || W.empty() || B.empty())
		return;

	for(size_t i = 0; i < W.size(); ++i){
		gpumat::GpuMat &Wi = W[i];
		gpumat::GpuMat &Bi = B[i];
		gpumat::write_fs(fs, Wi);
		gpumat::write_fs(fs, Bi);
	}
}

void convnn::read(std::fstream &fs)
{
	if(!fs.is_open() || W.empty() || B.empty())
		return;

	for(size_t i = 0; i < W.size(); ++i){
		gpumat::GpuMat &Wi = W[i];
		gpumat::GpuMat &Bi = B[i];
		gpumat::read_fs(fs, Wi);
		gpumat::read_fs(fs, Bi);
	}
}

/////////////////////////////////////////
/////////////////////////////////////////


void ConvOptim::init(const std::vector<std::vector<convnn> > &cnv)
{
	if(cnv.empty())
		return;

	m_optim.resize(cnv.size());

	for(size_t i = 0; i < cnv.size(); ++i){
		m_optim[i].resize(cnv[i].size());
		for(size_t j = 0; j < cnv[i].size(); ++j){
			const convnn& c = cnv[i][j];
			AdamOptimizer& a = m_optim[i][j];
			a.init(c.W, c.B);
		}
	}
}

void ConvOptim::pass(std::vector<std::vector<convnn> > &cnv)
{
	if(cnv.empty() || m_optim.empty())
		return;

	for(size_t i = 0; i < m_optim.size(); ++i){
		for(size_t j = 0; j < m_optim[i].size(); ++j){
			convnn& c = cnv[i][j];
			AdamOptimizer& a = m_optim[i][j];
			a.pass(c.gradW, c.gradB, c.W, c.B);
		}
	}

}

void ConvOptim::setAlpha(double val)
{
	if(m_optim.empty())
		return;
	for(size_t i = 0; i < m_optim.size(); ++i){
		for(size_t j = 0; j < m_optim[i].size(); ++j){
			AdamOptimizer& a = m_optim[i][j];
			a.setAlpha(val);
		}
	}
}

/////////////////////////////////////////
///////**********************////////////
/////////////////////////////////////////

ConvNN::ConvNN()
{

}

void ConvNN::setAlpha(float alpha)
{
	m_optim.setAlpha(alpha);
}

int ConvNN::outputFeatures() const
{
	return (int)(m_conv.back()[0].W.size() * m_conv.back()[0].szOut().area() * m_conv.back().size());
}

int ConvNN::outputMatrices() const
{
	return (int)(m_conv.back()[0].W.size() * m_conv.back().size());
}

void ConvNN::init()
{
	if(m_cnvlayers.empty() || m_cnvweights.empty())
		throw new std::invalid_argument("empty arguments");

	m_conv.resize(m_cnvlayers.size());

	ct::Size szA0 =m_szA0;

	int input = 1;
	for(size_t i = 0; i < m_conv.size(); ++i){
		m_conv[i].resize(input);

		bool pool = true;
		if(m_cnvpooling.size())
			pool = m_cnvpooling[i];

		for(size_t j = 0; j < m_conv[i].size(); ++j){
			convnn& cnv = m_conv[i][j];
			cnv.setWeightSize(m_cnvweights[i], pool);
			cnv.init(m_cnvlayers[i], szA0, pool);
		}
		input = m_cnvlayers[i] * input;
		szA0 = m_conv[i][0].szOut();
	}

	m_optim.init(m_conv);
}

void ConvNN::setConvLayers(const std::vector<int> &layers, std::vector<int> weight_sizes,
						   const ct::Size szA0, std::vector<char> *pooling)
{
	if(layers.empty() || weight_sizes.empty())
		throw new std::invalid_argument("empty parameters");

	if(pooling)
		m_cnvpooling = *pooling;
	m_cnvlayers = layers;
	m_cnvweights = weight_sizes;
	m_szA0 = szA0;
}

void ConvNN::conv(const GpuMat &X,GpuMat &XOut)
{
	if(X.empty())
		return;

	for(size_t i = 0; i < m_conv.size(); ++i){
		std::vector< gpumat::convnn >& ls = m_conv[i];

		if(i == 0){
			gpumat::convnn& m0 = ls[0];
			m0.forward(&X, gpumat::RELU);
		}else{
//#pragma omp parallel for
			for(int j = 0; j < (int)m_conv[i - 1].size(); ++j){
				size_t off1 = j * m_cnvlayers[i - 1];
				gpumat::convnn& m0 = m_conv[i - 1][j];
				for(int k = 0; k < m_cnvlayers[i - 1]; ++k){
					size_t col = off1 + k;
					gpumat::convnn& mi = ls[col];
					if(m0.use_pool())
						mi.forward(&m0.A2[k], gpumat::RELU);
					else
						mi.forward(&m0.A1[k], gpumat::RELU);
				}
			}
		}
	}

	m_adds.hconcat(m_conv.back(), XOut);
}

void ConvNN::backward(const GpuMat &X)
{
	if(m_cnvlayers.empty() || m_cnvweights.empty())
		throw new std::invalid_argument("empty arguments");

	int cols = (int)(m_conv.back().size() * m_conv.back()[0].W.size());

	hsplit(X, cols, m_features);

	for(int i = (int)m_conv.size() - 1; i > -1; i--){
		std::vector< convnn >& lrs = m_conv[i];

//			qDebug("LR[%d]-----", i);
		size_t kidx = 0;

//#pragma omp parallel for
		for(int j = 0; j < (int)lrs.size(); ++j){
			convnn &cnv = lrs[j];

			size_t kfirst = kidx;
			kidx += cnv.W.size();

			if(i == (int)m_conv.size() - 1)
				cnv.backward(m_features, gpumat::RELU, (int)kfirst, (int)kidx, i == 0);
			else
				cnv.backward(m_conv[i + 1], gpumat::RELU, (int)kfirst, (int)kidx, i == 0);
		}
//			qDebug("----");
	}

	m_optim.pass(m_conv);
}

std::vector<tvconvnn> &ConvNN::cnv()
{
	return m_conv;
}

void ConvNN::write(std::fstream &fs)
{
	if(!fs.is_open() || !m_conv.size())
		return;

	for(size_t i = 0; i < m_conv.size(); ++i){
		for(size_t j = 0; j < m_conv[i].size(); ++j){
			gpumat::convnn& cnv = m_conv[i][j];
			cnv.write(fs);
		}
	}
}

void ConvNN::read(std::fstream &fs)
{
	if(!fs.is_open() || !m_conv.size())
		return;

	for(size_t i = 0; i < m_conv.size(); ++i){
		for(size_t j = 0; j < m_conv[i].size(); ++j){
			gpumat::convnn& cnv = m_conv[i][j];
			cnv.read(fs);
		}
	}
}

std::vector<tvconvnn> &ConvNN::operator ()()
{
	return m_conv;
}

/////////////////////////////////////////
///////**********************////////////
/////////////////////////////////////////

extern "C"
void cuda_conv2d(const GpuMat &A0,
				 const ct::Size &szI,
				 const ct::Size &szO,
				 int stride,
				 const std::vector<GpuMat> &W,
				 const std::vector<GpuMat> B,
				 std::vector<GpuMat> &A1,
				 etypefunction func);


extern "C"
void cuda_subsample(const GpuMat &A0,
					const ct::Size &szA0,
					const ct::Size &szA1,
					GpuMat &A1,
					GpuMat &Mask);

extern "C"
void cuda_upsample(const GpuMat &A1, const ct::Size &szA1,
			  const ct::Size &szA0, const GpuMat &Mask, GpuMat &A0);

extern "C"
void cuda_deriv_conv2d(const GpuMat &A0, const GpuMat &gradA1,
				  const ct::Size &szA0, const ct::Size &szA1,
				  int stride, GpuMat &gradW, GpuMat &gradB,
					   GpuMat* pblock);

extern "C"
void cuda_deriv_prev_conv2d(const std::vector<GpuMat> &deriv,
							const std::vector<GpuMat> &W,
							const ct::Size &sL, const ct::Size &sLsub1, int stride,
							GpuMat &D);

extern "C"
void cuda_hsplit(const GpuMat &res, std::vector<GpuMat> &list);

extern "C"
void cuda_hconcat(const std::vector<GpuMat> &list, GpuMat &res);

extern "C"
void cuda_reduce_all(const GpuMat& A, GpuMat &res);

/////////////////////////////

namespace gpumat{

ct::Size conv2D(const GpuMat &A0,
			const ct::Size &szI,
			int stride,
			const std::vector<GpuMat> &W,
			const std::vector<GpuMat> B,
			std::vector<GpuMat> &A1,
			etypefunction func)
{
	if(A0.empty() || W.empty() || B.empty())
		throw new std::invalid_argument("gpumat::conv2D: check parameters");

	if(A1.size() != W.size())
		A1.resize(W.size());

	int w_rows = W[0].rows;
	int w_cols = W[0].cols;

	ct::Size szO;
	szO.width	= (szI.width - w_cols + 1) / stride;
	szO.height	= (szI.height - w_rows + 1) / stride;

	int sz = szO.area();

	for(size_t i = 0; i < A1.size(); ++i)
		A1[i].resize(A0.rows, sz, A0.type);

	cuda_conv2d(A0, szI, szO, stride, W, B, A1, func);

	return szO;
}

void subsample(const GpuMat &A0, const ct::Size &szA0, GpuMat &A1, GpuMat &Mask, ct::Size &szA1)
{
	if(A0.empty())
		throw new std::invalid_argument("gpumat::subsample: invalid parameters");

	int rows = A0.rows;
	int cols = A0.cols;

	if(!rows || !cols)
		throw new std::invalid_argument("gpumat::subsample: invalid parameters");

	szA1 = ct::Size(szA0.width/2, szA0.height/2);

	A1.resize(rows, szA1.area(), A0.type);
	Mask.resize(rows, szA0.area(), A0.type);

	cuda_subsample(A0, szA0, szA1, A1, Mask);
}

void subsample(const std::vector<GpuMat> &A0, const ct::Size &szA0,
			   std::vector<GpuMat> &A1, std::vector<GpuMat> &Masks,
			   ct::Size &szA1)
{
	if(A0.empty())
		throw new std::invalid_argument("gpumat::subsample: invalid parameters");
	A1.resize(A0.size());
	Masks.resize(A0.size());

//#pragma omp parallel for
	for(int i = 0; i < (int)A0.size(); i++){
		subsample(A0[i], szA0, A1[i], Masks[i], szA1);
	}
}

void upsample(const GpuMat &A1, const ct::Size &szA1,
			  const ct::Size &szA0, const GpuMat &Mask, GpuMat &A0)
{
	if(A1.empty() || Mask.empty())
		throw new std::invalid_argument("gpumat::upsample: invalid parameters");

	int m = A1.rows;

	A0.resize(m, szA0.area(), A1.type);

	cuda_upsample(A1, szA1, szA0, Mask, A0);
}

void upsample(const std::vector<GpuMat> &A1, ct::Size &szA1, const ct::Size &szA0, const std::vector<GpuMat> &Masks, std::vector<GpuMat> &A0, int first, int last)
{
	if(A1.empty() || Masks.empty())
		throw new std::invalid_argument("gpumat::upsample: invalid parameters");

	if(first >= 0 && last > first){
		A0.resize(last - first);
	}else{
		A0.resize(A1.size());
		first = 0;
		last = (int)A1.size();
	}

//#pragma omp parallel for
	for(int j = 0; j < (int)A0.size(); ++j){
		int i = first + j;
		upsample(A1[i], szA1, szA0, Masks[j], A0[j]);
	}
}

void deriv_conv2D(const GpuMat &A0, const GpuMat &gradA1,
				  const ct::Size &szA0, const ct::Size &szA1,
				  const ct::Size &szW, int stride,
				  GpuMat &gradW, GpuMat &gradB, GpuMat *pblock)
{
	if(A0.empty() || gradA1.empty() || !stride){
		std::cout << "gpumat::deriv_conv2D wrong parameters\n";
	}

	gradW.resize(szW.height, szW.width, A0.type);
	gradB.resize(1, 1, A0.type);

	memset(gradW, 0);

	cuda_deriv_conv2d(A0, gradA1, szA0, szA1, stride, gradW, gradB, pblock);

	// need reduce for B
	// may be work in cuda_deriv_conv2d
}


void deriv_conv2D(const GpuMat &A0,
				  const std::vector<GpuMat> &gradA1,
				  const ct::Size &szA0,
				  const ct::Size &szA1,
				  const ct::Size &szW,
				  int stride,
				  std::vector<GpuMat> &gradW, std::vector<GpuMat> &gradB,
				  std::vector<GpuMat> *pblocks)
{
	if(A0.empty() || gradA1.empty())
		throw new std::invalid_argument("gpumat::deriv_conv2D: invalid parameters");

	gradW.resize(gradA1.size());
	gradB.resize(gradA1.size());

	if(pblocks){
		pblocks->resize(gradA1.size());
	}

//#pragma omp parallel for
	for(int i = 0; i < (int)gradA1.size(); ++i){
		deriv_conv2D(A0, gradA1[i], szA0, szA1, szW, stride, gradW[i], gradB[i], pblocks? &(*pblocks)[i] : nullptr);
	}
}

void deriv_prev_cnv(const std::vector<GpuMat> &deriv, const std::vector<GpuMat> &W,
					const ct::Size &sL, const ct::Size &sLsub1, int stride, GpuMat &D)
{
	if(deriv.empty() || W.empty())
		std::cout << "gpumat::deriv_prev_cnv wrong parameters\n";

	D.resize(deriv[0].rows, sLsub1.area(), deriv[0].type);

	cuda_deriv_prev_conv2d(deriv, W, sL, sLsub1, stride, D);
}

void hsplit(const GpuMat &res, int cols, std::vector<GpuMat> &list)
{
	if(res.empty() || (res.cols % cols) != 0)
		throw new std::invalid_argument("hsplit: wrong parameters");

	int len = res.cols / cols;

	list.resize(cols);

	for(int i = 0; i < cols; ++i){
		list[i].resize(res.rows, len, res.type);
	}

	cuda_hsplit(res, list);
}

void hconcat(const std::vector<GpuMat> &list, GpuMat &res)
{
	if(list.empty())
		return;
	int rows		= list[0].rows;
	int loc_cols	= list[0].cols;
	int cols		= loc_cols * (int)list.size();

	res.resize(rows, cols, list[0].type);

	cuda_hconcat(list, res);

//	T *dR = res.ptr();

//#pragma omp parallel for
//	for(int i = 0; i < rows; ++i){
//#pragma omp parallel for
//		for(int j = 0; j < (int)list.size(); ++j){
//			T* dL = list[j].ptr();
//#ifdef __GNUC__
//#pragma omp simd
//#endif
//			for(int k = 0; k < loc_cols; ++k){
//				dR[i * cols + j * loc_cols + k] = dL[i * loc_cols + k];
//			}
//		}
	//	}
}

void reduce(const GpuMat &mat, GpuMat &res)
{
	if(mat.empty())
		throw new std::invalid_argument("reduce: mat is empty");

	res.resize(1, 1, mat.type);

	cuda_reduce_all(mat, res);
}


}

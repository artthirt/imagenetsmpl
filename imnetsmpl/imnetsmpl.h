#ifndef IMNETSMPL_H
#define IMNETSMPL_H

#include "imreader.h"

#include "convnn2.h"
#include "mlp.h"

class ImNetSmpl
{
public:
	ImNetSmpl();

	void setReader(ImReader* ir);
	void setLearningRate(double lr);

	void init();
	void doPass(int pass, int batch);

	void forward(const std::vector< ct::Matf >& X, ct::Matf& yOut, bool dropout = false);
	void backward(const ct::Matf& Delta);
	ct::Matf predict(ct::Matf &y);
    ct::Matf predict(const std::string& name, bool show_debug = false);
    void predicts(const std::string& sdir);
	float loss(const ct::Matf &y, ct::Matf &y_);

	/**
	 * @brief setSaveModelName
	 * name for saved model
	 * @param name
	 */
    void setSaveModelName(const std::string name);

	/**
	 * @brief save_net
	 * @param name
	 */
    void save_net(const std::string& name);
	/**
	 * @brief load_net
	 * @param name
	 */
    void load_net(const std::string& name);

	/**
	 * @brief save_net2
	 * @param name
	 */
    void save_net2(const std::string& name);
	/**
	 * @brief load_net2
	 * @param name
	 */
    void load_net2(const std::string& name);

    void setModelName(const std::string& name);

	void setUseBackConv(bool val);

private:
	ImReader *m_reader;
	double m_learningRate;

	bool m_useBackConv;

	std::vector< conv2::convnnf > m_conv;
	std::vector< ct::mlpf > m_mlp;
	int m_classes;
	ct::Matf m_A1;
//	ct::Matf m_A2;
//	ct::Matf m_Aout;
//	ct::Matf D1;
//	ct::Matf D2;
	std::vector< ct::Matf > deltas1;
//	std::vector< ct::Matf > deltas2;
	ct::MlpAdamOptimizer<float> m_optim;
	conv2::CnvAdamOptimizer<float> m_cnv_optim;

    std::string m_model;
    std::string m_save_model;
	int m_check_pass;


	int m_check_count;
	bool m_init;
};

#endif // IMNETSMPL_H

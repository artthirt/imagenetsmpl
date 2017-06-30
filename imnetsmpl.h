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

	void forward(const std::vector< ct::Matf >& X, ct::Matf& yOut);
	void backward(const ct::Matf& Delta);
	ct::Matf predict(ct::Matf &y);
	ct::Matf predict(const QString& name, bool show_debug = false);
	void predicts(const QString& sdir);
	float loss(const ct::Matf &y, ct::Matf &y_);

	/**
	 * @brief setSaveModelName
	 * name for saved model
	 * @param name
	 */
	void setSaveModelName(const QString name);

	/**
	 * @brief save_net
	 * @param name
	 */
	void save_net(const QString& name);
	/**
	 * @brief load_net
	 * @param name
	 */
	void load_net(const QString& name);

	/**
	 * @brief save_net2
	 * @param name
	 */
	void save_net2(const QString& name);
	/**
	 * @brief load_net2
	 * @param name
	 */
	void load_net2(const QString& name);

	void setModelName(const QString& name);

private:
	ImReader *m_reader;
	double m_learningRate;

	std::vector< conv2::convnn<float> > m_conv;
	std::vector< ct::mlp<float> > m_mlp;
	std::vector< ct::MomentOptimizer<float> > m_mg;
	int m_classes;
	ct::Matf m_A1;
//	ct::Matf m_A2;
//	ct::Matf m_Aout;
//	ct::Matf D1;
//	ct::Matf D2;
	std::vector< ct::Matf > deltas1;
//	std::vector< ct::Matf > deltas2;
	ct::MlpOptim<float> m_optim;
	QString m_model;
	QString m_save_model;


	int m_check_count;
	bool m_init;
};

#endif // IMNETSMPL_H

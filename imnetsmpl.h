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

	void init();
	void doPass(int pass, int batch);

	void forward(const std::vector< ct::Matf >& X, ct::Matf& yOut);
	void backward(const ct::Matf& Delta);
	ct::Matf predict(ct::Matf &y);
	ct::Matf predict(const QString& name, bool show_debug = false);
	float loss(const ct::Matf &y, ct::Matf &y_);

	void save_net(const QString& name);
	void load_net(const QString& name);

	void setModelName(const QString& name);

private:
	ImReader *m_reader;

	std::vector< conv2::convnn<float> > m_conv;
	std::vector< ct::mlp<float> > m_mlp;
	int m_classes;
	ct::Matf m_A1;
	ct::MlpOptim<float> m_optim;
	QString m_model;

	bool m_init;
};

#endif // IMNETSMPL_H

#include "qt_work_mat.h"

#include <QFile>
#include <QString>

#include "helper_gpu.h"

void qt_work_mat::q_save_mat(const ct::Matf &mat, const QString &filename)
{
	if(filename.isEmpty() || mat.empty())
		return;

	QFile fs(filename);

	if(!fs.open(QIODevice::WriteOnly))
		return;

	QString tmp;

	tmp = QString("rows=%1\ncols=%2\n").arg(mat.rows).arg(mat.cols);
	fs.write(tmp.toLatin1());

	float* dM = mat.ptr();
	for(int i = 0; i < mat.rows; ++i){
		tmp = "";
		for (int j = 0; j < mat.cols; ++j){
			float val = dM[i * mat.cols + j];
			if(!tmp.isEmpty())
				tmp += " ";
			tmp += QString::number(val, 'f', 8);
		}
		tmp += "\n";
		fs.write(tmp.toLatin1());
	}

	fs.close();
}

void qt_work_mat::q_load_mat(const QString &filename, ct::Matf &mat)
{
	if(filename.isEmpty())
		return;

	if(!QFile::exists(filename))
		return;

	QFile fs(filename);

	if(!fs.open(QIODevice::ReadOnly))
		return;

	QString tmp;

	int rows = 0, cols = 0, idx = 0;

	int type = 0;
	float *dM = 0;
	QStringList sl;

	while(!fs.atEnd()){
		if(type == 0){
			tmp = fs.readLine();
			tmp = tmp.trimmed();
			sl = tmp.split('=');
			if(sl[0] == "rows")
				rows = sl[1].toFloat();

			if(sl[0] == "cols")
				cols = sl[1].toFloat();
			if(rows && cols){
				type = 1;

				mat.setSize(rows, cols);
				dM = mat.ptr();
				idx = 0;
			}
		}else
		if(type == 1){
			tmp = fs.readLine();
			tmp = tmp.trimmed();
			if(tmp.isEmpty())
				continue;
			sl = tmp.split(' ');
			for (int j = 0; j < mat.cols; ++j){
				dM[idx * mat.cols + j] = sl[j].toFloat();
			}
			idx++;
		}
	}


	fs.close();
}

void qt_work_mat::q_save_mat(const gpumat::GpuMat &mat, const QString &filename)
{
	if(mat.type != gpumat::GPU_FLOAT || mat.empty())
		return;

	ct::Matf lmat;
	gpumat::convert_to_mat(mat, lmat);
	q_save_mat(lmat, filename);
}

void qt_work_mat::q_load_mat(const QString &filename, gpumat::GpuMat &mat)
{
	ct::Matf matf;
	q_load_mat(filename, matf);
	if(!matf.empty())
		gpumat::convert_to_gpu(matf, mat);
}

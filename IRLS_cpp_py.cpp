#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <cmath>
#include "math.h"

#include <Eigen/LU>
#include <Eigen/Dense>

namespace py = pybind11;
using namespace Eigen;

double ss(VectorXd x)
{
	using namespace std;
	return std::sqrt(x.transpose()*x);
}

VectorXd logit(VectorXd x)
{
	int n = x.size();
	VectorXd res(n);
	for(int i=0; i<n; i++)
	{
		res(i) = 1/(1+std::exp(-x(i)));
	}
	return res;
}

VectorXd poi_mean(VectorXd x)
{
	int n = x.size();
	VectorXd res(n);
	for(int i=0; i<n; i++)
	{
		res(i) = std::exp(x(i));
	}
	return res;
}


MatrixXd mat_times_vec(MatrixXd mat, VectorXd vec)
{
	MatrixXd res(mat.rows(), mat.cols());
	for(int i=0; i<mat.rows(); i++)
	{
		for(int j=0; j<mat.cols(); j++)
		{
			res(i,j) = mat(i,j) * vec(i);
		}
	}
	return res;
}

VectorXd IRLS(MatrixXd X, VectorXd y, double tol) {


	int nc = X.cols();
	int nr = X.rows();
	double crit;
	VectorXd dif(nc);
	VectorXd linpred(nr);
	VectorXd ones = VectorXd::Ones(nr);
	VectorXd res(nc+1);
	MatrixXd W;

	VectorXd beta = VectorXd::Zero(nc);

	int iter = 0;

	do {
		iter += 1;
		linpred = logit(X*beta);
		W = (linpred.cwiseProduct(ones - linpred)).asDiagonal();
		dif = (X.transpose()*W*X).
			colPivHouseholderQr().
			solve(X.transpose()*(y-linpred));
		beta += dif;
		crit = ss(dif) / (ss(beta) + 0.1);
	
	} while(crit > tol);

	res << beta, iter;

	return res;
}

VectorXd IRLS2(MatrixXd X, VectorXd y, double tol) {


	int nc = X.cols();
	int nr = X.rows();
	double crit;
	VectorXd dif(nc);
	VectorXd linpred(nr);
	VectorXd linpred2(nr);
	VectorXd res(nc+1);
	VectorXd ones = VectorXd::Ones(nr);
	VectorXd beta = VectorXd::Zero(nc);

	int iter = 0;

	do {
		iter += 1;
		linpred = logit(X*beta);
		linpred2 = linpred.cwiseProduct(ones - linpred);
		dif = (X.transpose() * mat_times_vec(X, linpred2)).
			colPivHouseholderQr().
			solve(X.transpose()*(y-linpred));
		beta += dif;
		crit = ss(dif) / (ss(beta) + 0.1);
	
	} while(crit > tol);

	res << beta, iter;

	return res;
}

VectorXd IRLS_poi(MatrixXd X, VectorXd y, double tol) {


	int nc = X.cols();
	int nr = X.rows();
	double crit;
	VectorXd dif(nc);
	VectorXd linpred(nr);
	VectorXd res(nc+1);
	VectorXd ones = VectorXd::Ones(nr);
	VectorXd beta = VectorXd::Zero(nc);
	beta(0) = y.mean();

	int iter = 0;

	do {
		iter += 1;
		linpred = poi_mean(X*beta);
		dif = (X.transpose() * mat_times_vec(X, linpred)).
			colPivHouseholderQr().
			solve(X.transpose()*(y-linpred));
		beta += dif;
		crit = ss(dif) / (ss(beta) + 0.1);
	
	} while(crit > tol);

	res << beta, iter;

	return res;
}



PYBIND11_PLUGIN(irls) {
    pybind11::module m("irls", "auto-compiled c++ extension");
    m.def("logit", &logit);
	m.def("IRLS", &IRLS);
	m.def("mat_times_vec", &mat_times_vec);
	m.def("IRLS2", &IRLS2);
	m.def("IRLS_poi", &IRLS_poi);
    return m.ptr();
}


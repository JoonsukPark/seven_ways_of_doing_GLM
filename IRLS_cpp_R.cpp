#include <RcppEigen.h>
#include <math.h>
#include <Eigen/LU>
#include <Eigen/Dense>

// [[Rcpp::depends(RcppEigen)]]
using namespace Eigen;
using namespace std;

// [[Rcpp::export]]
double ss(VectorXd x)
{
  return sqrt(x.transpose()*x);
}

// [[Rcpp::export]]
VectorXd logit(VectorXd x)
{
  int n = x.size();
  VectorXd res(n);
  for(int i=0; i<n; i++)
  {
    res(i) = 1/(1+exp(-x(i)));
  }
  return res;
}

// [[Rcpp::export]]
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

// [[Rcpp::export]]
VectorXd IRLS_cpp(MatrixXd X, VectorXd y, double tol) {
  
  
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

library(pracma)
library(Rcpp)
library(RcppEigen)
sourceCpp('IRLS_cpp_R.cpp')

logit <- function(x) 1/(1+exp(-x))

W_mat <- function(logits) diag(logits*(1-logits))

brd <- function(X, wt)
{
  for (i in 1:ncol(X))
  {
    X[,i] <- X[,i]*wt
  }
  return(X)
}

loglik_logit <- function(X, y, b)
{
  return(sum(dbinom(y, 1, logit(X %*% b), log=T)))
}

#1

IRLS_R <- glm(
  Species ~ .,
  data=iris[51:150,],
  family='binomial'
)

#2

IRLS_conceptual <- function(X, y, init = NULL, tol=1e-6)
{
  if(is.null(init)) init <- rep(0, ncol(X))
  crit = Inf
  iter <- 0
  b <- init
  n_max <- 100
  pred <- logit(X %*% b)
  wt <- pred*(1-pred)
  
  while((crit > tol) & iter < n_max)
  {
    iter <- iter+1
    wk_resp <- (y-pred)/wt
    dif <- coef(lm(wk_resp ~ X - 1, weights=wt))
    crit <- sum(dif^2) / (sum(dif^2)+0.1)
    b <- b + dif
    pred <- logit(X %*% b)
    wt <- pred*(1-pred)
  }
  return(list(beta=as.vector(b), iter=iter))
}

#3

IRLS_with_W <- function(X, y, init = NULL, tol=1e-6)
{
  if(is.null(init)) init <- rep(0, ncol(X))
  crit = Inf
  iter <- 0
  b <- init
  n_max <- 100
  
  while((crit > tol) & iter < n_max)
  {
    pred <- as.vector(logit(X %*% b))
    W <- diag(pred*(1-pred))
    dif <- solve(t(X) %*% W %*% X) %*% t(X) %*% (y-pred)
    crit <- sum(dif^2) / (sum(dif^2)+0.1)
    b <- b+dif
    iter <- iter + 1
  }
  return(list(beta=as.vector(b), iter=iter))
}

#3

IRLS_without_W <- function(X, y, init = NULL, tol=1e-6)
{
  if(is.null(init)) init <- rep(0, ncol(X))
  crit = Inf
  iter <- 0
  b <- init
  n_max <- 100
  
  while((crit > tol) & iter < n_max)
  {
    pred <- as.vector(logit(X %*% b))
    wt <- pred*(1-pred)
    dif <- solve(t(X) %*% brd(X, wt)) %*% t(X) %*% (y-pred)
    crit <- sum(dif^2) / (sum(dif^2)+0.1)
    b <- b + dif
    iter <- iter + 1
  }
  return(list(beta=as.vector(b), iter=iter))
}

#5

IRLS_num_deriv <- function(X, y, init = NULL, tol=1e-6)
{
  if(is.null(init)) init <- rep(0, ncol(X))
  crit = Inf
  iter <- 0
  b <- init
  n_max <- 100
  
  while((crit > tol) & iter < n_max)
  {
    dif <- solve(hessian(loglik_logit, b, X=X, y=y)) %*% grad(loglik_logit, b, X=X, y=y)
    crit <- sum(dif^2)
    b <- b - dif
    iter <- iter + 1
  }
  return(c(as.vector(b), iter))
}

#6
IRLS_cpp(X, y, 1e-06)

X <- cbind(1, as.matrix(iris[51:150, 1:4]))
y <- 1-as.numeric(iris[51:150, 5] == 'versicolor')
iter = 1000

#7
formula <- as.formula('Species ~ Sepal.Length')
speedglm.wfit(y, X, intercept = FALSE, family = binomial(), method = "Chol")

start = Sys.time()
for(i in 1:iter) glm(
  Species ~ .,
  data=iris[51:150,],
  family='binomial'
)
time_R <- Sys.time()-start

start = Sys.time()
for(i in 1:iter) IRLS_conceptual(X, y)
time_concept <- Sys.time()-start

start = Sys.time()
for(i in 1:iter) IRLS_with_W(X, y)
time_with_W <- Sys.time()-start

start = Sys.time()
for(i in 1:iter) IRLS_without_W(X, y)
time_without_W <- Sys.time()-start

start = Sys.time()
for(i in 1:iter) IRLS_num_deriv(X, y)
time_num_deriv <- Sys.time()-start

start = Sys.time()
for(i in 1:iter) IRLS_cpp(X, y, 1e-06)
time_cpp <- Sys.time()-start

start = Sys.time()
for(i in 1:iter) speedglm.wfit(y, X, intercept = FALSE, family = binomial())
time_speedglm <- Sys.time()-start

time_R
time_concept
time_with_W
time_without_W
time_num_deriv
time_cpp
time_speedglm

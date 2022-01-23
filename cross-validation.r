data = read.table('diabetes.txt')
X = data.matrix(data[ ,1:10])
y = data[ ,11]
p = ncol(X)
bet_OLS = c(solve(crossprod(X))%*%t(X)%*%y)
set.seed(1)
# prepare data for K-folds cross validation
K_folds = 5
folds = cut(seq(1,nrow(X)), breaks=K_folds, labels=FALSE)
folds = sample(folds)
candidate_lambda = 2^seq(-10,10,1)
cv_loss = numeric()
for(i in 1:length(candidate_lambda))
{
	lambda = candidate_lambda[i]
	valid_mse = numeric()
	for(k in 1:K_folds)
	{
		idx = which(folds==k)
    		y_valid = y[idx]
    		X_valid = X[idx, ]
    		y_train = y[-idx]
    		X_train = X[-idx, ]
    		#### In HW1, you should replace the following command by an optim function
    		bet_ridge = c(solve(crossprod(X_train)+diag(lambda, p))%*%t(X_train)%*%y_train) 
    		####
    		y_valid_hat = c(X_valid%*%bet_ridge)
    		valid_mse[k] = mean((y_valid-y_valid_hat)^2)	
	}
	cv_loss[i] = mean(valid_mse)
}
lambda = candidate_lambda[which.min(cv_loss)]


# centralize X with dummies
X1 = matrix(runif(1000*10), nrow=1000)
X2 = t(rmultinom(1000, size=1, prob=c(1,1,1)))
X = cbind(1, X1, X2[,1:2])
bet = c(1:ncol(X))
y = c(X%*%bet + rnorm(nrow(X)))
bet_OLS = c(solve(crossprod(X))%*%t(X)%*%y)

# centralize X and remove 'intercept'
X_c = scale(X, center=T, scale=F)
X_c = X_c[ ,-1]
apply(X_c, 2, mean)
y_c = y-mean(y)
bet_OLS_c = c(solve(crossprod(X_c))%*%t(X_c)%*%y_c)
print(bet_OLS_c)
print(bet_OLS[-1])












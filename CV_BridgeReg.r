#### Cross validation for ridge regression.
#### Note that the cross validation is done on training data
#### One should partition the training data into K folds. 
data = read.table('diabetes.txt') # note that we use a different data set 
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

loss = function(feature, response, lambda, par)
{
    Xbet = c(feature %*% par)
    l = sum((response-Xbet)^2) + lambda*sum(par^2) # ridge 
    # or
    # l= sum((response-Xbet)^2) + lambda*sum(abs(par)^q) # Bridge with L_q penalty
    return(l)
}

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
    #bet_ridge = c(solve(crossprod(X_train)+diag(lambda, p))%*%t(X_train)%*%y_train)
    bet_ridge = optim(par=rep(0,p), fn=loss, feature=X_train, response=y_train, lambda=lambda, method='BFGS')$par
    ####
    y_valid_hat = c(X_valid%*%bet_ridge)
    valid_mse[k] = mean((y_valid-y_valid_hat)^2)
}
cv_loss[i] = mean(valid_mse)
}
# select the lambda with the smallest CV loss
lambda = candidate_lambda[which.min(cv_loss)]

####

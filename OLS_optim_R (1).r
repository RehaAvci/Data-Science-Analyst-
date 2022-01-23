X = matrix(rnorm(5000), ncol=5)
X[,1] = 1
bet_true = c(-2:2)
y = c(X%*%bet_true) + rnorm(nrow(X))
bet_OLS = c(solve(crossprod(X))%*%t(X)%*%y)

loss = function(feature, response, par) 
{
    Xbet = c(feature %*% par)
    mse = mean((response-Xbet)^2)
    return(mse)
}

initial_bet = rnorm(5)
result = optim(par=initial_bet, fn=loss, feature=X, response=y, method='BFGS')$par
# BFGS: quasi-Newton method (this is one of the most commonly used methods)



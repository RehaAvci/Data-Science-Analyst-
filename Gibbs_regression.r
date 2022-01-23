# Gibbs sampling
# simulate data
n = 100
x = rnorm(n, sd=2)
alpha_true = -3
bet_true = 2
tau_true = 1
y = alpha_true + x*bet_true + rnorm(n, sd=1/sqrt(tau_true))


S = 1000
Bet = numeric(S)
Alpha = numeric(S)
Tau = numeric(S)
alpha = 0
bet = 0
tau = 0.001
tau_alpha = 0.001
tau_bet = 0.001
a = b = 0.1
for(s in 1:S)
{
	alpha = rnorm(1, 
			mean=tau*sum(y-bet*x)/(tau_alpha+n*tau), 
			sd=1/sqrt(tau_alpha+n*tau))
	Alpha[s] = alpha
	bet = rnorm(1, 
			mean = tau*sum(x*(y-alpha))/(tau_bet + tau*sum(x^2)),
			sd = 1/sqrt(tau_bet+tau*sum(x^2))
			)
	Bet[s] = bet
	tau = rgamma(1, shape=a+n/2, rate=b+0.5*sum((y-alpha-bet*x)^2))
	Tau[s] = tau
}

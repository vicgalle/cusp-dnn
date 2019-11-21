import numpy as np
import torch

from scipy.stats import wishart, multivariate_normal

def gen_data(d, l, n):

    Sigma = wishart.rvs(df=l, scale=np.eye(l))
    x = multivariate_normal.rvs(cov=Sigma, size=n)

    # generate dummy features
    tmp = np.zeros([n, d-l])

    return np.hstack([x, tmp]), Sigma


d = 500
n = 100
l = 5
k = 10

x, Sigma_tr = gen_data(500, 5, 100)

Lambda = 0.1*torch.zeros(d, k, requires_grad = True)
Diag = 0.1*torch.ones(d, requires_grad = True)
Sigma_m = Lambda @ Lambda.T + torch.eye(d)

mvn = torch.distributions.MultivariateNormal(loc=torch.zeros(d), covariance_matrix=Sigma_m)

print(mvn.log_prob(x).shape)

print(x.shape)
print(Sigma)




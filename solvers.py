import numpy as np
import cvxpy as cp
from utils import prior_cov
#---------------------------------------------
#  image inverse problem part
#---------------------------------------------
def get_x_ml(y,T_obs_inv,hypers):
    """
    max p(y|x)
    """
    n,m = hypers['data_dim'],hypers['x_dim']
    G = hypers['G']

    noise_var = 1/T_obs_inv[1,1]
    data = y.flatten()
    # Construct the problem.

    x = cp.Variable(m)
    f = cp.sum_squares(data-G@x)
    objective = cp.Minimize(f)
    p = cp.Problem(objective)

    # Assign a value to gamma and find the optimal x.
    result = p.solve(solver=cp.SCS,verbose=False)
    return (x.value).reshape(-1,1)

def get_x_map(y,T_pr,T_obs_inv,mus,hypers):
    """
    max p(x|y)
    """
    n,m = hypers['data_dim'],hypers['x_dim']
    G = hypers['G']

    mu = mus[1]
    T_pr_inv = np.linalg.inv(T_pr)
    noise_var = 1/T_obs_inv[1,1]
    data = y.flatten()
    # Construct the problem.

    x = cp.Variable(m)
    f = cp.quad_form(data-G@x,T_obs_inv)+cp.quad_form(x-mu,T_pr_inv)
    objective = cp.Minimize(f)
    p = cp.Problem(objective)

    # Assign a value to gamma and find the optimal x.
    result = p.solve(solver=cp.SCS,verbose=False)
    return (x.value).reshape(-1,1)

#---------------------------------------------
#  Log Gaussian Cox model - LGC
#---------------------------------------------
def map_solution(theta,y,hypers,rho=2,verbose=False):
    """
    Parameters
    ----------
    mu C： mean and covriance of prior
       y: data
     rho: weight of prior(在某个测试中，rho取得比较大map结果比较准)

    Returns
    -------
    x_map
    """
    mu,gamma,d = theta
    n,m = hypers['data_dim'],hypers['x_dim']
    C = prior_cov(hypers['distM'], gamma=gamma, d=d)
    C_inv = np.linalg.inv(C)
    # Construct the problem.
    x = cp.Variable(m)
    f = cp.sum(cp.exp(x))-x.T@y+cp.quad_form(x-mu, C_inv)/rho


    objective = cp.Minimize(f)
    p = cp.Problem(objective)

    # Assign a value to gamma and find the optimal x.
    result = p.solve(solver=cp.SCS,verbose=verbose)
    return x.value
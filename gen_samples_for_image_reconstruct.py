import pandas as pd
import cv2
import os
from utils import prior_cov
import numpy as np
from itertools import product
import argparse

"""
x = N(u,C)
y = Gx+\eta 
(1) G=I, \eta=N(0,\Gamma_obs)
(2) G=convolution operator


paras_nums = hypers['mu_nums']*hypers['gamma_nums']*hypers['d_nums']##*hypers['noise_num']
samples_nums = paras_nums*hypers['M_samples_per_para']

Returns  
-------
thetas [samples_nums,para_dim]
    xs [samples_nums,x_dim]          x_dim =hypers['image_size']**2
    ys [samples_nums,data_dim]    data_dim = x_dim
"""
parser = argparse.ArgumentParser(description='give parallel nodes, max=all_nodes*0.5')
parser.add_argument('--parall_nodes', type=int, default=2, help='parallel nodes, need to be less than all_nodes*0.5')
parser.add_argument('--bin_num_total', type=int, default=3, help='split all parameters into bin_num_total group')
parser.add_argument('--bin_num', type=int, default=0, help='simulate data for bin_num th groups parameters, 0<=bin_num<bin_num_total')
args = parser.parse_args()

# %%
if __name__ == "__main__":
    from problem_setting import *
    from pathos.pools import ProcessPool


    def gen_one_para_samples(theta, hypers=hypers):
        """
        x = N(u,C)
        y = Gx+b 

            Parameters
            ----------
                 theta: [mu,gamma,d]
                 ####### [mu,gamma,d,sigma]
                hypers:

            Returns
                thetas, xs, ys:  [M,dim]
            -------

        """
        G = hypers['G']
        M = hypers['M_samples_per_para']
        x_dim = hypers['x_dim']
        # mu, gamma, d = theta
        mu, gamma, d, ratio = theta

        C = prior_cov(hypers['distM'], gamma=gamma, d=d)
        xs = np.random.multivariate_normal([mu] * x_dim, C, M)  # [M,x_dim],M表示每一种参数的样本数目，x_dim是64**2

        # sigmas = hypers['noise_signal_ratio'] * xs.max(axis=1) # # # #只有固定了才能够输出这个函数？？
        sigma_s = ratio * xs.max(axis=1)
        b = np.random.randn(*xs.shape) * sigma_s.reshape(-1, 1)
        ys = xs@G.T + b  # broadcasting    (M,x_dim)+(M,1)
        return np.array([theta] * M), xs, ys

    # range of hypers
    mus = np.linspace(hypers['mu_range'][0], hypers['mu_range'][1], hypers['mu_nums']).round(3)
    gammas = np.linspace(hypers['gamma_range'][0], hypers['gamma_range'][1], hypers['gamma_nums']).round(3)
    ds = np.linspace(hypers['d_range'][0], hypers['d_range'][1], hypers['d_nums']).round(3)
    ratios = np.linspace(hypers['noise_signal_ratio'][0], hypers['noise_signal_ratio'][1], hypers['noise_num']).round(3)
    thetas_all = np.array(list(product(mus, gammas, ds, ratios)))

    # parallel
    pool = ProcessPool(nodes=args.parall_nodes)
    thetas_parts = np.split(thetas_all, args.bin_num_total)
    for _ in range(1):
        # data_file
        data_file = hypers['data_file_prefix'] + f'_{args.bin_num}.npz'
        print(f"simulate samples:{data_file}")
        
        theta_part = thetas_parts[args.bin_num]
        data_all = pool.map(gen_one_para_samples, theta_part)
        #         paras_nums = hypers['mu_nums'] * hypers['gamma_nums'] * hypers['d_nums']
        paras_nums = len(theta_part)
        # thetas = np.empty((paras_nums, hypers['M_samples_per_para'], 3))
        thetas = np.empty((paras_nums, hypers['M_samples_per_para'], 4))
        xs = np.empty((paras_nums, hypers['M_samples_per_para'], hypers['x_dim']))
        ys = np.empty((paras_nums, hypers['M_samples_per_para'], hypers['data_dim']))

        for i, (thetas_i, xs_i, ys_i) in enumerate(data_all):
            thetas[i], xs[i], ys[i] = thetas_i, xs_i, ys_i

        # thetas = thetas.reshape(-1, 3)
        thetas = thetas.reshape(-1, 4)
        xs = xs.reshape(-1, hypers['x_dim'])
        ys = ys.reshape(-1, hypers['data_dim'])

        np.savez(data_file, thetas=thetas, xs=xs, ys=ys)
        print(f"saved samples:{data_file}")
        

# %%
#     from visualize import show_images
#     index_start,index_end = 10,13
#     xs_img = [x.reshape(hypers['image_size'],hypers['image_size']) for x in xs[index_start:index_end,:]]
#     ys_img = [x.reshape(hypers['image_size'],hypers['image_size']) for x in ys[index_start:index_end,:]]
#     show_images(xs_img)
#     show_images(ys_img)






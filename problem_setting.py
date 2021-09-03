# # ---------------------------------------
# #   load basic packages
# # ---------------------------------------
import cv2
import numpy as np
np.set_printoptions(suppress=True)          # suppress=True取消科学计数法
import pandas as pd
pd.set_option('display.width', 1000)
pd.set_option('display.max_columns', None)  # 显示所有列
pd.set_option('display.max_rows', None)       # 显示20行
import matplotlib
# matplotlib.use('Agg')    #服务器使用matplotlib，不会显式图像

import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import cvxpy as cp
from multiprocessing import Pool


#%
import torch
torch.multiprocessing.freeze_support()
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader,random_split
from torchsummary import summary

from utils import samples_file,prior_cov,potential_of_prior_cov,PSF_matrix,blurring_matrix
from visualize import show_images,printdict
import pprint
import sys
from collections import OrderedDict
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# # ---------------------------------------
# #   problem setting
# # ---------------------------------------
problem_type = "deblur"

hypers = OrderedDict(
    {"image_size":64 ,
     "noise_signal_ratio":[0.01,0.1],"noise_num":3,
     "M_samples_per_para": 10000,
#      "mu_range": [0.4,0.6],"mu_nums":20,
#      "gamma_range": [0.1,0.3],"gamma_nums":20,
#      "d_range": [5/64, 12/64],"d_nums":10}
     "mu_range":[0.5,0.5],"mu_nums":1,
     "gamma_range": [0.2,0.2],"gamma_nums":1,
     "d_range":[10/64,10/64],"d_nums":1}
)
# blurring matrix
if problem_type=='deblur':
#     t = 0.0015 #
    hypers['t_in_G'] = 0.04  # more ill-posed 
    hypers['r_in_G'] = 4  
    G_blur = blurring_matrix(hypers['r_in_G'],hypers['t_in_G'],hypers)
    hypers['G'] = G_blur
  


    
# precompute potential function of prior cov
distM = potential_of_prior_cov(hypers['image_size'],hypers['image_size'])
hypers['distM'] = distM


# data_file
data_file_prefix = samples_file(hypers,info=problem_type)
hypers['data_file_prefix'] = data_file_prefix

hypers['data_dim'] = hypers['image_size']**2
hypers['x_dim'] = hypers['image_size']**2



# # ---------------------------------------
# #   print hyperparameters
# # ---------------------------------------
printdict(hypers)
    
print('-'*75)
print(f"number of samples: {hypers['noise_num']*hypers['mu_nums']*hypers['gamma_nums']*hypers['d_nums']}*  {hypers['M_samples_per_para']}  (noise_num*mu_nums*gamma_nums*d_nums*M_samples_per_para)")
print('-'*75)

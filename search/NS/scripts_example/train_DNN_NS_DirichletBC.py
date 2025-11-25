import sys
# update your projecty root path before running
# sys.path.insert(0, '/path/to/nsga-net')
sys.path.append(r'/home/weiz/Astar_Work/NAS/NAS-Unet-PINN5')
sys.path.append(r'/home/weiz/Software/Software/anaconda3/pkgs/mkl-2023.1.0-h6d00ec8_46342/lib')


import os
import numpy as np
import torch
import logging
import argparse
import torch.nn as nn
import torch.utils
# import torchvision.datasets as dset
import torch.backends.cudnn as cudnn
from torch.utils.data.dataset import Dataset

from models.micro_models import NasUnet
from models.unet import UNET, UNET_PINN

import time
from util import utils
from search import micro_encoding
from misc.flops_counter import add_flops_counting_methods

from PDEBench.pdebench.models.fno.utils import FNODatasetSingle, FNODatasetMult
from PDEBench.pdebench.models.unet.utils import UNetDatasetSingle
from PDEBench.pdebench.models.unet.utils_PINN import GetPDEData, GetPDEData_Transient

from PDEBench.pdebench.models.metrics import metric_func, metrics
import pandas as pd
import matplotlib.pyplot as plt

from PDEBench.pdebench.models.unet.unet import UNet2d
from scipy.signal import convolve2d




device = 'cuda'
# device = 'cpu'


ls_all = np.load('/home/weiz/Astar_Work/NAS/NAS-Unet-PINN2/search/NS/data/ls_all.npz')['arr_0']
u_all = np.load('/home/weiz/Astar_Work/NAS/NAS-Unet-PINN2/search/NS/data/u_all.npz')['arr_0']
v_all = np.load('/home/weiz/Astar_Work/NAS/NAS-Unet-PINN2/search/NS/data/v_all.npz')['arr_0']
p_all = np.load('/home/weiz/Astar_Work/NAS/NAS-Unet-PINN2/search/NS/data/p_all.npz')['arr_0']
T_all = np.load('/home/weiz/Astar_Work/NAS/NAS-Unet-PINN2/search/NS/data/T_all.npz')['arr_0']

u_inner = u_all[:, 1:-1, 1:-1]
v_inner = v_all[:, 1:-1, 1:-1]
p_inner = p_all[:, 1:-1, 1:-1]
t_inner = T_all[:, 1:-1, 1:-1]
ls_inner = ls_all[:, 1:-1, 1:-1]


# # get the downsampling data
# # padding to 512*512
# u_inner_s = np.pad(u_inner, ((0, 0), (128, 128), (0, 0)), 'constant', constant_values=0)
# v_inner_s = np.pad(v_inner, ((0, 0), (128, 128), (0, 0)), 'constant', constant_values=0)
# p_inner_s = np.pad(p_inner, ((0, 0), (128, 128), (0, 0)), 'constant', constant_values=0)
# t_inner_s = np.pad(t_inner, ((0, 0), (128, 128), (0, 0)), 'constant', constant_values=1)
# ls_inner_s = np.pad(ls_inner, ((0, 0), (128, 128), (0, 0)), 'constant', constant_values=1)
# u_inner_s[np.isnan(u_inner_s)] = 0
# v_inner_s[np.isnan(v_inner_s)] = 0
# p_inner_s[np.isnan(p_inner_s)] = 0
# t_inner_s[np.isnan(t_inner_s)] = 1

# # down sampling
# # u_inner_s_all = np.zeros((0, 128, 128))
# # v_inner_s_all = np.zeros((0, 128, 128))
# # p_inner_s_all = np.zeros((0, 128, 128))
# # t_inner_s_all = np.zeros((0, 128, 128))
# # ls_inner_s_all = np.zeros((0, 128, 128))
# u_inner_s_all = np.zeros((0, 256, 256))
# v_inner_s_all = np.zeros((0, 256, 256))
# p_inner_s_all = np.zeros((0, 256, 256))
# t_inner_s_all = np.zeros((0, 256, 256))
# ls_inner_s_all = np.zeros((0, 256, 256))
# for i in np.arange(1000):
#     # kernel = np.ones((4, 4))  # 128*128
#     kernel = np.ones((2, 2))  # 256*256

#     # u
#     u_inner_conv2d = convolve2d(u_inner_s[i, :, :], kernel, mode='valid')
#     u_inner_s1 = u_inner_conv2d[::kernel.shape[0], ::kernel.shape[1]] / (kernel.shape[0]*kernel.shape[1])
#     u_inner_s1 = np.expand_dims(u_inner_s1, axis=0)
#     u_inner_s_all = np.concatenate([u_inner_s_all, u_inner_s1], axis=0)

#     # v
#     v_inner_conv2d = convolve2d(v_inner_s[i, :, :], kernel, mode='valid')
#     v_inner_s1 = v_inner_conv2d[::kernel.shape[0], ::kernel.shape[1]] / (kernel.shape[0]*kernel.shape[1])
#     v_inner_s1 = np.expand_dims(v_inner_s1, axis=0)
#     v_inner_s_all = np.concatenate([v_inner_s_all, v_inner_s1], axis=0)

#     # p
#     p_inner_conv2d = convolve2d(p_inner_s[i, :, :], kernel, mode='valid')
#     p_inner_s1 = p_inner_conv2d[::kernel.shape[0], ::kernel.shape[1]] / (kernel.shape[0]*kernel.shape[1])
#     p_inner_s1 = np.expand_dims(p_inner_s1, axis=0)
#     p_inner_s_all = np.concatenate([p_inner_s_all, p_inner_s1], axis=0)

#     # t
#     t_inner_conv2d = convolve2d(t_inner_s[i, :, :], kernel, mode='valid')
#     t_inner_s1 = t_inner_conv2d[::kernel.shape[0], ::kernel.shape[1]] / (kernel.shape[0]*kernel.shape[1])
#     t_inner_s1 = np.expand_dims(t_inner_s1, axis=0)
#     t_inner_s_all = np.concatenate([t_inner_s_all, t_inner_s1], axis=0)

#     # ls
#     ls_inner_conv2d = convolve2d(ls_inner_s[i, :, :], kernel, mode='valid')
#     ls_inner_s1 = ls_inner_conv2d[::kernel.shape[0], ::kernel.shape[1]] / (kernel.shape[0]*kernel.shape[1])
#     ls_inner_s1 = np.expand_dims(ls_inner_s1, axis=0)
#     ls_inner_s_all = np.concatenate([ls_inner_s_all, ls_inner_s1], axis=0)

# 64*64
# u_inner_s = np.load('/home/weiz/Astar_Work/NAS/NAS-Unet-PINN2/search/NS/data/u_inner_s_all_64.npz')['arr_0']
# v_inner_s = np.load('/home/weiz/Astar_Work/NAS/NAS-Unet-PINN2/search/NS/data/v_inner_s_all_64.npz')['arr_0']
# p_inner_s = np.load('/home/weiz/Astar_Work/NAS/NAS-Unet-PINN2/search/NS/data/p_inner_s_all_64.npz')['arr_0']
# t_inner_s = np.load('/home/weiz/Astar_Work/NAS/NAS-Unet-PINN2/search/NS/data/t_inner_s_all_64.npz')['arr_0']
# ls_inner_s = np.load('/home/weiz/Astar_Work/NAS/NAS-Unet-PINN2/search/NS/data/ls_inner_s_all_64.npz')['arr_0']

# 128*128
u_inner_s = np.load('/home/weiz/Astar_Work/NAS/NAS-Unet-PINN2/search/NS/data/u_inner_s_all_128.npz')['arr_0']
v_inner_s = np.load('/home/weiz/Astar_Work/NAS/NAS-Unet-PINN2/search/NS/data/v_inner_s_all_128.npz')['arr_0']
p_inner_s = np.load('/home/weiz/Astar_Work/NAS/NAS-Unet-PINN2/search/NS/data/p_inner_s_all_128.npz')['arr_0']
t_inner_s = np.load('/home/weiz/Astar_Work/NAS/NAS-Unet-PINN2/search/NS/data/t_inner_s_all_128.npz')['arr_0']
ls_inner_s = np.load('/home/weiz/Astar_Work/NAS/NAS-Unet-PINN2/search/NS/data/ls_inner_s_all_128.npz')['arr_0']

# 256*256
# u_inner_s = np.load('/home/weiz/Astar_Work/NAS/NAS-Unet-PINN2/search/NS/data/u_inner_s_all_256.npz')['arr_0']
# v_inner_s = np.load('/home/weiz/Astar_Work/NAS/NAS-Unet-PINN2/search/NS/data/v_inner_s_all_256.npz')['arr_0']
# p_inner_s = np.load('/home/weiz/Astar_Work/NAS/NAS-Unet-PINN2/search/NS/data/p_inner_s_all_256.npz')['arr_0']
# t_inner_s = np.load('/home/weiz/Astar_Work/NAS/NAS-Unet-PINN2/search/NS/data/t_inner_s_all_256.npz')['arr_0']
# ls_inner_s = np.load('/home/weiz/Astar_Work/NAS/NAS-Unet-PINN2/search/NS/data/ls_inner_s_all_256.npz')['arr_0']


# # visualize simulation data
# # label
# fig = plt.figure()
# con_lv = 15
# # t
# t0 = t_inner_s[100, :, :]
# plt.contourf(t0, con_lv, origin='lower', cmap='rainbow', aspect='auto');
# plt.colorbar(); plt.xlabel('x'); plt.ylabel('y')
# plt.show()



# computational boundary
# x_l, x_u, y_l, y_u = 0, 2, 0, 1
x_l, x_u, y_l, y_u = 0, 2, 0, 2
ext = [x_l, x_u, y_l, y_u]


# dx, dy, dt
# n_x, n_y = 512, 256
# n_x, n_y = 64, 64
n_x, n_y = 128, 128
# n_x, n_y = 256, 256
dx = (x_u - x_l) / n_x  #/ 2
dy = (y_u - y_l) / n_y  #/ 2

x = np.linspace(x_l + dx/2, x_u - dx/2, n_x)
y = np.linspace(y_l + dy/2, y_u - dy/2, n_y)
x_grid, y_grid = np.meshgrid(x, y)



# add BC indicator
i_solid = 30.
x_ind_bc = np.zeros((1000, n_y, n_x))
# x_ind_bc[ls_inner > 0] = i_solid
x_ind_bc[ls_inner_s > 0] = i_solid
x_ind_bc_padding = np.pad(x_ind_bc, ((0, 0), (1, 1), (0, 0)), 'constant', constant_values=(i_solid, i_solid))
x_ind_bc_padding = np.pad(x_ind_bc_padding, ((0, 0), (0, 0), (1, 1)), 'constant')
x_ind_bc_C = x_ind_bc.copy()
x_ind_bc_W = x_ind_bc_padding[:, 1:-1, 0:-2]
x_ind_bc_E = x_ind_bc_padding[:, 1:-1, 2:]
x_ind_bc_S = x_ind_bc_padding[:, 0:-2, 1:-1]
x_ind_bc_N = x_ind_bc_padding[:, 2:, 1:-1]


# one solid boundary
_W_1 = (x_grid != x[0]) & (x_grid != x[-1]) & (x_ind_bc_C == 0) & (x_ind_bc_W == i_solid) & (x_ind_bc_E == 0) & (x_ind_bc_S == 0) & (x_ind_bc_N == 0)
_E_1 = (x_grid != x[0]) & (x_grid != x[-1]) & (x_ind_bc_C == 0) & (x_ind_bc_W == 0) & (x_ind_bc_E == i_solid) & (x_ind_bc_S == 0) & (x_ind_bc_N == 0)
_S_1 = (x_grid != x[0]) & (x_grid != x[-1]) & (x_ind_bc_C == 0) & (x_ind_bc_W == 0) & (x_ind_bc_E == 0) & (x_ind_bc_S == i_solid) & (x_ind_bc_N == 0)
_N_1 = (x_grid != x[0]) & (x_grid != x[-1]) & (x_ind_bc_C == 0) & (x_ind_bc_W == 0) & (x_ind_bc_E == 0) & (x_ind_bc_S == 0) & (x_ind_bc_N == i_solid)
# two solid boundaries
_WS_2 = (x_grid != x[0]) & (x_grid != x[-1]) & (x_ind_bc_C == 0) & (x_ind_bc_W == i_solid) & (x_ind_bc_E == 0) & (x_ind_bc_S == i_solid) & (x_ind_bc_N == 0)
_WE_2 = (x_grid != x[0]) & (x_grid != x[-1]) & (x_ind_bc_C == 0) & (x_ind_bc_W == i_solid) & (x_ind_bc_E == i_solid) & (x_ind_bc_S == 0) & (x_ind_bc_N == 0)
_WN_2 = (x_grid != x[0]) & (x_grid != x[-1]) & (x_ind_bc_C == 0) & (x_ind_bc_W == i_solid) & (x_ind_bc_E == 0) & (x_ind_bc_S == 0) & (x_ind_bc_N == i_solid)
_SE_2 = (x_grid != x[0]) & (x_grid != x[-1]) & (x_ind_bc_C == 0) & (x_ind_bc_W == 0) & (x_ind_bc_E == i_solid) & (x_ind_bc_S == i_solid) & (x_ind_bc_N == 0)
_SN_2 = (x_grid != x[0]) & (x_grid != x[-1]) & (x_ind_bc_C == 0) & (x_ind_bc_W == 0) & (x_ind_bc_E == 0) & (x_ind_bc_S == i_solid) & (x_ind_bc_N == i_solid)
_EN_2 = (x_grid != x[0]) & (x_grid != x[-1]) & (x_ind_bc_C == 0) & (x_ind_bc_W == 0) & (x_ind_bc_E == i_solid) & (x_ind_bc_S == 0) & (x_ind_bc_N == i_solid)
# three solid boundaries
_WSE_3 = (x_grid != x[0]) & (x_grid != x[-1]) & (x_ind_bc_C == 0) & (x_ind_bc_W == i_solid) & (x_ind_bc_E == i_solid) & (x_ind_bc_S == i_solid) & (x_ind_bc_N == 0)
_SEN_3 = (x_grid != x[0]) & (x_grid != x[-1]) & (x_ind_bc_C == 0) & (x_ind_bc_W == 0) & (x_ind_bc_E == i_solid) & (x_ind_bc_S == i_solid) & (x_ind_bc_N == i_solid)
_ENW_3 = (x_grid != x[0]) & (x_grid != x[-1]) & (x_ind_bc_C == 0) & (x_ind_bc_W == i_solid) & (x_ind_bc_E == i_solid) & (x_ind_bc_S == 0) & (x_ind_bc_N == i_solid)
_NWS_3 = (x_grid != x[0]) & (x_grid != x[-1]) & (x_ind_bc_C == 0) & (x_ind_bc_W == i_solid) & (x_ind_bc_E == 0) & (x_ind_bc_S == i_solid) & (x_ind_bc_N == i_solid)
# left and right bc
_left = (x_grid == x[0]) & (x_ind_bc_C == 0) & (x_ind_bc_S == 0) & (x_ind_bc_N == 0) & (x_ind_bc_E == 0)
_left_E = (x_grid == x[0]) & (x_ind_bc_C == 0) & (x_ind_bc_S == 0) & (x_ind_bc_N == 0) & (x_ind_bc_E == i_solid)
_left_N = (x_grid == x[0]) & (x_ind_bc_C == 0) & (x_ind_bc_S == 0) & (x_ind_bc_N == i_solid) & (x_ind_bc_E == 0)
_left_NE = (x_grid == x[0]) & (x_ind_bc_C == 0) & (x_ind_bc_S == 0) & (x_ind_bc_N == i_solid) & (x_ind_bc_E == i_solid)
_left_S = (x_grid == x[0]) & (x_ind_bc_C == 0) & (x_ind_bc_S == i_solid) & (x_ind_bc_N == 0) & (x_ind_bc_E == 0)
_left_SE = (x_grid == x[0]) & (x_ind_bc_C == 0) & (x_ind_bc_S == i_solid) & (x_ind_bc_N == 0) & (x_ind_bc_E == i_solid)
_right = (x_grid == x[-1]) & (x_ind_bc_C == 0) & (x_ind_bc_S == 0) & (x_ind_bc_N == 0) & (x_ind_bc_W == 0)
_right_W = (x_grid == x[-1]) & (x_ind_bc_C == 0) & (x_ind_bc_S == 0) & (x_ind_bc_N == 0) & (x_ind_bc_W == i_solid)
_right_N = (x_grid == x[-1]) & (x_ind_bc_C == 0) & (x_ind_bc_S == 0) & (x_ind_bc_N == i_solid) & (x_ind_bc_W == 0)
_right_NW = (x_grid == x[-1]) & (x_ind_bc_C == 0) & (x_ind_bc_S == 0) & (x_ind_bc_N == i_solid) & (x_ind_bc_W == i_solid)
_right_S = (x_grid == x[-1]) & (x_ind_bc_C == 0) & (x_ind_bc_S == i_solid) & (x_ind_bc_N == 0) & (x_ind_bc_W == 0)
_right_SW = (x_grid == x[-1]) & (x_ind_bc_C == 0) & (x_ind_bc_S == i_solid) & (x_ind_bc_N == 0) & (x_ind_bc_W == i_solid)


# one solid boundary
x_ind_bc[_W_1] = 1
x_ind_bc[_S_1] = 2
x_ind_bc[_E_1] = 3
x_ind_bc[_N_1] = 4
# two solid boundaries
x_ind_bc[_WS_2] = 5
x_ind_bc[_WE_2] = 6
x_ind_bc[_WN_2] = 7
x_ind_bc[_SE_2] = 8
x_ind_bc[_SN_2] = 9
x_ind_bc[_EN_2] = 10
# three solid boundaries
x_ind_bc[_WSE_3] = 11
x_ind_bc[_SEN_3] = 12
x_ind_bc[_ENW_3] = 13
x_ind_bc[_NWS_3] = 14
# left and right bc
x_ind_bc[_left] = 15
x_ind_bc[_left_E] = 16
x_ind_bc[_left_N] = 17
x_ind_bc[_left_NE] = 18
x_ind_bc[_left_S] = 19
x_ind_bc[_left_SE] = 20
x_ind_bc[_right] = 21
x_ind_bc[_right_W] = 22
x_ind_bc[_right_N] = 23
x_ind_bc[_right_NW] = 24
x_ind_bc[_right_S] = 25
x_ind_bc[_right_SW] = 26

x_ind_bc = np.expand_dims(x_ind_bc, axis=1)


# x_ind_bc = np.zeros((1000, n_y, n_x))
# x_ind_bc[ls_inner_s <= 0] = 0
# x_ind_bc[ls_inner_s > 0] = 1
# x_ind_bc = np.expand_dims(x_ind_bc, axis=1)






# # visualize simulation data
# # label
# fig = plt.figure()
# con_lv = 15
# # u
# u0 = x_ind_bc[5, 0, :, :]
# plt.contourf(u0, con_lv, origin='lower', cmap='rainbow', extent=ext, aspect='auto');
# plt.colorbar(); plt.xlabel('x'); plt.ylabel('y')
# plt.show()


# # get BC
# u_bc_s_all = np.zeros((0, 128, 128))
# v_bc_s_all = np.zeros((0, 128, 128))
# p_bc_s_all = np.zeros((0, 128, 128))
# for i in np.arange(1000):
#     # u
#     u_bc_s = np.zeros((256, 512))
#     u_bc_s[:, 0] = u_all[i, 1:-1, 0]
#     u_bc_s[:, -1] = u_all[i, 1:-1, -1]
#     u_bc_s = np.pad(u_bc_s, ((128, 128), (0, 0)), 'constant', constant_values=0)

#     # v
#     v_bc_s = np.zeros((256, 512))
#     v_bc_s[:, 0] = v_all[i, 1:-1, 0]
#     v_bc_s[:, -1] = v_all[i, 1:-1, -1]
#     v_bc_s = np.pad(v_bc_s, ((128, 128), (0, 0)), 'constant', constant_values=0)

#     # p
#     p_bc_s = np.zeros((256, 512))
#     p_bc_s[:, 0] = p_all[i, 1:-1, 0]
#     p_bc_s[:, -1] = p_all[i, 1:-1, -1]
#     p_bc_s = np.pad(p_bc_s, ((128, 128), (0, 0)), 'constant', constant_values=0)

#     # down sampling
#     kernel = np.ones((4, 4))  # 128*128
#     # u
#     u_bc_conv2d = convolve2d(u_bc_s, kernel, mode='valid')
#     u_bc_s1 = u_bc_conv2d[::kernel.shape[0], ::kernel.shape[1]] / (kernel.shape[0]*kernel.shape[1])
#     u_bc_s1 = np.expand_dims(u_bc_s1, axis=0)
#     u_bc_s_all = np.concatenate([u_bc_s_all, u_bc_s1], axis=0)

#     # v
#     v_bc_conv2d = convolve2d(v_bc_s, kernel, mode='valid')
#     v_bc_s1 = v_bc_conv2d[::kernel.shape[0], ::kernel.shape[1]] / (kernel.shape[0]*kernel.shape[1])
#     v_bc_s1 = np.expand_dims(v_bc_s1, axis=0)
#     v_bc_s_all = np.concatenate([v_bc_s_all, v_bc_s1], axis=0)

#     # p
#     p_bc_conv2d = convolve2d(p_bc_s, kernel, mode='valid')
#     p_bc_s1 = p_bc_conv2d[::kernel.shape[0], ::kernel.shape[1]] / (kernel.shape[0]*kernel.shape[1])
#     p_bc_s1 = np.expand_dims(p_bc_s1, axis=0)
#     p_bc_s_all = np.concatenate([p_bc_s_all, p_bc_s1], axis=0)

# u_bc_s_all[np.isnan(u_bc_s_all)] = 0
# v_bc_s_all[np.isnan(v_bc_s_all)] = 0
# p_bc_s_all[np.isnan(p_bc_s_all)] = 0


u_bc_s = np.load('/home/weiz/Astar_Work/NAS/NAS-Unet-PINN2/search/NS/data/u_bc_s_all_128.npz')['arr_0']
v_bc_s = np.load('/home/weiz/Astar_Work/NAS/NAS-Unet-PINN2/search/NS/data/v_bc_s_all_128.npz')['arr_0']
p_bc_s = np.load('/home/weiz/Astar_Work/NAS/NAS-Unet-PINN2/search/NS/data/p_bc_s_all_128.npz')['arr_0']


u_inner_s, v_inner_s, p_inner_s = np.expand_dims(u_inner_s, axis=1), np.expand_dims(v_inner_s, axis=1), np.expand_dims(p_inner_s, axis=1)
u_bc_s, v_bc_s, p_bc_s = np.expand_dims(u_bc_s, axis=1), np.expand_dims(v_bc_s, axis=1), np.expand_dims(p_bc_s, axis=1)

# x_train = np.concatenate([x_ind_bc, u_bc_s, v_bc_s, p_bc_s], axis=1)
x_train = x_ind_bc
y_train = np.concatenate([u_inner_s, v_inner_s, p_inner_s], axis=1)

# n_padding = 4
# x_train1 = np.pad(x_train[:, 0:1, :, :], ((0, 0), (0, 0), (n_padding, n_padding), (n_padding, n_padding)), 'constant', constant_values=(40, 40))
# x_train2 = np.pad(x_train[:, 1:, :, :], ((0, 0), (0, 0), (n_padding, n_padding), (n_padding, n_padding)), 'constant', constant_values=(0, 0))
# x_train = np.concatenate([x_train1, x_train2], axis=1)

# y_train = np.pad(y_train, ((0, 0), (0, 0), (n_padding, n_padding), (n_padding, n_padding)), 'constant')
# # y_train = np.pad(y_train, ((0, 0), (0, 0), (0, 0), (n_padding, n_padding)), 'edge')
# # y_train = np.pad(y_train, ((0, 0), (0, 0), (n_padding, n_padding), (0, 0)), 'constant')


# # training data
# data_inputs_train = x_train[0:90]
# data_inputs_valid = x_train[90:100]

# # test data
# data_labels_train = y_train[0:90]
# data_labels_valid = y_train[90:100]


# training data
data_inputs_train = x_train[0:45]
data_inputs_valid = x_train[45:50]

# test data
data_labels_train = y_train[0:45]
data_labels_valid = y_train[45:50]


class TrainDataset(Dataset):
    def __init__(self, data_inputs_train, data_labels_train):
        self.data_inputs_train = data_inputs_train
        self.data_labels_train = data_labels_train

    def __getitem__(self, index):
        data = self.data_inputs_train[index]
        label = self.data_labels_train[index]
        return data, label
    
    def __len__(self):
        return self.data_inputs_train.shape[0]
    
class ValidDataset(Dataset):
    def __init__(self, data_inputs_valid, data_labels_valid):
        self.data_inputs_valid = data_inputs_valid
        self.data_labels_valid = data_labels_valid

    def __getitem__(self, index):
        data = self.data_inputs_valid[index]
        label = self.data_labels_valid[index]
        return data, label
    
    def __len__(self):
        return self.data_inputs_valid.shape[0]
    
train_data = TrainDataset(data_inputs_train, data_labels_train)
valid_data = ValidDataset(data_inputs_valid, data_labels_valid)



def main(x_inputs, genome, epochs, save='Design_1', expr_root='search', seed=0, gpu=0, init_channels=24, depth=5, lamda=1):

    # ---- train logger ----------------- #
    save_pth = os.path.join(expr_root, '{}'.format(save))
    utils.create_exp_dir(save_pth)
    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                        format=log_format, datefmt='%m/%d %I:%M:%S %p')
    # fh = logging.FileHandler(os.path.join(save_pth, 'log.txt'))
    # fh.setFormatter(logging.Formatter(log_format))
    # logging.getLogger().addHandler(fh)

    # ---- parameter values setting ----- #
    PDE_class = 3

    # SGD parameter
    # learning_rate = 0.025
    momentum = 0.9
    # weight_decay = 3e-4

    # Adam parameter
    learning_rate = 5e-4
    weight_decay = 1e-4
    scheduler_step = 10
    scheduler_gamma = 0.9

    batch_size = 1

    genotype = micro_encoding.decode(genome)
    model = NasUnet(nclass=PDE_class, in_channels=1, depth=depth, c=init_channels, genotype=genotype, double_down_channel=True)  # PDE
    # model = UNET(nclass=PDE_class, in_channels=4)
    # model = UNET_PINN(nclass=PDE_class, in_channels=3)
    # model = UNet2d(in_channels=3, out_channels=1)

    # logging.info("Genome = %s", genome)
    logging.info("Architecture = %s", genotype)

    torch.cuda.set_device(gpu)
    cudnn.benchmark = True
    torch.manual_seed(seed)
    cudnn.enabled = True
    torch.cuda.manual_seed(seed)

    n_params = (np.sum(np.prod(v.size()) for v in filter(lambda p: p.requires_grad, model.parameters())) / 1e6)
    model = model.to(device)

    logging.info("param size = %fMB", n_params)

    parameters = filter(lambda p: p.requires_grad, model.parameters())

    optimizer = torch.optim.Adam(
        parameters,
        lr = learning_rate,
        weight_decay = weight_decay
    )
    

    
    train_queue = torch.utils.data.DataLoader(
        train_data, batch_size=batch_size,
        # sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
        pin_memory=True, num_workers=4)

    valid_queue = torch.utils.data.DataLoader(
        valid_data, batch_size=batch_size,
        # sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:num_train]),
        pin_memory=True, num_workers=4)
    

    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, int(epochs))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.7, patience=10, min_lr=1e-6)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step, gamma=scheduler_gamma)

    MSE_min = np.infty  # minimal MSE to save model

    for epoch in range(epochs):
        # start = time.time()
        loss, mse = train(train_queue, model, optimizer)
        # end = time.time()
        # runtime = end - start
        if epoch % 1 == 0:
            # logging.info('epoch %d lr %e', epoch, scheduler.get_lr()[0])
            logging.info('epoch: %d, lr: %e', epoch, optimizer.state_dict()['param_groups'][0]['lr'])  # save lr of ReduceLROnPlateau
            # logging.info('PINN loss: %f, BC loss: %f, PDE loss: %f, MSE: %f, MSEu: %f, MSEv: %f, MSEu_BC: %f, MSEu_PDE: %f, MSEv_BC: %f, MSEv_PDE: %f', loss, bc_loss, pde_loss, mse, mse_u, mse_v, mse_u_bc, mse_u_pde, mse_v_bc, mse_v_pde)
            logging.info('loss: %f, MSE: %f', loss, mse)
        
        valid_MSE = infer(valid_queue, model)
        logging.info('valid MSE: %f', valid_MSE)

        # save model
        # if mse < MSE_min:
        #     MSE_min = mse
        #     torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'MSE': mse}, 'train_model.pt')

        if epoch in np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]):
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'MSE': mse}, 'train_model_epoch_{}.pt'.format(int(epoch)))

        # scheduler.step()
        scheduler.step(loss)
    
    # valid_MSE = infer(valid_queue, model)
    # logging.info('valid MSE: %f', valid_MSE)
        
    # # save model
    # torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'MSE': mse}, 'train_model.pt')
    
 

    # calculate for flops
    model = add_flops_counting_methods(model)
    model.eval()
    model.start_flops_count()
    random_data = torch.randn(1, 1, 128, 128)
    data_test = torch.autograd.Variable(random_data).to(device)
    model(data_test, data_test)
    n_flops = np.round(model.compute_average_flops_cost() / 1e6, 4)
    logging.info('flops = %f', n_flops)

    # save to file
    # os.remove(os.path.join(save_pth, 'log.txt'))
    with open(os.path.join(save_pth, 'log.txt'), "w") as file:
        file.write("x = {}\n".format(x_inputs))
        file.write("Genome = {}\n".format(genome))
        file.write("Architecture = {}\n".format(genotype))
        file.write("param size = {}MB\n".format(n_params))
        file.write("flops = {}MB\n".format(n_flops))
        file.write("valid MSE = {}\n".format(valid_MSE))


    # logging.info("Architecture = %s", genotype))

    return {
        'valid_mse': valid_MSE,
        'params': n_params,
        'flops': n_flops,
    }




# Training
def train(train_queue, net, optimizer):
    net.train()
    loss_all = 0
    mse_all = 0
    flag = 0
    i_d = 0
    for data, label in train_queue:
        # ind = ind + 1
        # print(ind)
        # data: input tensor (first few time steps) [b, x1, ..., xd, t_init, v]
        # label: target tensor [b, x1, ..., xd, t, v]
        # grid: meshgrid [b, x1, ..., xd, dims]
        data = data.float().to(device)
        label = label.float().to(device)

        outputs = net(data)

        # MSE
        x_ind_bc = data[:, 0:1, :, :]
        ind_bc = x_ind_bc.repeat(1, 3, 1, 1)
        _mse = torch.lt(ind_bc, 30)
        uv_gt = torch.masked_select(label[:, 0:3, :, :], _mse)
        uv_pred = torch.masked_select(outputs[:, 0:3, :, :], _mse)
        loss = mse = torch.mean((uv_gt - uv_pred)**2)

        loss_all = loss_all + loss
        mse_all = mse_all + mse

        i_d = i_d + 1

        optimizer.zero_grad()
        loss.backward()  # get the gradient
        optimizer.step()  # update the weights and biases

    loss_all = loss_all/i_d
    mse_all = mse_all/i_d

    return loss_all, mse_all


# infer
def infer(valid_queue, net):
    net.eval()
    valid_mse = 0
    i_d = 0
    mse_all = []
    outputs_all = torch.zeros((0, 3, 128, 128)).to(device)
    # outputs_all = torch.zeros((0, 3, 64, 64)).to(device)
    with torch.no_grad():  # change requires_grad to False
        for data, label in valid_queue:
            # ind = ind + 1
            # print(ind)
            # data: input tensor (first few time steps) [b, x1, ..., xd, t_init, v]
            # label: target tensor [b, x1, ..., xd, t, v]
            # grid: meshgrid [b, x1, ..., xd, dims]
            data = data.float().to(device)
            label = label.float().to(device)

            outputs = net(data)

            # MSE
            x_ind_bc = data[:, 0:1, :, :]
            ind_bc = x_ind_bc.repeat(1, 3, 1, 1)
            _mse = torch.lt(ind_bc, 30)
            uv_gt = torch.masked_select(label[:, 0:3, :, :], _mse)
            uv_pred = torch.masked_select(outputs[:, 0:3, :, :], _mse)
            mse = torch.mean((uv_gt - uv_pred)**2)

            # uv_pred_conv2d = torch.masked_select(outputs_conv2d[:, 0:3, :, :], _mse)
            # mse_conv2d = torch.mean((uv_gt - uv_pred_conv2d)**2)

            valid_mse = valid_mse + mse
            mse_all.append(mse)
            outputs_all = torch.concat([outputs_all, outputs], axis=0)

            i_d = i_d + 1

        valid_mse = valid_mse/i_d

    return valid_mse





if __name__ == "__main__":

    # parser = argparse.ArgumentParser("Multi-Fidelity Surrogate Assisted Differential Evolution for NAS")
    # parser.add_argument('--save', type=str, default='LDC', help='experiment name')
    # parser.add_argument('--seed', type=int, default=0, help='random seed')
    # # arguments for micro search space
    # parser.add_argument('--depth', type=int, default=4, help='depth of a cell')  # CIFAR 10: 4  CHASE_DB1: 3
    # parser.add_argument('--n_ops', type=int, default=6, help='number of operations considered')
    # parser.add_argument('--n_cells', type=int, default=2, help='number of cells to search')
    # # arguments for macro search space
    # parser.add_argument('--n_nodes', type=int, default=3, help='number of nodes per phases')
    # # hyper-parameters for algorithm
    # parser.add_argument('--pop_size', type=int, default=50, help='population size of networks')
    # parser.add_argument('--n_gens', type=int, default=2, help='population size')
    # parser.add_argument('--n_offspring', type=int, default=4, help='number of offspring created per generation')
    # # arguments for back-propagation training during search
    # parser.add_argument('--init_channels', type=int, default=32, help='# of filters for first cell')
    # parser.add_argument('--layers', type=int, default=11, help='equivalent with N = 3')
    # parser.add_argument('--epochs_HF', type=int, default=25, help='# of epochs to HF train during architecture search')
    # parser.add_argument('--epochs_LF', type=int, default=5, help='# of epochs to LF train during architecture search')
    # args = parser.parse_args()
    # args.save = 'search-{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
    # utils.create_exp_dir(args.save)

    # log_format = '%(asctime)s %(message)s'
    # logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    #                     format=log_format, datefmt='%m/%d %I:%M:%S %p')
    # fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
    # fh.setFormatter(logging.Formatter(log_format))
    # logging.getLogger().addHandler(fh)

    start = time.time()


    # x_inputs = np.array([1.95918367e+00, 2.44897959e-01, 1.83673469e-01, 1.34693878e+00, 2.28571429e+00, 3.51020408e+00, 
    #                      1.79591837e+00, 1.67346939e+00, 2.69387755e+00, 1.95918367e+00, 8.16326531e-01, 1.79591837e+00, 1.42857143e+03])  # best
    
    # x_inputs = np.array([1.12280073e-04, 1.99999764e+00, 4.42199728e-05, 2.99976940e+00, 9.87665140e-01, 1.94210178e+00, 
    #                      1.99998456e+00, 1.99998742e+00, 2.46019280e+00, 2.02861862e+00, 5.41231459e-05, 1.17594695e+00, 4.42531405e-02])  # best1
    
    # x_inputs = np.array([1.63265306e-01, 5.71428571e-01, 1.46938776e+00, 2.38775510e+00, 2.61224490e+00, 3.34693878e+00, 
    #                      1.59183673e+00, 1.30612245e+00, 1.46938776e+00, 3.06122449e-01, 6.53061225e-01, 3.91836735e+00, 1.83673469e+03])
    
    # x_inputs = np.array([1.00000000e-10, 1.00000000e-10, 2.59836735e+00, 2.26530612e+00, 3.9, 6.53061225e-01, 
    #                      1.9, 4.08163265e-01, 2.11836735e+00, 1.58448980e+00, 1.00000000e-10, 1.95918367e+00, 1.00000000e+00, 1.00000000e-03])
    
    x_inputs = np.array([7.84961313e-01, 1.9, 4.28102438e-01, 2.9, 3.26424542e-01, 1.00000000e-10, 
                         1.9, 1.00000000e-10, 2.9, 1.70607957e+00, 1.84904965e+00, 1.00000000e-10, 1.00000000e+00, 1.00000000e-04])

    genome = micro_encoding.convert_continuous_variables(x_inputs[0:12])
    init_channels = 32
    depth = 2
    epochs = 1500
    save_dir = 'MF-SADE-BiObj'
    current_nfe = 0
    gpu = 0
    # lamda = 9.98836659e+01
    # lamda = 79.59183878
    # lamda = 1
    # lamda = 8.36734710e+01
    # lamda = 5.71428576e+03
    lamda = 1.00000000e-04
    # lamda = 9.99508923e+03
    performance = main(x_inputs=x_inputs,
                       genome=genome,
                       gpu=gpu,
                       init_channels=init_channels,
                       depth=depth,
                       epochs=epochs,
                       save='arch_{}'.format(0), 
                    #    expr_root=args.save,
                       lamda=x_inputs[-1]) 
    end = time.time()
    runtime = end - start
    

    a

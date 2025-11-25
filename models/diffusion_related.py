# -*- coding: utf-8 -*-
"""
Created on Fri Mar  3 15:15:08 2023

@author: weizh
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch.nn as nn
import torch

device = 'cuda'


def Index_Transformation_01(ls_inner_s):

    ls_inner_s = torch.squeeze(ls_inner_s, 0)

    # add BC indicator
    i_solid = 30.
    # x_ind_bc = np.zeros((1, n_y, n_x))

    mask0 = torch.ones_like(ls_inner_s)
    new_values0 = torch.zeros_like(ls_inner_s)
    new_values0[ls_inner_s > (torch.max(ls_inner_s) + torch.min(ls_inner_s)) / 2] = i_solid
    x_ind_bc = ls_inner_s * (1 - mask0) + new_values0 * mask0
    ls_inner_s.data = x_ind_bc.data

    return ls_inner_s



def Index_Transformation(ls_inner_s):

    # computational boundary
    # x_l, x_u, y_l, y_u = 0, 2, 0, 1
    x_l, x_u, y_l, y_u = 0, 2, 0, 2
    ext = [x_l, x_u, y_l, y_u]

    # dx, dy, dt
    # n_x, n_y = 512, 512
    # n_x, n_y = 64, 64
    n_x, n_y = 128, 128
    # n_x, n_y = 256, 256
    dx = (x_u - x_l) / n_x  #/ 2
    dy = (y_u - y_l) / n_y  #/ 2

    x = torch.linspace(x_l + dx/2, x_u - dx/2, n_x).to(device)
    y = torch.linspace(y_l + dy/2, y_u - dy/2, n_y).to(device)
    y_grid, x_grid = torch.meshgrid(x, y)
    x_grid, y_grid = x_grid.to(device), y_grid.to(device)

    # add BC indicator
    i_solid = 30.
    # x_ind_bc = np.zeros((1, n_y, n_x))


    x_ind_bc1 = ls_inner_s.clone()
    # x_ind_bc[ls_inner_s <= (torch.max(ls_inner_s) + torch.min(ls_inner_s)) / 2] = 0
    # x_ind_bc[ls_inner_s > (torch.max(ls_inner_s) + torch.min(ls_inner_s)) / 2] = i_solid
    x_ind_bc_padding = nn.functional.pad(x_ind_bc1, (0, 0, 1, 1, 0, 0), 'constant', i_solid)
    x_ind_bc_padding = nn.functional.pad(x_ind_bc_padding, (1, 1, 0, 0, 0, 0), 'constant', 0)
    x_ind_bc_padding_padding = nn.functional.pad(x_ind_bc1, (0, 0, 2, 2, 0, 0), 'constant', i_solid)
    x_ind_bc_padding_padding = nn.functional.pad(x_ind_bc_padding_padding, (2, 2, 0, 0, 0, 0), 'constant', 0)
    x_ind_bc_C = ls_inner_s.clone()
    x_ind_bc_W = x_ind_bc_padding[:, 1:-1, 0:-2]
    x_ind_bc_E = x_ind_bc_padding[:, 1:-1, 2:]
    x_ind_bc_S = x_ind_bc_padding[:, 0:-2, 1:-1]
    x_ind_bc_N = x_ind_bc_padding[:, 2:, 1:-1]
    x_ind_bc_WW = x_ind_bc_padding_padding[:, 2:-2, 0:-4]
    x_ind_bc_EE = x_ind_bc_padding_padding[:, 2:-2, 4:]
    x_ind_bc_SS = x_ind_bc_padding_padding[:, 0:-4, 2:-2]
    x_ind_bc_NN = x_ind_bc_padding_padding[:, 4:, 2:-2]
    x_ind_bc_WN = x_ind_bc_padding_padding[:, 3:-1, 1:-3]
    x_ind_bc_WS = x_ind_bc_padding_padding[:, 1:-3, 1:-3]
    x_ind_bc_EN = x_ind_bc_padding_padding[:, 3:-1, 3:-1]
    x_ind_bc_ES = x_ind_bc_padding_padding[:, 1:-3, 3:-1]

    # one solid boundary
    _W_1 = (x_grid != x[0]) & (x_grid != x[-1]) & (x_ind_bc_C < 30) & (x_ind_bc_W == i_solid) & (x_ind_bc_E < 30) & (x_ind_bc_S < 30) & (x_ind_bc_N < 30)
    _E_1 = (x_grid != x[0]) & (x_grid != x[-1]) & (x_ind_bc_C < 30) & (x_ind_bc_W < 30) & (x_ind_bc_E == i_solid) & (x_ind_bc_S < 30) & (x_ind_bc_N < 30)
    _S_1 = (x_grid != x[0]) & (x_grid != x[-1]) & (x_ind_bc_C < 30) & (x_ind_bc_W < 30) & (x_ind_bc_E < 30) & (x_ind_bc_S == i_solid) & (x_ind_bc_N < 30)
    _N_1 = (x_grid != x[0]) & (x_grid != x[-1]) & (x_ind_bc_C < 30) & (x_ind_bc_W < 30) & (x_ind_bc_E < 30) & (x_ind_bc_S < 30) & (x_ind_bc_N == i_solid)
    # two solid boundaries
    _WS_2 = (x_grid != x[0]) & (x_grid != x[-1]) & (x_ind_bc_C < 30) & (x_ind_bc_W == i_solid) & (x_ind_bc_E < 30) & (x_ind_bc_S == i_solid) & (x_ind_bc_N < 30)
    _WE_2 = (x_grid != x[0]) & (x_grid != x[-1]) & (x_ind_bc_C < 30) & (x_ind_bc_W == i_solid) & (x_ind_bc_E == i_solid) & (x_ind_bc_S < 30) & (x_ind_bc_N < 30)
    _WN_2 = (x_grid != x[0]) & (x_grid != x[-1]) & (x_ind_bc_C < 30) & (x_ind_bc_W == i_solid) & (x_ind_bc_E < 30) & (x_ind_bc_S < 30) & (x_ind_bc_N == i_solid)
    _SE_2 = (x_grid != x[0]) & (x_grid != x[-1]) & (x_ind_bc_C < 30) & (x_ind_bc_W < 30) & (x_ind_bc_E == i_solid) & (x_ind_bc_S == i_solid) & (x_ind_bc_N < 30)
    _SN_2 = (x_grid != x[0]) & (x_grid != x[-1]) & (x_ind_bc_C < 30) & (x_ind_bc_W < 30) & (x_ind_bc_E < 30) & (x_ind_bc_S == i_solid) & (x_ind_bc_N == i_solid)
    _EN_2 = (x_grid != x[0]) & (x_grid != x[-1]) & (x_ind_bc_C < 30) & (x_ind_bc_W < 30) & (x_ind_bc_E == i_solid) & (x_ind_bc_S < 30) & (x_ind_bc_N == i_solid)
    # three solid boundaries
    _WSE_3 = (x_grid != x[0]) & (x_grid != x[-1]) & (x_ind_bc_C < 30) & (x_ind_bc_W == i_solid) & (x_ind_bc_E == i_solid) & (x_ind_bc_S == i_solid) & (x_ind_bc_N < 30)
    _SEN_3 = (x_grid != x[0]) & (x_grid != x[-1]) & (x_ind_bc_C < 30) & (x_ind_bc_W < 30) & (x_ind_bc_E == i_solid) & (x_ind_bc_S == i_solid) & (x_ind_bc_N == i_solid)
    _ENW_3 = (x_grid != x[0]) & (x_grid != x[-1]) & (x_ind_bc_C < 30) & (x_ind_bc_W == i_solid) & (x_ind_bc_E == i_solid) & (x_ind_bc_S < 30) & (x_ind_bc_N == i_solid)
    _NWS_3 = (x_grid != x[0]) & (x_grid != x[-1]) & (x_ind_bc_C < 30) & (x_ind_bc_W == i_solid) & (x_ind_bc_E < 30) & (x_ind_bc_S == i_solid) & (x_ind_bc_N == i_solid)
    # left and right bc
    _left = (x_grid == x[0]) & (x_ind_bc_C < 30) & (x_ind_bc_S < 30) & (x_ind_bc_N < 30) & (x_ind_bc_E < 30)
    _left_E = (x_grid == x[0]) & (x_ind_bc_C < 30) & (x_ind_bc_S < 30) & (x_ind_bc_N < 30) & (x_ind_bc_E == i_solid)
    _left_N = (x_grid == x[0]) & (x_ind_bc_C < 30) & (x_ind_bc_S < 30) & (x_ind_bc_N == i_solid) & (x_ind_bc_E < 30)
    _left_NE = (x_grid == x[0]) & (x_ind_bc_C < 30) & (x_ind_bc_S < 30) & (x_ind_bc_N == i_solid) & (x_ind_bc_E == i_solid)
    _left_S = (x_grid == x[0]) & (x_ind_bc_C < 30) & (x_ind_bc_S == i_solid) & (x_ind_bc_N < 30) & (x_ind_bc_E < 30)
    _left_SE = (x_grid == x[0]) & (x_ind_bc_C < 30) & (x_ind_bc_S == i_solid) & (x_ind_bc_N < 30) & (x_ind_bc_E == i_solid)
    _right = (x_grid == x[-1]) & (x_ind_bc_C < 30) & (x_ind_bc_S < 30) & (x_ind_bc_N < 30) & (x_ind_bc_W < 30)
    _right_W = (x_grid == x[-1]) & (x_ind_bc_C < 30) & (x_ind_bc_S < 30) & (x_ind_bc_N < 30) & (x_ind_bc_W == i_solid)
    _right_N = (x_grid == x[-1]) & (x_ind_bc_C < 30) & (x_ind_bc_S < 30) & (x_ind_bc_N == i_solid) & (x_ind_bc_W < 30)
    _right_NW = (x_grid == x[-1]) & (x_ind_bc_C < 30) & (x_ind_bc_S < 30) & (x_ind_bc_N == i_solid) & (x_ind_bc_W == i_solid)
    _right_S = (x_grid == x[-1]) & (x_ind_bc_C < 30) & (x_ind_bc_S == i_solid) & (x_ind_bc_N < 30) & (x_ind_bc_W < 30)
    _right_SW = (x_grid == x[-1]) & (x_ind_bc_C < 30) & (x_ind_bc_S == i_solid) & (x_ind_bc_N < 30) & (x_ind_bc_W == i_solid)
    # impossible conditions
    _left_down_NE = (x_grid == x[0]) & (y_grid == y[0]) & (x_ind_bc_C < 30) & (x_ind_bc_N == i_solid) & (x_ind_bc_E == i_solid)
    _left_up_SE = (x_grid == x[0]) & (y_grid == y[-1]) & (x_ind_bc_C < 30) & (x_ind_bc_S == i_solid) & (x_ind_bc_E == i_solid)
    _left_NES = (x_grid == x[0]) & (x_ind_bc_C < 30) & (x_ind_bc_N == i_solid) & (x_ind_bc_E == i_solid) & (x_ind_bc_S == i_solid)
    _right_down_NW = (x_grid == x[-1]) & (y_grid == y[0]) & (x_ind_bc_C < 30) & (x_ind_bc_N == i_solid) & (x_ind_bc_W == i_solid)
    _right_up_SW = (x_grid == x[-1]) & (y_grid == y[-1]) & (x_ind_bc_C < 30) & (x_ind_bc_S == i_solid) & (x_ind_bc_W == i_solid)
    _right_NWS = (x_grid == x[-1]) & (x_ind_bc_C < 30) & (x_ind_bc_N == i_solid) & (x_ind_bc_W == i_solid) & (x_ind_bc_S == i_solid)
    _down_NEW = (y_grid == y[0]) & (x_ind_bc_C < 30) & (x_ind_bc_N == i_solid) & (x_ind_bc_E == i_solid) & (x_ind_bc_W == i_solid)
    _up_SEW = (y_grid == y[-1]) & (x_ind_bc_C < 30) & (x_ind_bc_S == i_solid) & (x_ind_bc_E == i_solid) & (x_ind_bc_W == i_solid)
    _NSEW = (x_ind_bc_C < 30) & (x_ind_bc_N == i_solid) & (x_ind_bc_S == i_solid) & (x_ind_bc_E == i_solid) & (x_ind_bc_W == i_solid)
    _left_NS = (x_grid == x[0]) & (x_ind_bc_C < 30) & (x_ind_bc_N == i_solid) & (x_ind_bc_S == i_solid)
    _right_NS = (x_grid == x[-1]) & (x_ind_bc_C < 30) & (x_ind_bc_N == i_solid) & (x_ind_bc_S == i_solid)
    _NSW_EE = ((x_ind_bc_C < 30) & (x_ind_bc_N == i_solid) & (x_ind_bc_S == i_solid) & (x_ind_bc_W == i_solid) & (x_ind_bc_E < 30) & (x_ind_bc_EE == i_solid) & (x_ind_bc_EN == i_solid) & (x_ind_bc_ES == i_solid)) | \
        ((x_ind_bc_C < 30) & (x_ind_bc_N == i_solid) & (x_ind_bc_S == i_solid) & (x_ind_bc_E == i_solid) & (x_ind_bc_W < 30) & (x_ind_bc_WW == i_solid) & (x_ind_bc_WN == i_solid) & (x_ind_bc_WS == i_solid))
    _SWE_NN = ((x_ind_bc_C < 30) & (x_ind_bc_S == i_solid) & (x_ind_bc_W == i_solid) & (x_ind_bc_E == i_solid) & (x_ind_bc_N < 30) & (x_ind_bc_NN == i_solid) & (x_ind_bc_WN == i_solid) & (x_ind_bc_EN == i_solid)) | \
        ((x_ind_bc_C < 30) & (x_ind_bc_N == i_solid) & (x_ind_bc_W == i_solid) & (x_ind_bc_E == i_solid) & (x_ind_bc_S < 30) & (x_ind_bc_SS == i_solid) & (x_ind_bc_WS == i_solid) & (x_ind_bc_ES == i_solid))
    # solids to fluids
    _fluids_NSEW_C = (x_ind_bc_C == i_solid) & (x_ind_bc_N < 30) & (x_ind_bc_S < 30) & (x_ind_bc_E < 30) & (x_ind_bc_W < 30)
    _fluids_NSW_EE_C = ((x_ind_bc_C == i_solid) & (x_ind_bc_N < 30) & (x_ind_bc_S < 30) & (x_ind_bc_W < 30) & (x_ind_bc_E == i_solid) & (x_ind_bc_EE < 30) & (x_ind_bc_EN < 30) & (x_ind_bc_ES < 30)) | \
        ((x_ind_bc_C == i_solid) & (x_ind_bc_N < 30) & (x_ind_bc_S < 30) & (x_ind_bc_E < 30) & (x_ind_bc_W == i_solid) & (x_ind_bc_WW < 30) & (x_ind_bc_WN < 30) & (x_ind_bc_WS < 30))
    _fluids_SWE_NN_C = ((x_ind_bc_C == i_solid) & (x_ind_bc_S < 30) & (x_ind_bc_W < 30) & (x_ind_bc_E < 30) & (x_ind_bc_N == i_solid) & (x_ind_bc_NN < 30) & (x_ind_bc_WN < 30) & (x_ind_bc_EN < 30)) | \
        ((x_ind_bc_C == i_solid) & (x_ind_bc_N < 30) & (x_ind_bc_W < 30) & (x_ind_bc_E < 30) & (x_ind_bc_S == i_solid) & (x_ind_bc_SS < 30) & (x_ind_bc_WS < 30) & (x_ind_bc_ES < 30))


    mask = torch.ones_like(ls_inner_s)
    new_values = ls_inner_s.clone()
    # one solid boundary
    new_values[_W_1] = 1
    new_values[_S_1] = 2
    new_values[_E_1] = 3
    new_values[_N_1] = 4
    # two solid boundaries
    new_values[_WS_2] = 5
    new_values[_WE_2] = 6
    new_values[_WN_2] = 7
    new_values[_SE_2] = 8
    new_values[_SN_2] = 9
    new_values[_EN_2] = 10
    # three solid boundaries
    new_values[_WSE_3] = 11
    new_values[_SEN_3] = 12
    new_values[_ENW_3] = 13
    new_values[_NWS_3] = 14
    # left and right bc
    new_values[_left] = 15
    new_values[_left_E] = 16
    new_values[_left_N] = 17
    new_values[_left_NE] = 18
    new_values[_left_S] = 19
    new_values[_left_SE] = 20
    new_values[_right] = 21
    new_values[_right_W] = 22
    new_values[_right_N] = 23
    new_values[_right_NW] = 24
    new_values[_right_S] = 25
    new_values[_right_SW] = 26
    # impossible conditions
    new_values[_left_down_NE] = i_solid
    new_values[_left_up_SE] = i_solid
    new_values[_left_NES] = i_solid
    new_values[_right_down_NW] = i_solid
    new_values[_right_up_SW] = i_solid
    new_values[_right_NWS] = i_solid
    new_values[_down_NEW] = i_solid
    new_values[_up_SEW] = i_solid
    new_values[_NSEW] = i_solid
    new_values[_left_NS] = i_solid
    new_values[_right_NS] = i_solid
    new_values[_NSW_EE] = i_solid
    new_values[_SWE_NN] = i_solid
    # solids to fluids
    # new_values[_fluids_NSEW_C] = 0
    # new_values[_fluids_NSW_EE_C] = 0
    # new_values[_fluids_SWE_NN_C] = 0


    x_ind_bc = ls_inner_s * (1 - mask) + new_values * mask
    x_ind_bc = torch.unsqueeze(x_ind_bc, axis=1)
    ls_inner_s.data = x_ind_bc.data

    return ls_inner_s





def Identify_Bubbles(ls_inner_s):

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

    x = torch.linspace(x_l + dx/2, x_u - dx/2, n_x).to(device)
    y = torch.linspace(y_l + dy/2, y_u - dy/2, n_y).to(device)
    y_grid, x_grid = torch.meshgrid(x, y)
    x_grid, y_grid = x_grid.to(device), y_grid.to(device)

    # add BC indicator
    i_solid = 30.
    # x_ind_bc = np.zeros((1, n_y, n_x))


    x_ind_bc1 = ls_inner_s.clone()
    # x_ind_bc[ls_inner_s <= (torch.max(ls_inner_s) + torch.min(ls_inner_s)) / 2] = 0
    # x_ind_bc[ls_inner_s > (torch.max(ls_inner_s) + torch.min(ls_inner_s)) / 2] = i_solid
    x_ind_bc_padding = nn.functional.pad(x_ind_bc1, (0, 0, 1, 1, 0, 0), 'constant', i_solid)
    x_ind_bc_padding = nn.functional.pad(x_ind_bc_padding, (1, 1, 0, 0, 0, 0), 'constant', 0)
    x_ind_bc_padding_padding = nn.functional.pad(x_ind_bc1, (0, 0, 2, 2, 0, 0), 'constant', i_solid)
    x_ind_bc_padding_padding = nn.functional.pad(x_ind_bc_padding_padding, (2, 2, 0, 0, 0, 0), 'constant', 0)
    x_ind_bc_C = ls_inner_s.clone()
    x_ind_bc_W = x_ind_bc_padding[:, 1:-1, 0:-2]
    x_ind_bc_E = x_ind_bc_padding[:, 1:-1, 2:]
    x_ind_bc_S = x_ind_bc_padding[:, 0:-2, 1:-1]
    x_ind_bc_N = x_ind_bc_padding[:, 2:, 1:-1]
    x_ind_bc_WW = x_ind_bc_padding_padding[:, 2:-2, 0:-4]
    x_ind_bc_EE = x_ind_bc_padding_padding[:, 2:-2, 4:]
    x_ind_bc_SS = x_ind_bc_padding_padding[:, 0:-4, 2:-2]
    x_ind_bc_NN = x_ind_bc_padding_padding[:, 4:, 2:-2]
    x_ind_bc_WN = x_ind_bc_padding_padding[:, 3:-1, 1:-3]
    x_ind_bc_WS = x_ind_bc_padding_padding[:, 1:-3, 1:-3]
    x_ind_bc_EN = x_ind_bc_padding_padding[:, 3:-1, 3:-1]
    x_ind_bc_ES = x_ind_bc_padding_padding[:, 1:-3, 3:-1]


    # solids to fluids
    _fluids_NSEW_C = (x_ind_bc_C == i_solid) & (x_ind_bc_N < 30) & (x_ind_bc_S < 30) & (x_ind_bc_E < 30) & (x_ind_bc_W < 30)
    _fluids_NSW_EE_C = ((x_ind_bc_C == i_solid) & (x_ind_bc_N < 30) & (x_ind_bc_S < 30) & (x_ind_bc_W < 30) & (x_ind_bc_E == i_solid) & (x_ind_bc_EE < 30) & (x_ind_bc_EN < 30) & (x_ind_bc_ES < 30)) | \
        ((x_ind_bc_C == i_solid) & (x_ind_bc_N < 30) & (x_ind_bc_S < 30) & (x_ind_bc_E < 30) & (x_ind_bc_W == i_solid) & (x_ind_bc_WW < 30) & (x_ind_bc_WN < 30) & (x_ind_bc_WS < 30))
    _fluids_SWE_NN_C = ((x_ind_bc_C == i_solid) & (x_ind_bc_S < 30) & (x_ind_bc_W < 30) & (x_ind_bc_E < 30) & (x_ind_bc_N == i_solid) & (x_ind_bc_NN < 30) & (x_ind_bc_WN < 30) & (x_ind_bc_EN < 30)) | \
        ((x_ind_bc_C == i_solid) & (x_ind_bc_N < 30) & (x_ind_bc_W < 30) & (x_ind_bc_E < 30) & (x_ind_bc_S == i_solid) & (x_ind_bc_SS < 30) & (x_ind_bc_WS < 30) & (x_ind_bc_ES < 30))


    x_ind_bc = torch.ones_like(ls_inner_s)
    # solids to fluids
    x_ind_bc[_fluids_NSEW_C] = 0.
    x_ind_bc[_fluids_NSW_EE_C] = 0.
    x_ind_bc[_fluids_SWE_NN_C] = 0.

    x_ind_bc = torch.unsqueeze(x_ind_bc, axis=1)

    return x_ind_bc






def Index_Transformation1(ls_inner_s):

    # ls_inner_s = ls_inner_s.detach().cpu().numpy()
    ls_inner_s = ls_inner_s.reshape(1, 128, 128)

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
    x_ind_bc = np.zeros((1, n_y, n_x))
    # x_ind_bc = ls_inner_s.copy()
    # x_ind_bc[ls_inner_s > 0] = i_solid
    x_ind_bc[ls_inner_s > (np.max(ls_inner_s) + np.min(ls_inner_s)) / 2] = i_solid
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
    # impossible conditions
    _left_down_NE = (x_grid == x[0]) & (y_grid == y[0]) & (x_ind_bc_C == 0) & (x_ind_bc_N == i_solid) & (x_ind_bc_E == i_solid)
    _left_up_SE = (x_grid == x[0]) & (y_grid == y[-1]) & (x_ind_bc_C == 0) & (x_ind_bc_S == i_solid) & (x_ind_bc_E == i_solid)
    _left_NES = (x_grid == x[0]) & (x_ind_bc_C == 0) & (x_ind_bc_N == i_solid) & (x_ind_bc_E == i_solid) & (x_ind_bc_S == i_solid)
    _right_down_NW = (x_grid == x[-1]) & (y_grid == y[0]) & (x_ind_bc_C == 0) & (x_ind_bc_N == i_solid) & (x_ind_bc_W == i_solid)
    _right_up_SW = (x_grid == x[-1]) & (y_grid == y[-1]) & (x_ind_bc_C == 0) & (x_ind_bc_S == i_solid) & (x_ind_bc_W == i_solid)
    _right_NWS = (x_grid == x[-1]) & (x_ind_bc_C == 0) & (x_ind_bc_N == i_solid) & (x_ind_bc_W == i_solid) & (x_ind_bc_S == i_solid)
    _down_NEW = (y_grid == y[0]) & (x_ind_bc_C == 0) & (x_ind_bc_N == i_solid) & (x_ind_bc_E == i_solid) & (x_ind_bc_W == i_solid)
    _up_SEW = (y_grid == y[-1]) & (x_ind_bc_C == 0) & (x_ind_bc_S == i_solid) & (x_ind_bc_E == i_solid) & (x_ind_bc_W == i_solid)
    _NSEW = (x_ind_bc_C == 0) & (x_ind_bc_N == i_solid) & (x_ind_bc_S == i_solid) & (x_ind_bc_E == i_solid) & (x_ind_bc_W == i_solid)
    _left_NS = (x_grid == x[0]) & (x_ind_bc_C == 0) & (x_ind_bc_N == i_solid) & (x_ind_bc_S == i_solid)
    _right_NS = (x_grid == x[-1]) & (x_ind_bc_C == 0) & (x_ind_bc_N == i_solid) & (x_ind_bc_S == i_solid)

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
    # impossible conditions
    x_ind_bc[_left_down_NE] = 31
    x_ind_bc[_left_up_SE] = 32
    x_ind_bc[_left_NES] = 33
    x_ind_bc[_right_down_NW] = 34
    x_ind_bc[_right_up_SW] = 35
    x_ind_bc[_right_NWS] = 36
    x_ind_bc[_down_NEW] = 37
    x_ind_bc[_up_SEW] = 38
    x_ind_bc[_NSEW] = 39
    x_ind_bc[_left_NS] = 40
    x_ind_bc[_right_NS] = 41

    x_ind_bc = np.expand_dims(x_ind_bc, axis=1)

    return x_ind_bc





if __name__ == "__main__":

    a = np.array([[1, 2, 3], [4, 5, 6]])
    b = np.array([[1, 1, 0], [0, 0, 1]])
    c = np.array([[11, 12, 13], [14, 15, 16]])

    a[b == 0] = c[b == 0]


    file = np.load('/home/weiz/Astar_Work/NAS/NAS-Unet-PINN5/search/NS/scripts_example/PI_denoising_number_10_step_18.npz')
    x_in_PI_all, x_in_PI, delta_p_PI_all, delta_p_PI = file['arr_0'], file['arr_1'], file['arr_2'], file['arr_3']
    data = torch.tensor(x_in_PI)
    # data = torch.nn.functional.interpolate(data, size=(64, 128), mode='bilinear', align_corners=True)
    # data = torch.nn.functional.pad(data, (0, 0, 32, 32, 0, 0, 0, 0), 'constant', torch.max(data).detach().cpu().float())
    data = torch.sigmoid(data)
    data = Index_Transformation_01(data)
    data = Identify_Bubbles(data)




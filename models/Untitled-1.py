import os
import numpy as np
import torch
import logging
import argparse
import torch.nn as nn
import torch.utils
# import torchvision.datasets as dset
import torch.backends.cudnn as cudnn

from scipy.signal import convolve2d
from scipy.ndimage import zoom
import scipy.stats as stats

from diffusion_related import Index_Transformation, Index_Transformation_01, Index_Transformation1
import matplotlib.pyplot as plt
import matplotlib



# data_load_PI = np.loadtxt('/home/weiz/Astar_Work/NAS/NAS-Unet-PINN5/search/NS/scripts_example/PI_results/CFD_inputs/PI_0_seed_0.dat', delimiter=',')
data_load_PI = np.loadtxt('/home/weiz/Astar_Work/NAS/NAS-Unet-PINN5/models/input_4.dat', delimiter=',')
# data_load_uncon = np.loadtxt('/home/weiz/Astar_Work/NAS/NAS-Unet-PINN5/search/NS/scripts_example/PI_results/CFD_inputs/Uncon_0_seed_0.dat', delimiter=',')
data_load_uncon = np.loadtxt('/home/weiz/Astar_Work/NAS/NAS-Unet-PINN5/search/NS/scripts_example/PI_results/CFD_inputs/5cells_same/pipe_3/Uncon_0_seed_22_b.dat', delimiter=',')
ls_PI = data_load_PI[:, 2].reshape(256, 512)
ls_uncon = data_load_uncon[:, 2].reshape(256, 512)
fig = plt.figure(figsize=(20, 4.5))
x_l, x_u, y_l, y_u = 0, 2, 0, 1
ext = [x_l, x_u, y_l, y_u]
con_lv = 15
ax1 = fig.add_subplot(1,2,1)
plt.contourf(ls_PI, con_lv, origin='lower', cmap='rainbow', extent=ext, aspect='auto');
plt.colorbar(); plt.xlabel('x'); plt.ylabel('y')
ax1 = fig.add_subplot(1,2,2)
plt.contourf(ls_uncon, con_lv, origin='lower', cmap='rainbow', extent=ext, aspect='auto');
plt.colorbar(); plt.xlabel('x'); plt.ylabel('y')
plt.show()







# # ls_all = np.load('/home/weiz/Astar_Work/NAS/NAS-Unet-PINN2/search/NS/data/ls_all.npz')['arr_0']
# # ls_inner = ls_all[:, 1:-1, 1:-1][45]
# # data_PI = ls_inner
# # data_PI[data_PI > 0] = 1
# # data_PI[data_PI <= 0] = -1


# x_in_PI = np.load('/home/weiz/Astar_Work/NAS/NAS-Unet-PINN2/search/NS/out-of-distribution/4pipes_4.npy')
# x_in_PI = torch.tensor(x_in_PI)
# data_PI = torch.nn.functional.interpolate(x_in_PI, size=(256, 512), mode='bilinear', align_corners=True)
# data_PI = torch.sigmoid(data_PI)
# data_PI = Index_Transformation_01(data_PI)
# data_PI[data_PI == 30] = 1
# data_PI[data_PI == 0] = -1
# data_PI = data_PI.detach().cpu().numpy()[0, :, :]



# # fig = plt.figure(figsize=(20, 4.5))
# # x_l, x_u, y_l, y_u = 0, 2, 0, 1
# # ext = [x_l, x_u, y_l, y_u]
# # con_lv = 15
# # norm = matplotlib.colors.Normalize(vmin=-1, vmax=1)
# # # u
# # plt.figure(1)
# # ax1 = fig.add_subplot(1,2,1)
# # plt.contourf(data_PI, con_lv, origin='lower', cmap='rainbow', extent=ext, aspect='auto');
# # plt.colorbar(); plt.xlabel('x'); plt.ylabel('y')
# # ax1 = fig.add_subplot(1,2,2)
# # plt.contourf(data_PI, con_lv, origin='lower', cmap='rainbow', extent=ext, aspect='auto');
# # plt.colorbar(); plt.xlabel('x'); plt.ylabel('y')
# # # plt.show()


# grid_x = np.linspace(2/(512*2), 2 - 2/(512*2), num=512)
# grid_y = np.linspace(1/(256*2), 1 - 1/(256*2), num=256)
# grid_x, grid_y = np.meshgrid(grid_x, grid_y)
# labels_PI = np.hstack([grid_x.reshape(-1, 1), grid_y.reshape(-1, 1), data_PI.reshape(-1, 1)])
# labels_PI = np.asarray(labels_PI)
# np.savetxt('input_4.dat', labels_PI, delimiter=',', fmt='%.18f')








# file = np.load('/home/weiz/Astar_Work/NAS/NAS-Unet-PINN6/search/NS/scripts_example/PI_denoising_number_22_step_0.npz')
# x_in_PI_all, x_in_PI, delta_p_PI_all, delta_p_PI = file['arr_0'], file['arr_1'], file['arr_2'], file['arr_3']

file = np.load('/home/weiz/Astar_Work/NAS/NAS-Unet-PINN5/search/NS/scripts_example/diffusion_denoising_results/5cells_same/pipe_3/diffusion_denoising_number_22_step_39.npz')
x_in_PI = file['arr_0']
device = 'cuda'
x_in_PI = torch.tensor(x_in_PI).to(device)
data_PI = torch.nn.functional.interpolate(x_in_PI, size=(256, 512), mode='bilinear', align_corners=True)
# data_PI = torch.nn.functional.interpolate(x_in_PI, size=(256, 512), mode='nearest', align_corners=None)
data_PI = torch.nn.functional.pad(data_PI, (0, 0, 128, 128, 0, 0, 0, 0), 'constant', torch.max(data_PI).detach().cpu().float())
data_PI = torch.sigmoid(data_PI)
data_PI = Index_Transformation_01(data_PI)
data_PI = Index_Transformation(data_PI)
data_PI[data_PI >= 30] = 30
data_PI[data_PI < 30] = 0

data_PI = data_PI.detach().cpu().numpy()

# fig = plt.figure(figsize=(8, 4.5))
# x_l, x_u, y_l, y_u = 0, 2, 0, 2
# ext = [x_l, x_u, y_l, y_u]
# con_lv = 15
# plt.contourf(data_PI, con_lv, origin='lower', cmap='rainbow', extent=ext, aspect='auto');
# plt.show()
np.savetxt('test.dat', data_PI[0, 0, 128:384, :], fmt='%2d')

data_PI = np.loadtxt('/home/weiz/Astar_Work/NAS/NAS-Unet-PINN5/search/NS/scripts_example/PI_results/CFD_inputs/5cells_same/pipe_3/test_PI_39_25.dat')
data_PI[data_PI == 30] = 1
data_PI[data_PI == 0] = -1
grid_x = np.linspace(2/(512*2), 2 - 2/(512*2), num=512)
grid_y = np.linspace(1/(256*2), 1 - 1/(256*2), num=256)
grid_x, grid_y = np.meshgrid(grid_x, grid_y)
labels_PI = np.hstack([grid_x.reshape(-1, 1), grid_y.reshape(-1, 1), data_PI.reshape(-1, 1)])
labels_PI = np.asarray(labels_PI)
np.savetxt(f'PI_39_25_seed_22_b.dat', labels_PI, delimiter=',', fmt='%.18f')






# # data_load_PI = np.loadtxt('/home/weiz/Astar_Work/NAS/NAS-Unet-PINN5/search/NS/scripts_example/PI_results/CFD_inputs/PI_0_seed_0.dat', delimiter=',')
# data_load_PI = np.loadtxt('/home/weiz/Astar_Work/NAS/NAS-Unet-PINN5/search/NS/scripts_example/PI_results/CFD_inputs/5cells_same/pipe_3/PI_0_seed_22_b.dat', delimiter=',')
# # data_load_uncon = np.loadtxt('/home/weiz/Astar_Work/NAS/NAS-Unet-PINN5/search/NS/scripts_example/PI_results/CFD_inputs/Uncon_0_seed_0.dat', delimiter=',')
# data_load_uncon = np.loadtxt('/home/weiz/Astar_Work/NAS/NAS-Unet-PINN5/search/NS/scripts_example/PI_results/CFD_inputs/5cells_same/pipe_3/Uncon_0_seed_22_b.dat', delimiter=',')
# ls_PI = data_load_PI[:, 2].reshape(256, 512)
# ls_uncon = data_load_uncon[:, 2].reshape(256, 512)
# fig = plt.figure(figsize=(20, 4.5))
# x_l, x_u, y_l, y_u = 0, 2, 0, 1
# ext = [x_l, x_u, y_l, y_u]
# con_lv = 15
# ax1 = fig.add_subplot(1,2,1)
# plt.contourf(ls_PI, con_lv, origin='lower', cmap='rainbow', extent=ext, aspect='auto');
# plt.colorbar(); plt.xlabel('x'); plt.ylabel('y')
# ax1 = fig.add_subplot(1,2,2)
# plt.contourf(ls_uncon, con_lv, origin='lower', cmap='rainbow', extent=ext, aspect='auto');
# plt.colorbar(); plt.xlabel('x'); plt.ylabel('y')
# plt.show()


seed = 22
file = np.load(f'/home/weiz/Astar_Work/NAS/NAS-Unet-PINN6/search/NS/scripts_example/PI_denoising_number_{int(seed)}_step_0.npz')
# file = np.load(f'/home/weiz/Astar_Work/NAS/NAS-Unet-PINN5/search/NS/scripts_example/PI_denoising_results/PGPE_30_last_10/alpha_50_5cells_same_mean_revised_10/pipe_3/PI_denoising_number_{int(seed)}_step_0.npz')
x_in_PI, delta_p_PI = file['arr_1'], file['arr_3']

# file = np.load('/home/weiz/Astar_Work/NAS/NAS-Unet-PINN5/search/NS/scripts_example/test_PGPE/diffusion_denoising_step_97.npz')
# file = np.load('/home/weiz/Astar_Work/NAS/NAS-Unet-PINN5/search/NS/scripts_example/diffusion_denoising_results/100_samples/unconditional/diffusion_denoising_8.npz')
file = np.load(f'/home/weiz/Astar_Work/NAS/NAS-Unet-PINN5/search/NS/scripts_example/diffusion_denoising_results/5cells_same/pipe_3/diffusion_denoising_number_{int(seed)}_step_0.npz')
# file = np.load('/home/weiz/Astar_Work/NAS/NAS-Unet-PINN5/search/NS/scripts_example/diffusion_denoising_results/Case8/U-Net/unconditional/diffusion_denoising_step_0.npz')
# x_in_diffusion, delta_p_diffusion, grad_mean_diffusion, grad_std_diffusion = file['arr_1'], file['arr_3'], file['arr_4'], file['arr_5']
x_in_diffusion, delta_p_diffusion = file['arr_0'], file['arr_2']

# x_in_PI = torch.tensor(x_in_PI)
# data_PI = torch.nn.functional.interpolate(x_in_PI, size=(32, 64), mode='bilinear', align_corners=True)
# data_PI = torch.nn.functional.pad(data_PI, (0, 0, 16, 16, 0, 0, 0, 0), 'constant', torch.max(data_PI).detach().cpu().float())
# data_PI = torch.nn.functional.interpolate(data_PI, size=(128, 128), mode='bilinear', align_corners=True)
# data_PI = torch.nn.functional.interpolate(data_PI, size=(512, 512), mode='bilinear', align_corners=True)
# data_PI = torch.sigmoid(data_PI)
# data_PI = Index_Transformation_01(data_PI)
# data_PI[data_PI == 30] = 1
# data_PI = data_PI[0, 128:384, :]
# data_PI = data_PI.detach().cpu().numpy()


# x_in_diffusion = torch.tensor(x_in_diffusion)
# data_diffusion = torch.nn.functional.interpolate(x_in_diffusion, size=(32, 64), mode='bilinear', align_corners=True)
# data_diffusion = torch.nn.functional.pad(data_diffusion, (0, 0, 16, 16, 0, 0, 0, 0), 'constant', torch.max(data_diffusion).detach().cpu().float())
# data_diffusion = torch.nn.functional.interpolate(data_diffusion, size=(128, 128), mode='bilinear', align_corners=True)
# data_diffusion = torch.nn.functional.interpolate(data_diffusion, size=(512, 512), mode='bilinear', align_corners=True)
# data_diffusion = torch.sigmoid(data_diffusion)
# data_diffusion = Index_Transformation_01(data_diffusion)
# data_diffusion[data_diffusion == 30] = 1
# data_diffusion = data_diffusion[0, 128:384, :]
# data_diffusion = data_diffusion.detach().cpu().numpy()



x_in_PI = torch.tensor(x_in_PI)
# data_PI = torch.nn.functional.interpolate(x_in_PI, size=(256, 512), mode='bilinear', align_corners=True)
data_PI = torch.nn.functional.interpolate(x_in_PI, size=(256, 512), mode='nearest', align_corners=None)
data_PI = torch.sigmoid(data_PI)
data_PI = Index_Transformation_01(data_PI)
data_PI[data_PI == 30] = 1
data_PI[data_PI == 0] = -1
data_PI = data_PI.detach().cpu().numpy()[0, :, :]


x_in_diffusion = torch.tensor(x_in_diffusion)
data_diffusion = torch.nn.functional.interpolate(x_in_diffusion, size=(256, 512), mode='bilinear', align_corners=True)
# data_diffusion = torch.nn.functional.interpolate(x_in_diffusion, size=(256, 512), mode='nearest', align_corners=None)
data_diffusion = torch.sigmoid(data_diffusion)
data_diffusion = Index_Transformation_01(data_diffusion)
data_diffusion[data_diffusion == 30] = 1
data_diffusion[data_diffusion == 0] = -1
data_diffusion = data_diffusion.detach().cpu().numpy()[0, :, :]

fig = plt.figure(figsize=(20, 4.5))
x_l, x_u, y_l, y_u = 0, 2, 0, 1
ext = [x_l, x_u, y_l, y_u]
con_lv = 15
norm = matplotlib.colors.Normalize(vmin=-1, vmax=1)
# u
plt.figure(1)
ax1 = fig.add_subplot(1,2,1)
plt.contourf(data_PI, con_lv, origin='lower', cmap='rainbow', extent=ext, aspect='auto', norm=norm);
plt.colorbar(); plt.xlabel('x'); plt.ylabel('y')
ax1 = fig.add_subplot(1,2,2)
plt.contourf(data_PI - data_diffusion, con_lv, origin='lower', cmap='rainbow', extent=ext, aspect='auto', norm=norm);
plt.colorbar(); plt.xlabel('x'); plt.ylabel('y')
# plt.show()


grid_x = np.linspace(2/(512*2), 2 - 2/(512*2), num=512)
grid_y = np.linspace(1/(256*2), 1 - 1/(256*2), num=256)
grid_x, grid_y = np.meshgrid(grid_x, grid_y)
labels_PI = np.hstack([grid_x.reshape(-1, 1), grid_y.reshape(-1, 1), data_PI.reshape(-1, 1)])
labels_PI = np.asarray(labels_PI)
labels_diffusion = np.hstack([grid_x.reshape(-1, 1), grid_y.reshape(-1, 1), data_diffusion.reshape(-1, 1)])
labels_diffusion = np.asarray(labels_diffusion)
np.savetxt(f'PI_0_seed_{int(seed)}_n.dat', labels_PI, delimiter=',', fmt='%.18f')
np.savetxt(f'Uncon_0_seed_{int(seed)}_n.dat', labels_diffusion, delimiter=',', fmt='%.18f')

a

# data_load = np.loadtxt('/home/weiz/Astar_Work/NAS/NAS-Unet-PINN5/search/NS/scripts_example/PI_results/CFD_inputs/output_PI_seed_0.dat')
# # data_load = np.loadtxt('/home/weiz/Astar_Work/NAS/NAS-Unet-PINN5/search/NS/scripts_example/PI_results/CFD_inputs/output_PI_seed_2.dat')
# # data_load = np.loadtxt('/home/weiz/Astar_Work/NAS/NAS-Unet-PINN5/search/NS/scripts_example/PI_results/CFD_inputs/output_uncon_seed_0.dat')
# # data_load = np.loadtxt('/home/weiz/Astar_Work/NAS/NAS-Unet-PINN5/search/NS/scripts_example/PI_results/CFD_inputs/output_uncon_seed_2.dat')
# data_u, data_v, data_p, ls = data_load[:, 2].reshape(258, 514), data_load[:, 3].reshape(258, 514), data_load[:, 4].reshape(258, 514), data_load[:, 6].reshape(258, 514)
# ls[ls < 0] = 0
# ls[ls > 0] = 1
# data_u[ls == 1], data_v[ls == 1], data_p[ls == 1] = np.nan, np.nan, np.nan
# delta_p = np.max(data_p[:, 1])
# delta_p_n = np.log(delta_p)/5  # normalization
# print(delta_p, delta_p_n)

# # # case 5
# norm_u0 = matplotlib.colors.Normalize(vmin=0, vmax=1.8)
# norm_v0 = matplotlib.colors.Normalize(vmin=-0.8, vmax=0.4)
# norm_p0 = matplotlib.colors.Normalize(vmin=0, vmax=2.1)



# fig = plt.figure(figsize=(20, 4.5))
# x_l, x_u, y_l, y_u = 0, 2, 0, 1
# ext = [x_l, x_u, y_l, y_u]
# con_lv = 15
# ax1 = fig.add_subplot(1,3,1)
# plt.contourf(data_u, con_lv, origin='lower', cmap='rainbow', extent=ext, aspect='auto', norm=norm_u0);
# plt.colorbar(); plt.xlabel('x'); plt.ylabel('y')
# ax1 = fig.add_subplot(1,3,2)
# plt.contourf(data_v, con_lv, origin='lower', cmap='rainbow', extent=ext, aspect='auto', norm=norm_v0);
# plt.colorbar(); plt.xlabel('x'); plt.ylabel('y')
# ax1 = fig.add_subplot(1,3,3)
# plt.contourf(data_p, con_lv, origin='lower', cmap='rainbow', extent=ext, aspect='auto', norm=norm_p0);
# plt.colorbar(); plt.xlabel('x'); plt.ylabel('y')
# plt.show()



# # file = np.load('/home/weiz/Astar_Work/NAS/NAS-Unet-PINN5/search/NS/scripts_example/PI_results/unconditional_0.npz')
# file = np.load('/home/weiz/Astar_Work/NAS/NAS-Unet-PINN5/search/NS/scripts_example/PI_results/PI_0_1300.npz')
# outputs_pi, delta_p, delta_p_n = file['arr_0'], file['arr_1'], file['arr_2']
# outputs_pi[outputs_pi == 0] = np.nan
# outputs_pi = outputs_pi[:, :, 32:96, :]
# print(delta_p, delta_p_n)


# x_l, x_u, y_l, y_u = 0, 2, 0, 2
# ext = [x_l, x_u, y_l, y_u]

# fig = plt.figure(figsize=(20, 4.5))
# con_lv = 15
# # u
# ax1 = fig.add_subplot(1,3,1)
# u0 = outputs_pi[0, 0, :, :]
# plt.contourf(u0, con_lv, origin='lower', cmap='rainbow', extent=ext, aspect='auto', norm=norm_u0);
# plt.colorbar(); plt.xlabel('x'); plt.ylabel('y')
# ax1 = fig.add_subplot(1,3,2)
# v0 = outputs_pi[0, 1, :, :]
# plt.contourf(v0, con_lv, origin='lower', cmap='rainbow', extent=ext, aspect='auto', norm=norm_v0);
# plt.colorbar(); plt.xlabel('x'); plt.ylabel('y')
# ax1 = fig.add_subplot(1,3,3)
# p0 = outputs_pi[0, 2, :, :]
# plt.contourf(p0, con_lv, origin='lower', cmap='rainbow', extent=ext, aspect='auto', norm=norm_p0);
# plt.colorbar(); plt.xlabel('x'); plt.ylabel('y')
# plt.show()



















# # file = np.load('/home/weiz/Astar_Work/NAS/NAS-Unet-PINN5/search/NS/scripts_example/diffusion_denoising_results/100_samples/PGPE_30_last_10/alpha_9e4/diffusion_denoising_8.npz')
# # file = np.load('/home/weiz/Astar_Work/NAS/NAS-Unet-PINN5/search/NS/scripts_example/PI_denoising_results/PGPE_30_last_10/alpha_5e3/PI_denoising_number_0_step_0.npz')
# file = np.load('/home/weiz/Astar_Work/NAS/NAS-Unet-PINN5/search/NS/scripts_example/PI_denoising_results/PGPE_30_last_20/alpha_1e3/PI_denoising_number_0_step_0.npz')
# # file = np.load('/home/weiz/Astar_Work/NAS/NAS-Unet-PINN5/search/NS/scripts_example/PI_denoising_number_0_step_14.npz')
# # file = np.load('/home/weiz/Astar_Work/NAS/NAS-Unet-PINN5/search/NS/scripts_example/diffusion_denoising_results/100_samples/PGPE_30_last_20/alpha_5e6/diffusion_denoising_0.npz')
# # file = np.load('/home/weiz/Astar_Work/NAS/NAS-Unet-PINN5/search/NS/scripts_example/diffusion_denoising_results/100_samples/PGPE_30_last_30/alpha_1e4/diffusion_denoising_8.npz')
# x_in_PI, delta_p_PI = file['arr_1'], file['arr_3']

# # file = np.load('/home/weiz/Astar_Work/NAS/NAS-Unet-PINN5/search/NS/scripts_example/test_PGPE/diffusion_denoising_step_97.npz')
# # file = np.load('/home/weiz/Astar_Work/NAS/NAS-Unet-PINN5/search/NS/scripts_example/diffusion_denoising_results/100_samples/unconditional/diffusion_denoising_0.npz')
# file = np.load('/home/weiz/Astar_Work/NAS/NAS-Unet-PINN5/search/NS/scripts_example/diffusion_denoising_results/Case1/U-Net/unconditional/diffusion_denoising_step_0.npz')
# # x_in_diffusion, delta_p_diffusion, grad_mean_diffusion, grad_std_diffusion = file['arr_1'], file['arr_3'], file['arr_4'], file['arr_5']
# x_in_diffusion, delta_p_diffusion = file['arr_0'], file['arr_2']

# # # file = np.load('/home/weiz/Astar_Work/NAS/NAS-Unet-PINN5/search/NS/scripts_example/diffusion_denoising_results/Case5/U-Net/unconditional/diffusion_denoising_step_0.npz')
# # file = np.load('/home/weiz/Astar_Work/NAS/NAS-Unet-PINN5/search/NS/scripts_example/diffusion_denoising_results/Case5/U-Net/conditional/alpha_100/diffusion_denoising_step_0.npz')
# # x_in_diffusion, delta_p_diffusion = file['arr_0'], file['arr_2']

# print(delta_p_PI, delta_p_diffusion)


# fig = plt.figure(figsize=(20, 4.5))
# x_l, x_u, y_l, y_u = 0, 2, 0, 2
# ext = [x_l, x_u, y_l, y_u]
# con_lv = 15
# norm = matplotlib.colors.Normalize(vmin=-1, vmax=1)
# # u
# plt.figure(1)
# ax1 = fig.add_subplot(1,2,1)
# plt.contourf(x_in_PI[0, 0, :, :], con_lv, origin='lower', cmap='rainbow', extent=ext, aspect='auto', norm=norm);
# plt.colorbar(); plt.xlabel('x'); plt.ylabel('y')
# ax1 = fig.add_subplot(1,2,2)
# plt.contourf(x_in_diffusion[0, 0, :, :], con_lv, origin='lower', cmap='rainbow', extent=ext, aspect='auto', norm=norm);
# plt.colorbar(); plt.xlabel('x'); plt.ylabel('y')
# plt.show()


# device = 'cuda:0'
# x_PI = torch.tensor(x_in_PI).to(device)
# # x_PI = torch.nn.functional.interpolate(x_PI, size=(32, 64), mode='bilinear', align_corners=True)
# # x_PI = torch.nn.functional.pad(x_PI, (0, 0, 16, 16, 0, 0, 0, 0), 'constant', torch.max(x_PI).detach().cpu().float())
# # x_PI = torch.nn.functional.interpolate(x_PI, size=(128, 128), mode='bilinear', align_corners=True)
# x_PI = torch.sigmoid(x_PI)
# x_PI = Index_Transformation_01(x_PI)
# # x = Index_Transformation(x)

# x_diffusion = torch.tensor(x_in_diffusion).to(device)
# x_diffusion = torch.sigmoid(x_diffusion)
# x_diffusion = Index_Transformation_01(x_diffusion)
# # x = Index_Transformation(x)


# fig = plt.figure(figsize=(20, 4.5))
# ax1 = fig.add_subplot(1,2,1)
# plt.contourf(x_PI.detach().cpu().numpy()[0, :, :], con_lv, origin='lower', cmap='rainbow', extent=ext, aspect='auto');
# plt.colorbar(); plt.xlabel('x'); plt.ylabel('y')
# ax1 = fig.add_subplot(1,2,2)
# plt.contourf(x_diffusion.detach().cpu().numpy()[0, :, :], con_lv, origin='lower', cmap='rainbow', extent=ext, aspect='auto');
# plt.colorbar(); plt.xlabel('x'); plt.ylabel('y')
# plt.show()



# fig = plt.figure(figsize=(20, 4.5))
# ax1 = fig.add_subplot(1,2,1)
# plt.contourf(x_PI.detach().cpu().numpy()[0, :, :], con_lv, origin='lower', cmap='rainbow', extent=ext, aspect='auto');
# plt.colorbar(); plt.xlabel('x'); plt.ylabel('y')
# ax1 = fig.add_subplot(1,2,2)
# plt.contourf(x_PI.detach().cpu().numpy()[0, :, :] - x_diffusion.detach().cpu().numpy()[0, :, :], con_lv, origin='lower', cmap='rainbow', extent=ext, aspect='auto');
# plt.colorbar(); plt.xlabel('x'); plt.ylabel('y')
# plt.show()







# delta_p_PGPE_all = []
# delta_p_uncon_all = []
# for i in np.arange(0, 100, 1):
#     name_PGPE = '/home/weiz/Astar_Work/NAS/NAS-Unet-PINN5/search/NS/scripts_example/diffusion_denoising_results/100_samples/PGPE_30_last_10/alpha_9e4/diffusion_denoising_{n}.npz'.format(n = int(i))
#     # name_PGPE = '/home/weiz/Astar_Work/NAS/NAS-Unet-PINN5/search/NS/scripts_example/diffusion_denoising_results/100_samples/PGPE_30_last_30/alpha_1e4/diffusion_denoising_{n}.npz'.format(n = int(i))
#     # name_PGPE = '/home/weiz/Astar_Work/NAS/NAS-Unet-PINN5/search/NS/scripts_example/diffusion_denoising_{n}.npz'.format(n = int(i))
#     # name_PGPE = '/home/weiz/Astar_Work/NAS/NAS-Unet-PINN5/search/NS/scripts_example/diffusion_denoising_results/100_samples/conditional/alpha_10/diffusion_denoising_{n}.npz'.format(n = int(i))
#     file_PGPE = np.load(name_PGPE)
#     x_in_PGPE, delta_p_PGPE = file_PGPE['arr_1'], file_PGPE['arr_3']
#     # x_in_PGPE, delta_p_PGPE = file_PGPE['arr_0'], file_PGPE['arr_2']
#     name_uncon = '/home/weiz/Astar_Work/NAS/NAS-Unet-PINN5/search/NS/scripts_example/diffusion_denoising_results/100_samples/unconditional/diffusion_denoising_{n}.npz'.format(n = int(i))
#     file_uncon = np.load(name_uncon)
#     x_in_uncon, delta_p_uncon = file_uncon['arr_0'], file_uncon['arr_2']

#     # name_uncon = '/home/weiz/Astar_Work/NAS/NAS-Unet-PINN5/search/NS/scripts_example/diffusion_denoising_results/100_samples/conditional/alpha_10/diffusion_denoising_{n}.npz'.format(n = int(i))
#     # file_uncon = np.load(name_uncon)
#     # x_in_uncon, delta_p_uncon = file_uncon['arr_0'], file_uncon['arr_2']
#     # # name_uncon = '/home/weiz/Astar_Work/NAS/NAS-Unet-PINN5/search/NS/scripts_example/diffusion_denoising_results/100_samples/PGPE_50/alpha_100/diffusion_denoising_{n}.npz'.format(n = int(i))
#     # # file_uncon = np.load(name_uncon)
#     # # x_in_uncon, delta_p_uncon = file_uncon['arr_1'], file_uncon['arr_3']

#     delta_p_PGPE_all.append(delta_p_PGPE.reshape(1,))
#     delta_p_uncon_all.append(delta_p_uncon.reshape(1,))

# mean_PGPE = np.mean(delta_p_PGPE_all)
# std_PGPE = np.std(delta_p_PGPE_all)
# mean_uncon = np.mean(delta_p_uncon_all)
# std_uncon = np.std(delta_p_uncon_all)

# plt.figure(1)
# plt.plot(delta_p_PGPE_all, 's', label = 'PGPE')
# plt.plot(delta_p_uncon_all, '*', label = 'Unconditional')
# plt.xlabel('Seed')
# plt.ylabel('delta p')
# plt.legend()

# plt.figure(2)
# plt.hist(np.array(delta_p_PGPE_all) - np.array(delta_p_uncon_all))
# plt.show()




# # file = np.load('/home/weiz/Astar_Work/NAS/NAS-Unet-PINN5/search/NS/scripts_example/test_u/diffusion_denoising_step_97.npz')
# file = np.load('/home/weiz/Astar_Work/NAS/NAS-Unet-PINN5/search/NS/scripts_example/PI_denoising_results/PGPE_30_last_10/alpha_2e2/PI_denoising_number_0_step_9.npz')
# # x_in_PI, delta_p_PI, grad_mean_PI, grad_std_PI = file['arr_1'], file['arr_3'], file['arr_4'], file['arr_5']
# x_in_PI, delta_p_PI, grad_mean_PI = file['arr_1'], file['arr_3'], file['arr_4']

# # file = np.load('/home/weiz/Astar_Work/NAS/NAS-Unet-PINN5/search/NS/scripts_example/test_PGPE/diffusion_denoising_step_97.npz')
# file = np.load('/home/weiz/Astar_Work/NAS/NAS-Unet-PINN5/search/NS/scripts_example/diffusion_denoising_results/Case1/U-Net/unconditional/diffusion_denoising_step_0.npz')
# # file = np.load('/home/weiz/Astar_Work/NAS/NAS-Unet-PINN5/search/NS/scripts_example/diffusion_denoising_results/100_samples/unconditional/diffusion_denoising_0.npz')
# # x_in_diffusion, delta_p_diffusion, grad_mean_diffusion, grad_std_diffusion = file['arr_1'], file['arr_3'], file['arr_4'], file['arr_5']
# x_in_diffusion, delta_p_diffusion = file['arr_0'], file['arr_2']

# # # file = np.load('/home/weiz/Astar_Work/NAS/NAS-Unet-PINN5/search/NS/scripts_example/diffusion_denoising_results/Case5/U-Net/unconditional/diffusion_denoising_step_0.npz')
# # file = np.load('/home/weiz/Astar_Work/NAS/NAS-Unet-PINN5/search/NS/scripts_example/diffusion_denoising_results/Case5/U-Net/conditional/alpha_100/diffusion_denoising_step_0.npz')
# # x_in_diffusion, delta_p_diffusion = file['arr_0'], file['arr_2']

# fig = plt.figure(figsize=(20, 4.5))
# x_l, x_u, y_l, y_u = 0, 2, 0, 2
# ext = [x_l, x_u, y_l, y_u]
# con_lv = 15
# norm = matplotlib.colors.Normalize(vmin=-1, vmax=1)
# # u
# plt.figure(1)
# ax1 = fig.add_subplot(1,2,1)
# plt.contourf(x_in_PI[0, 0, :, :], con_lv, origin='lower', cmap='rainbow', extent=ext, aspect='auto', norm=norm);
# plt.colorbar(); plt.xlabel('x'); plt.ylabel('y')
# ax1 = fig.add_subplot(1,2,2)
# plt.contourf(x_in_diffusion[0, 0, :, :], con_lv, origin='lower', cmap='rainbow', extent=ext, aspect='auto', norm=norm);
# plt.colorbar(); plt.xlabel('x'); plt.ylabel('y')
# plt.show()


# # outputs = np.load('/home/weiz/Astar_Work/NAS/NAS-Unet-PINN5/search/NS/scripts_example/test.npy')
# # fig = plt.figure(figsize=(20, 4.5))
# # x_l, x_u, y_l, y_u = 0, 2, 0, 2
# # ext = [x_l, x_u, y_l, y_u]
# # con_lv = 15
# # ax1 = fig.add_subplot(1,3,1)
# # u0 = outputs[0, 0, :, :]
# # plt.contourf(u0, con_lv, origin='lower', cmap='rainbow', extent=ext, aspect='auto');
# # plt.colorbar(); plt.xlabel('x'); plt.ylabel('y')
# # ax1 = fig.add_subplot(1,3,2)
# # v0 = outputs[0, 1, :, :]
# # plt.contourf(v0, con_lv, origin='lower', cmap='rainbow', extent=ext, aspect='auto');
# # plt.colorbar(); plt.xlabel('x'); plt.ylabel('y')
# # ax1 = fig.add_subplot(1,3,3)
# # p0 = outputs[0, 2, :, :]
# # plt.contourf(p0, con_lv, origin='lower', cmap='rainbow', extent=ext, aspect='auto');
# # plt.colorbar(); plt.xlabel('x'); plt.ylabel('y')
# # plt.show()









file = np.load('/home/weiz/Astar_Work/NAS/NAS-Unet-PINN5/search/NS/scripts_example/PI_denoising_results/PGPE_30_last_20/alpha_15_5cells_same_mean_revised/pipe_3/PI_denoising_number_22_step_0.npz')
x_in_PI, delta_p_PI = file['arr_1'], file['arr_3']
# file = np.load('/home/weiz/Astar_Work/NAS/NAS-Unet-PINN5/search/NS/scripts_example/diffusion_denoising_results/Case5/PGPE/50_samples/alpha_100/diffusion_denoising_step_0.npz')
file = np.load('/home/weiz/Astar_Work/NAS/NAS-Unet-PINN5/search/NS/scripts_example/diffusion_denoising_results/Case1/U-Net/unconditional/diffusion_denoising_step_83.npz')
# file = np.load('/home/weiz/Astar_Work/NAS/NAS-Unet-PINN5/search/NS/scripts_example/diffusion_denoising_results/Case3/U-Net/conditional/alpha_100/diffusion_denoising_step_0.npz')
# file = np.load('/home/weiz/Astar_Work/NAS/NAS-Unet-PINN5/search/NS/scripts_example/diffusion_denoising_step_0.npz')
# x_in_diffusion, delta_p_diffusion, grad_mean_diffusion, grad_std_diffusion = file['arr_1'], file['arr_3'], file['arr_4'], file['arr_5']
x_in_diffusion = file['arr_0']


# # file = np.load('/home/weiz/Astar_Work/NAS/NAS-Unet-PINN5/search/NS/scripts_example/diffusion_denoising_results/Case5/U-Net/unconditional/diffusion_denoising_step_0.npz')
# file = np.load('/home/weiz/Astar_Work/NAS/NAS-Unet-PINN5/search/NS/scripts_example/diffusion_denoising_results/Case5/U-Net/conditional/alpha_100/diffusion_denoising_step_0.npz')
# x_in_diffusion, delta_p_diffusion = file['arr_0'], file['arr_2']

fig = plt.figure(figsize=(20, 4.5))
x_l, x_u, y_l, y_u = 0, 2, 0, 2
ext = [x_l, x_u, y_l, y_u]
con_lv = 15
norm = matplotlib.colors.Normalize(vmin=-1, vmax=1)
# u
plt.figure(1)
ax1 = fig.add_subplot(1,2,1)
plt.contourf(x_in_PI[0, 0, :, :], con_lv, origin='lower', cmap='rainbow', extent=ext, aspect='auto', norm=norm);
plt.colorbar(); plt.xlabel('x'); plt.ylabel('y')
ax1 = fig.add_subplot(1,2,2)
plt.contourf(x_in_diffusion[0, 0, :, :], con_lv, origin='lower', cmap='rainbow', extent=ext, aspect='auto', norm=norm);
plt.colorbar(); plt.xlabel('x'); plt.ylabel('y')
plt.show()

a_PI = []
a_diffusion = []
for i in np.arange(19, -1, -1):
    # name_PI = '/home/weiz/Astar_Work/NAS/NAS-Unet-PINN5/search/NS/scripts_example/PI_denoising_number_10_step_{n}.npz'.format(n = int(i))
    # name_PI = '/home/weiz/Astar_Work/NAS/NAS-Unet-PINN5/search/NS/scripts_example/diffusion_denoising_results/Case1/PGPE/30_samples/alpha_1e4/PI_denoising_step_{n}.npz'.format(n = int(i))
    # name_PI = '/home/weiz/Astar_Work/NAS/NAS-Unet-PINN5/search/NS/scripts_example/PI_denoising_results/PGPE_30_last_10/alpha_5e3/PI_denoising_number_0_step_{n}.npz'.format(n = int(i))
    name_PI = '/home/weiz/Astar_Work/NAS/NAS-Unet-PINN5/search/NS/scripts_example/PI_denoising_results/PGPE_30_last_20/alpha_1200_revised/PI_denoising_number_11_step_{n}.npz'.format(n = int(i))
    file_PI = np.load(name_PI)
    x_in_PI, delta_p_PI = file_PI['arr_1'], file_PI['arr_3']
    # name_diffusion = '/home/weiz/Astar_Work/NAS/NAS-Unet-PINN5/search/NS/scripts_example/diffusion_denoising_results/Case3/PGPE/30_samples/alpha_1e4/diffusion_denoising_step_{n}.npz'.format(n = int(i))
    # # name_diffusion = '/home/weiz/Astar_Work/NAS/NAS-Unet-PINN5/search/NS/scripts_example/diffusion_denoising_step_{n}.npz'.format(n = int(i))
    # file_diffusion = np.load(name_diffusion)
    # x_in_diffusion, delta_p_diffusion = file_diffusion['arr_1'], file_diffusion['arr_3']
    # name_diffusion = '/home/weiz/Astar_Work/NAS/NAS-Unet-PINN5/search/NS/scripts_example/diffusion_denoising_step_{n}.npz'.format(n = int(i))
    name_diffusion = '/home/weiz/Astar_Work/NAS/NAS-Unet-PINN5/search/NS/scripts_example/diffusion_denoising_results/Case1/U-Net/unconditional/diffusion_denoising_step_{n}.npz'.format(n = int(i))
    # name_diffusion = '/home/weiz/Astar_Work/NAS/NAS-Unet-PINN5/search/NS/scripts_example/diffusion_denoising_results/Case1/U-Net/conditional/alpha_100/diffusion_denoising_step_{n}.npz'.format(n = int(i))
    # name_diffusion = '/home/weiz/Astar_Work/NAS/NAS-Unet-PINN5/search/NS/scripts_example/diffusion_denoising_results/Case1/U-Net/conditional/alpha_100/diffusion_denoising_step_{n}.npz'.format(n = int(i))
    file_diffusion = np.load(name_diffusion)
    x_in_diffusion, delta_p_diffusion = file_diffusion['arr_0'], file_diffusion['arr_2']
    a_PI.append(delta_p_PI.reshape(1,))
    a_diffusion.append(delta_p_diffusion.reshape(1,))
plt.figure(2)
plt.plot(a_PI, '-*')
plt.figure(3)
plt.plot(a_diffusion, '-*')
plt.show()






file = np.load('/home/weiz/Astar_Work/NAS/NAS-Unet-PINN5/search/NS/scripts_example/PI_denoising_results/Case1/10_steps_kernel7/-1/PI_denoising_step_2.npz')
# file = np.load('/home/weiz/Astar_Work/NAS/NAS-Unet-PINN5/search/NS/scripts_example/PI_denoising_step_0.npz')
x_in_PI, delta_p_PI = file['arr_0'], file['arr_2']
file = np.load('/home/weiz/Astar_Work/NAS/NAS-Unet-PINN5/search/NS/scripts_example/diffusion_denoising_results/Case1/U-Net/diffusion_denoising_step_1.npz')
x_in_diffusion, delta_p_diffusion = file['arr_0'], file['arr_2']
# plt.figure(1)
# plt.imshow(x_in_PI[0, 0, :, :])
# plt.figure(2)
# plt.imshow(x_in_diffusion[0, 0, :, :])
# plt.show()
fig = plt.figure(figsize=(20, 4.5))
x_l, x_u, y_l, y_u = 0, 2, 0, 2
ext = [x_l, x_u, y_l, y_u]
con_lv = 15
norm = matplotlib.colors.Normalize(vmin=-1, vmax=1)
# u
ax1 = fig.add_subplot(1,2,1)
plt.contourf(x_in_PI[0, 0, :, :], con_lv, origin='lower', cmap='rainbow', extent=ext, aspect='auto', norm=norm);
plt.colorbar(); plt.xlabel('x'); plt.ylabel('y')
ax1 = fig.add_subplot(1,2,2)
plt.contourf(x_in_diffusion[0, 0, :, :], con_lv, origin='lower', cmap='rainbow', extent=ext, aspect='auto', norm=norm);
plt.colorbar(); plt.xlabel('x'); plt.ylabel('y')
plt.show()



ind_x_in_PI = Index_Transformation_01(torch.tensor(x_in_PI))
ind_x_in_diffusion = Index_Transformation_01(torch.tensor(x_in_diffusion))
ind_x_in_PI = ind_x_in_PI.detach().cpu().numpy()[0, :, :]
ind_x_in_diffusion = ind_x_in_diffusion.detach().cpu().numpy()[0, :, :]



# outputs_pi = np.load('/home/weiz/Astar_Work/NAS/NAS-Unet-PINN5/search/NS/scripts_example/PI_denoising_results/Case1/20_steps/PI_final_results_step_0.npy')
# # outputs_pi = np.load('/home/weiz/Astar_Work/NAS/NAS-Unet-PINN5/search/NS/scripts_example/PI_final_results_step_0.npy')
# fig = plt.figure(figsize=(20, 4.5))
# x_l, x_u, y_l, y_u = 0, 2, 0, 2
# ext = [x_l, x_u, y_l, y_u]
# con_lv = 15
# norm = matplotlib.colors.Normalize(vmin=-1, vmax=1)
# data0 = np.load('/home/weiz/Astar_Work/NAS/NAS-Unet-PINN5/search/NS/scripts_example/PI_denoising_results/Case1/10_steps/PI_denoising_step_0.npz')['arr_0']
# data = torch.tensor(data0)
# data = torch.nn.functional.interpolate(data, size=(128, 128), mode='bilinear', align_corners=True)
# data = data.detach().cpu().numpy()
# x_ind_bc = Index_Transformation1(data)
# ind = x_ind_bc.repeat(3, axis=1)
# outputs_pi[ind >= 30] = 0
# ax1 = fig.add_subplot(1,3,1)
# u0 = outputs_pi[0, 0, :, :]
# plt.contourf(u0, con_lv, origin='lower', cmap='rainbow', extent=ext, aspect='auto');
# plt.colorbar(); plt.xlabel('x'); plt.ylabel('y')
# ax1 = fig.add_subplot(1,3,2)
# v0 = outputs_pi[0, 1, :, :]
# plt.contourf(v0, con_lv, origin='lower', cmap='rainbow', extent=ext, aspect='auto');
# plt.colorbar(); plt.xlabel('x'); plt.ylabel('y')
# ax1 = fig.add_subplot(1,3,3)
# p0 = outputs_pi[0, 2, :, :]
# plt.contourf(p0, con_lv, origin='lower', cmap='rainbow', extent=ext, aspect='auto');
# plt.colorbar(); plt.xlabel('x'); plt.ylabel('y')
# plt.show()



# x_in_PI = np.load('/home/weiz/Astar_Work/NAS/NAS-Unet-PINN5/search/NS/scripts_example/PI_denoising_results/Case2/PI_final.npy')
# x_in_diffusion = np.load('/home/weiz/Astar_Work/NAS/NAS-Unet-PINN5/search/NS/scripts_example/diffusion_denoising_results/Case2/diffusion_final.npy')
# # plt.figure(1)
# # plt.imshow(x_in_PI[0, 0, :, :])
# # plt.figure(2)
# # plt.imshow(x_in_diffusion[0, 0, :, :])
# # plt.show()
# fig = plt.figure(figsize=(20, 4.5))
# x_l, x_u, y_l, y_u = 0, 2, 0, 2
# ext = [x_l, x_u, y_l, y_u]
# con_lv = 15
# norm = matplotlib.colors.Normalize(vmin=-1, vmax=1)
# # u
# ax1 = fig.add_subplot(1,2,1)
# plt.contourf(x_in_PI[0, 0, :, :], con_lv, origin='lower', cmap='rainbow', extent=ext, aspect='auto', norm=norm);
# plt.colorbar(); plt.xlabel('x'); plt.ylabel('y')
# ax1 = fig.add_subplot(1,2,2)
# plt.contourf(x_in_diffusion[0, 0, :, :], con_lv, origin='lower', cmap='rainbow', extent=ext, aspect='auto', norm=norm);
# plt.colorbar(); plt.xlabel('x'); plt.ylabel('y')
# plt.show()


a_PI = []
a_diffusion = []
for i in np.arange(10, -1, -1):
    # name_PI = '/home/weiz/Astar_Work/NAS/NAS-Unet-PINN5/search/NS/scripts_example/PI_denoising_results/Case1/20_steps/PI_denoising_step_{n}.npz'.format(n = int(i))
    name_PI = '/home/weiz/Astar_Work/NAS/NAS-Unet-PINN5/search/NS/scripts_example/PI_denoising_results/Case1/10_steps_kernel7/-e-1/PI_denoising_step_{n}.npz'.format(n = int(i))
    file_PI = np.load(name_PI)
    x_in_PI, delta_p_PI = file_PI['arr_0'], file_PI['arr_2']
    name_diffusion = '/home/weiz/Astar_Work/NAS/NAS-Unet-PINN5/search/NS/scripts_example/diffusion_denoising_results/Case1/diffusion_denoising_step_{n}.npz'.format(n = int(i))
    file_diffusion = np.load(name_diffusion)
    x_in_diffusion, delta_p_diffusion = file_diffusion['arr_0'], file_diffusion['arr_2']
    a_PI.append(delta_p_PI)
    a_diffusion.append(delta_p_diffusion.reshape(1,))
plt.figure(1)
plt.plot(a_PI, '-*')
plt.figure(2)
plt.plot(a_diffusion, '-*')
plt.show()



# # Case 1
# a_PI = [0.41418314, 0.4258571, 0.4315323, 0.4285984, 0.44204378, 0.4492397, 0.4650825, 0.4541004, 0.4494277, 0.45737198, 0.44372416]
# a_diffusion = [0.25252056, 0.25138175, 0.2508782, 0.2500553, 0.24987854, 0.24850978, 0.24898656, 0.24955024, 0.24996294, 0.24947663, 0.24878882]
# # Case 2
# # a_PI = [0.31437868, 0.3138416, 0.30689004, 0.31644124, 0.33102405, 0.43698627, 0.44172055, 0.43570587, 0.43476048, 0.44969285, 0.44571427]
# # a_diffusion = [0.21895397, 0.22064261, 0.21810731, 0.21600199, 0.21962282, 0.21971425, 0.21658146, 0.21428907, 0.21359974, 0.21346457, 0.21220939]
# plt.figure(1)
# plt.plot(a_PI, '-*')
# plt.figure(2)
# plt.plot(a_diffusion, '-*')
# plt.show()



a
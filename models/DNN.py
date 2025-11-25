#!/usr/bin/env python
# coding: utf-8

# # Attribute
# 
# **Original Work**: *Maziar Raissi, Paris Perdikaris, and George Em Karniadakis*
# 
# **Github Repo** : https://github.com/maziarraissi/PINNs
# 
# **Link:** https://github.com/maziarraissi/PINNs/tree/master/appendix/continuous_time_identification%20(Burgers)
# 
# @article{raissi2017physicsI,
#   title={Physics Informed Deep Learning (Part I): Data-driven Solutions of Nonlinear Partial Differential Equations},
#   author={Raissi, Maziar and Perdikaris, Paris and Karniadakis, George Em},
#   journal={arXiv preprint arXiv:1711.10561},
#   year={2017}
# }
# 
# @article{raissi2017physicsII,
#   title={Physics Informed Deep Learning (Part II): Data-driven Discovery of Nonlinear Partial Differential Equations},
#   author={Raissi, Maziar and Perdikaris, Paris and Karniadakis, George Em},
#   journal={arXiv preprint arXiv:1711.10566},
#   year={2017}
# }

# ## Libraries and Dependencies


import torch
from collections import OrderedDict



# the deep neural network
class DNN(torch.nn.Module):
    def __init__(self, layers, i_ac):
        super(DNN, self).__init__()
        
        # parameters
        self.depth = len(layers) - 1
        
        # set up layer order dict
        af_all = {'1': torch.nn.Tanh, '2': torch.nn.Sigmoid, '3': torch.nn.SELU, '4': torch.nn.Softmax, '5': torch.nn.ReLU, '6': torch.nn.ELU, '7': torch.nn.Softplus, '8': torch.nn.LeakyReLU}
        self.activation = af_all[str(i_ac)]
        
        layer_list = list()
        for i in range(self.depth - 1):
            layer_list.append(
                ('layer_%d' % i, torch.nn.Linear(layers[i], layers[i+1]))
            )
            layer_list.append(('activation_%d' % i, self.activation()))
            
        layer_list.append(
            ('layer_%d' % (self.depth - 1), torch.nn.Linear(layers[-2], layers[-1]))
        )
        layerDict = OrderedDict(layer_list)
        
        # deploy layers
        self.layers = torch.nn.Sequential(layerDict)
        
    def forward(self, x):
        out = self.layers(x)
        return out


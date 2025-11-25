import sys
# update your projecty root path before running
# sys.path.insert(0, r'/home/weiz/Astar_Work/NAS/nsga-net-master')
sys.path.append(r'/home/weiz/Astar_Work/NAS/NAS-Unet-PINN5')


import numpy as np
import pyLHD
import matplotlib.pyplot as plt
import random
import time

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.cluster import k_means, kmeans_plusplus

from pymoo.core.problem import Problem, ElementwiseProblem
from pymoo.core.variable import Real, Integer
from pymoo.optimize import minimize
from pymoo.algorithms.moo.nsga2 import NSGA2, RankAndCrowdingSurvival
from pymoo.core.mixed import MixedVariableMating, MixedVariableGA, MixedVariableSampling, MixedVariableDuplicateElimination
from pymoo.visualization.scatter import Scatter
from pymoo.indicators.hv import HV
from pymoo.problems import get_problem
from pymoo.core.population import Population

from models.Pareto_Front import Pareto
from scipy.spatial.distance import cdist
from scipy.stats import norm

from models.benchmarks import ZDT1, ZDT2, ZDT3, ZDT4, ZDT6, ZDT1_LF, ZDT2_LF, ZDT3_LF

from smt.applications.mfk import MFK, NestedLHS

from search import micro_encoding
# from search.heat_equation import train_DNN_Heat_Equation_DirichletBC_NAS
# from search.convection_diffusion import train_PINN_Convection_Diffusion_DirichletBC_NAS
# from search.poiseuille_flow import train_DNN_Poiseuille_Flow_DirichletBC_NAS
from search.NS.scripts_example import train_DNN_NS_DirichletBC_NAS


# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"



class SADE(object):

    def __init__(self, nv, ns, npop, nnear, F, CR, nobj, nfe_max, lb, ub, tolerance):

        self.nv = nv  # dimentionality of optimization problem
        self.ns = ns  # number of initial samples
        self.npop = npop  # number of population size
        self.nnear = nnear  # number of near individuals
        self.F = F  # mutation factor
        self.CR = CR  # crossover factor
        self.nobj = nobj  # number of objectives
        self.nfe_max = nfe_max  # maximum number of function evaluations
        self.tolerance = tolerance  # maximum tolerance
        self.lb = lb  # lower bound
        self.ub = ub  # upper bound


    def initial_sampling(self):
        x_init_LHD = pyLHD.rLHD(nrows = self.ns, ncols = self.nv, unit_cube = False)
        x_init_norm = (x_init_LHD - 1) / (self.ns - 1)
        x_init = self.lb + x_init_norm * (self.ub - self.lb)
        return x_init, x_init_norm
    
    
    def objective_evaluate(self, x, current_nfe, init_channels, depth, epochs, save_dir):
        # y =  3*(1-x[:, 0:1])**2*np.exp(-(x[:, 0:1]**2) - (x[:, 1:2]+1)**2) - 10*(x[:, 0:1]/5 - x[:, 0:1]**3 - x[:, 1:2]**5)*np.exp(-x[:, 0:1]**2-x[:, 1:2]**2) - 1/3*np.exp(-(x[:, 0:1]+1)**2 - x[:, 1:2]**2)  # PK function
        # y1 = x[:, 0:1]**4 - 10*x[:, 0:1]**2 + x[:, 0:1]*x[:, 1:2] + x[:, 1:2]**4 - (x[:, 0:1]**2)*(x[:, 1:2]**2)
        # y2 = x[:, 1:2]**4 - (x[:, 0:1]**2)*(x[:, 1:2]**2) + x[:, 0:1]**4 + x[:, 0:1]*x[:, 1:2]
        # y = np.hstack([y1, y2])

        # y = ZDT3(x, self.nv)

        # NAS evulation
        y = np.zeros((x.shape[0], self.nobj))
        for i in np.arange(x.shape[0]):
            x_inputs = x[i]
            # genome = micro_encoding.convert_continuous_variables(x_inputs[:-1])
            # lamda = x_inputs[-1]

            genome = micro_encoding.convert_continuous_variables(x_inputs[:-2])
            lamda = x_inputs[-2]
            reg = x_inputs[-1]

            # LDC
            gpu = 0

            # # 2D heat equation
            # performance = train_DNN_Heat_Equation_DirichletBC_NAS.main(x_inputs=x_inputs,
            #                                                            genome=genome, 
            #                                                            epochs=epochs, 
            #                                                            gpu=gpu,
            #                                                            init_channels=init_channels,
            #                                                            depth=depth,
            #                                                            save = 'arch_{}'.format(current_nfe + i + 1),
            #                                                            expr_root = save_dir,
            #                                                            lamda=lamda
            #                                                            )
            # y[i, 0] = performance['valid_RMSE']
            # y[i, 1] = performance['flops']


            # # 2D convection diffusion
            # performance = train_PINN_Convection_Diffusion_DirichletBC_NAS.main(x_inputs=x_inputs,
            #                                                            genome=genome, 
            #                                                            epochs=epochs, 
            #                                                            gpu=gpu,
            #                                                            init_channels=init_channels,
            #                                                            depth=depth,
            #                                                            save = 'arch_{}'.format(current_nfe + i + 1),
            #                                                            expr_root = save_dir,
            #                                                            lamda=lamda
            #                                                            )
            # y[i, 0] = performance['valid_RMSE']
            # y[i, 1] = performance['flops']

            # # 2D convection diffusion
            # performance = train_DNN_Poiseuille_Flow_DirichletBC_NAS.main(x_inputs=x_inputs,
            #                                                            genome=genome, 
            #                                                            epochs=epochs, 
            #                                                            gpu=gpu,
            #                                                            init_channels=init_channels,
            #                                                            depth=depth,
            #                                                            save = 'arch_{}'.format(current_nfe + i + 1),
            #                                                            expr_root = save_dir,
            #                                                            lamda=lamda
            #                                                            )
            # y[i, 0] = performance['valid_mse']
            # y[i, 1] = performance['flops']


            # NS
            performance = train_DNN_NS_DirichletBC_NAS.main(x_inputs=x_inputs,
                                                                 genome=genome, 
                                                                 epochs=epochs, 
                                                                 gpu=gpu,
                                                                 init_channels=init_channels,
                                                                 depth=depth,
                                                                 save = 'arch_{}'.format(current_nfe + i + 1),
                                                                 expr_root = save_dir,
                                                                 lamda=lamda,
                                                                 reg=reg
                                                                 )
            y[i, 0] = performance['valid_mse']
            y[i, 1] = performance['flops']

        return y
    


    def surrogate_prediction(self, x_train, y_train):
        gpr = {}
        for i in np.arange(self.nobj):
            kernel = 1 * RBF(length_scale = 1.0, length_scale_bounds = (1e-7, 1e7))
            gaussian_process = GaussianProcessRegressor(kernel = kernel, n_restarts_optimizer = 50)
            gpr[i] = gaussian_process.fit(x_train, y_train[:, i:i+1])
        return gpr
    

    def get_EI_pareto_front(self, gpr, y_all):
        problem = Multi_EI_Optimization(gpr, y_all, self.nv, self.nobj)
        algorithm = NSGA2(pop_size = 100, 
                      n_offsprings = 100,
                      sampling = MixedVariableSampling(),
                      mating = MixedVariableMating(eliminate_duplicates = MixedVariableDuplicateElimination()),
                      eliminate_duplicates = MixedVariableDuplicateElimination())
        res = minimize(problem, algorithm, termination=('n_gen', 100))
        return res
    
    def get_parent_population(self, x_all, x_all_norm, y_all):
        pareto = Pareto(x_all.shape[0], y_all)
        pareto.fast_non_dominate_sort()
        pareto.crowd_distance()
        # shuffle
        pop_index_all = []
        for i in np.arange(len(pareto.f)):
            index_pareto = pareto.f[i]
            np.random.shuffle(index_pareto)
            pop_index_all.append(index_pareto)

        pop_index = sum(pop_index_all, [])[0:self.npop]
        x_pp = x_all[pop_index, :]
        x_pp_norm = x_all_norm[pop_index, :]
        y_pp = y_all[pop_index, :]
        return x_pp, x_pp_norm, y_pp
    
    def evolutionary_operator(self, x_pp, x_pp_norm, y_pp):
        # find the best individual
        pareto = Pareto(x_pp.shape[0], y_pp)
        pareto.fast_non_dominate_sort()
        pareto.crowd_distance()
        i_best0 = pareto.f[0]
        i_best = np.random.choice(i_best0)
        
        x_os_rand_1 = np.zeros((x_pp.shape[0], x_pp.shape[1]))
        x_os_best_1 = np.zeros((x_pp.shape[0], x_pp.shape[1]))
        x_os_rand_2 = np.zeros((x_pp.shape[0], x_pp.shape[1]))
        x_os_best_2 = np.zeros((x_pp.shape[0], x_pp.shape[1]))
        x_os_current2rand_1 = np.zeros((x_pp.shape[0], x_pp.shape[1]))
        x_os_current2best_1 = np.zeros((x_pp.shape[0], x_pp.shape[1]))

        # DE/rand/1
        for i in np.arange(x_pp.shape[0]):
            # mutation
            n1 = np.random.choice(x_pp.shape[0], 3, replace = False)
            x_os_rand_1[i] = x_pp[n1[0]] + self.F*(x_pp[n1[1]] - x_pp[n1[2]])
            for j in np.arange(x_pp.shape[1]):
                # crossover
                n_rand = random.random()
                n_cro = np.random.choice(x_pp.shape[1])
                if (n_rand <= self.CR) | (j == n_cro):
                    x_os_rand_1[i, j] = x_os_rand_1[i, j]
                else:
                    x_os_rand_1[i, j] = x_pp[i, j]

            x_os_rand_1[i] = np.minimum(x_os_rand_1[i], self.ub)
            x_os_rand_1[i] = np.maximum(x_os_rand_1[i], self.lb)
        
        # DE/best/1
        for i in np.arange(x_pp.shape[0]):
            # mutation
            n1 = np.random.choice(x_pp.shape[0], 2, replace = False)
            x_os_best_1[i] = x_pp[i_best] + self.F*(x_pp[n1[0]] - x_pp[n1[1]])
            for j in np.arange(x_pp.shape[1]):
                # crossover
                n_rand = random.random()
                n_cro = np.random.choice(x_pp.shape[1])
                if (n_rand <= self.CR) | (j == n_cro):
                    x_os_best_1[i, j] = x_os_best_1[i, j]
                else:
                    x_os_best_1[i, j] = x_pp[i, j]

            x_os_best_1[i] = np.minimum(x_os_best_1[i], self.ub)
            x_os_best_1[i] = np.maximum(x_os_best_1[i], self.lb)

        # DE/rand/2
        for i in np.arange(x_pp.shape[0]):
            # mutation
            n1 = np.random.choice(x_pp.shape[0], 5, replace = False)
            x_os_rand_2[i] = x_pp[n1[0]] + self.F*(x_pp[n1[1]] - x_pp[n1[2]]) + self.F*(x_pp[n1[3]] - x_pp[n1[4]])
            for j in np.arange(x_pp.shape[1]):
                # crossover
                n_rand = random.random()
                n_cro = np.random.choice(x_pp.shape[1])
                if (n_rand <= self.CR) | (j == n_cro):
                    x_os_rand_2[i, j] = x_os_rand_2[i, j]
                else:
                    x_os_rand_2[i, j] = x_pp[i, j]

            x_os_rand_2[i] = np.minimum(x_os_rand_2[i], self.ub)
            x_os_rand_2[i] = np.maximum(x_os_rand_2[i], self.lb)

        # DE/best/2
        for i in np.arange(x_pp.shape[0]):
            # mutation
            n1 = np.random.choice(x_pp.shape[0], 4, replace = False)
            x_os_best_2[i] = x_pp[i_best] + self.F*(x_pp[n1[0]] - x_pp[n1[1]]) + self.F*(x_pp[n1[2]] - x_pp[n1[3]])
            for j in np.arange(x_pp.shape[1]):
                # crossover
                n_rand = random.random()
                n_cro = np.random.choice(x_pp.shape[1])
                if (n_rand <= self.CR) | (j == n_cro):
                    x_os_best_2[i, j] = x_os_best_2[i, j]
                else:
                    x_os_best_2[i, j] = x_pp[i, j]

            x_os_best_2[i] = np.minimum(x_os_best_2[i], self.ub)
            x_os_best_2[i] = np.maximum(x_os_best_2[i], self.lb)

        # DE/current-to-rand/1
        for i in np.arange(x_pp.shape[0]):
            # mutation
            n1 = np.random.choice(x_pp.shape[0], 3, replace = False)
            x_os_current2rand_1[i] = x_pp[i] + self.F*(x_pp[n1[0]] - x_pp[i]) + self.F*(x_pp[n1[1]] - x_pp[n1[2]])
            for j in np.arange(x_pp.shape[1]):
                # crossover
                n_rand = random.random()
                n_cro = np.random.choice(x_pp.shape[1])
                if (n_rand <= self.CR) | (j == n_cro):
                    x_os_current2rand_1[i, j] = x_os_current2rand_1[i, j]
                else:
                    x_os_current2rand_1[i, j] = x_pp[i, j]

            x_os_current2rand_1[i] = np.minimum(x_os_current2rand_1[i], self.ub)
            x_os_current2rand_1[i] = np.maximum(x_os_current2rand_1[i], self.lb)

        # DE/current-to-best/1
        for i in np.arange(x_pp.shape[0]):
            # mutation
            n1 = np.random.choice(x_pp.shape[0], 2, replace = False)
            x_os_current2best_1[i] = x_pp[i] + self.F*(x_pp[i_best] - x_pp[i]) + self.F*(x_pp[n1[0]] - x_pp[n1[1]])
            for j in np.arange(x_pp.shape[1]):
                # crossover
                n_rand = random.random()
                n_cro = np.random.choice(x_pp.shape[1])
                if (n_rand <= self.CR) | (j == n_cro):
                    x_os_current2best_1[i, j] = x_os_current2best_1[i, j]
                else:
                    x_os_current2best_1[i, j] = x_pp[i, j]

            x_os_current2best_1[i] = np.minimum(x_os_current2best_1[i], self.ub)
            x_os_current2best_1[i] = np.maximum(x_os_current2best_1[i], self.lb)

        x_offspring = np.vstack([x_os_rand_1, x_os_best_1, x_os_rand_2, x_os_best_2, x_os_current2rand_1, x_os_current2best_1])
        x_offspring_norm = (x_offspring - self.lb)/(self.ub - self.lb)
        return x_offspring, x_offspring_norm
    

    def find_near_individual(self, x_pp, x_pp_norm, y_pp, x_all, x_all_norm, y_all):
        pareto = Pareto(x_pp.shape[0], y_pp)
        pareto.fast_non_dominate_sort()
        pareto.crowd_distance()

        if (len(pareto.f[0]) == 1) & (len(pareto.f[1]) == 1):
            x_base = np.vstack([x_pp[pareto.f[0]], x_pp[pareto.f[1]], x_pp[np.random.choice(pareto.f[2])]])
        elif (len(pareto.f[0]) == 1) & (len(pareto.f[1]) > 1):
            x_base = np.vstack([x_pp[pareto.f[0]], x_pp[np.random.choice(pareto.f[1], 2, replace = False)]])
        elif len(pareto.f[0]) == 2:
            x_base = np.vstack([x_pp[pareto.f[0]], x_pp[np.random.choice(pareto.f[1])]])
        elif len(pareto.f[0]) > 2:
            x_base = x_pp[np.random.choice(pareto.f[0], 3, replace = False)]
        
        x_near = np.zeros((0, self.nv))
        x_near_norm = np.zeros((0, self.nv))
        y_near = np.zeros((0, self.nobj))
        for i in np.arange(x_base.shape[0]):
            dis = cdist(x_base[i].reshape(1, -1), x_all, metric = 'euclidean')
            arg_dis = np.argsort(dis)[0][0:self.nnear]
            x_near = np.vstack([x_near, x_all[arg_dis]])
            x_near_norm = np.vstack([x_near_norm, x_all_norm[arg_dis]])
            y_near = np.vstack([y_near, y_all[arg_dis]])
        
        xy_near = np.hstack([x_near, x_near_norm, y_near])
        # xy_near, index1 = np.unique(xy_near, axis = 0, return_index = True)
        xy_near = np.unique(xy_near, axis = 0)

        x_near = xy_near[:, 0:self.nv]
        x_near_norm = xy_near[:, self.nv:2*self.nv]
        y_near = xy_near[:, 2*self.nv:]
        return x_near, x_near_norm, y_near
    

    # def EI_function(self, x_norm, x_all, x_all_norm, y_all, gpr):
    #     EI_value = np.zeros((self.nobj, 1))
    #     for i in np.arange(self.nobj):
    #         y_min = np.min(y_all[:, i])
    #         y_predmean, y_predstd = gpr[i].predict(x_norm, return_std = True)
    #         EI_value[i] = (y_min - y_predmean)*norm.cdf((y_min - y_predmean)/y_predstd) + y_predstd*norm.pdf((y_min - y_predmean)/y_predstd)
    #     return EI_value


    def optimization_iteration(self, init_channels, depth, epochs, save_dir):
        start = time.time()
        current_nfe = 0
        x_init, x_init_norm = self.initial_sampling()
        y_init = self.objective_evaluate(x_init, current_nfe, init_channels, depth, epochs, save_dir)

        # train global surrogate
        gpr_global = self.surrogate_prediction(x_init_norm, y_init)
        # X = np.array([[1, 2]])
        # mean_prediction, std_prediction = gpr1.predict(X, return_std = True)

        # iteration
        current_nfe = current_nfe + self.ns
        x_all = x_init
        x_all_norm = x_init_norm
        y_all = y_init
        while current_nfe < self.nfe_max:
            # parent population
            x_pp, x_pp_norm, y_pp = self.get_parent_population(x_all, x_all_norm, y_all)
            x_offspring, x_offspring_norm = self.evolutionary_operator(x_pp, x_pp_norm, y_pp)

            # prediction for offspring
            y_offspring_predmean = np.zeros((x_offspring.shape[0], self.nobj))
            y_offspring_predstd = np.zeros((x_offspring.shape[0], self.nobj))
            for i in np.arange(self.nobj):
                y_offspring_predmean[:, i], y_offspring_predstd[:, i] = gpr_global[i].predict(x_offspring_norm, return_std = True)

            # get the Pareto front of prediction of multi-objectives
            pareto = Pareto(x_offspring.shape[0], y_offspring_predmean)
            pareto.fast_non_dominate_sort()
            pareto.crowd_distance()

            # select the best individual
            i_best0 = pareto.f[0]
            i_best = np.random.choice(i_best0)
            x_best = x_offspring[i_best].reshape(1, -1)
            x_best_norm = x_offspring_norm[i_best].reshape(1, -1)
            y_best = self.objective_evaluate(x_best, current_nfe, init_channels, depth, epochs, save_dir)

            # update database
            x_all = np.vstack([x_all, x_best])
            x_all_norm = np.vstack([x_all_norm, x_best_norm])
            y_all = np.vstack([y_all, y_best])
            current_nfe = current_nfe + 1

            # find near individuals near best individual in parent population
            x_near, x_near_norm, y_near = self.find_near_individual(x_pp, x_pp_norm, y_pp, x_all, x_all_norm, y_all)

            # train local surrogate
            gpr_local = self.surrogate_prediction(x_near_norm, y_near)

            # get the Pareto front of EI_1, EI_2, ...
            res_EI = self.get_EI_pareto_front(gpr_local, y_all)

            # plot EI Pareto front
            # PF_data_NSGA_II = res.F
            # PF_data_NSGA_II_sort = PF_data_NSGA_II[np.lexsort(PF_data_NSGA_II.T)]
            # fig = plt.figure(1)
            # # plt.plot(y_all[:, 0], y_all[:, 1], 'o')
            # plt.plot(PF_data_NSGA_II_sort[:, 0], PF_data_NSGA_II_sort[:, 1], '-k*')
            # plt.xlabel('$\mathit{f}_1$', fontsize = 14)
            # plt.ylabel('$\mathit{f}_2$', fontsize = 14)
            # # plt.xscale('log')
            # # plt.yscale('log')
            # plt.xticks(fontsize = 12)
            # plt.yticks(fontsize = 12)
            # plt.legend(fontsize = 12)
            # plt.grid()
            # plt.show()


            # k-means clustering
            x_pareto_front_EI = np.zeros((len(res_EI.X), self.nv))
            for i in range(len(res_EI.X)):
                x0 = res_EI.X[i]
                x_pareto_front_EI[i] = np.array([x0[f"x{k:02}"] for k in range(len(x0))])
            
            if x_pareto_front_EI.shape[0] == 1:
                n_clusters = 1
            elif x_pareto_front_EI.shape[0] == 2:
                n_clusters = 2
            else:
                n_clusters = 3

            clustering_results = k_means(x_pareto_front_EI, n_clusters, init = 'k-means++', random_state = 0)

            # select the cluster center for infill sampling
            x_add_norm = clustering_results[0]
            x_add = self.lb + x_add_norm * (self.ub - self.lb)
            y_add = self.objective_evaluate(x_add, current_nfe, init_channels, depth, epochs, save_dir)

            # update database
            x_all = np.vstack([x_all, x_add])
            x_all_norm = np.vstack([x_all_norm, x_add_norm])
            y_all = np.vstack([y_all, y_add])

            current_nfe = current_nfe + n_clusters
            print('current_nfe = {:3d}'.format(current_nfe))
            # print('lambda {:>.3f}: {:>5} {:>8} {:10.6f} {:10.6f} {:10.6f} {:10.6f}'.format(lmbda, i_s, train_iters, loss_all, ic_loss, pde_loss, mse))
            
            # update surrogate
            gpr_global = self.surrogate_prediction(x_all_norm, y_all)

        end = time.time()
        runtime = end - start
        return x_all_norm, x_all, y_all, gpr_global, runtime
    






class MF_SADE(object):

    def __init__(self, nv, ns, npop, nnear, F, CR, nobj, nfe_max, lb, ub, tolerance):

        self.nv = nv  # dimentionality of optimization problem
        self.ns = ns  # number of initial HF samples
        self.npop = npop  # number of population size
        self.nnear = nnear  # number of near individuals
        self.F = F  # mutation factor
        self.CR = CR  # crossover factor
        self.nobj = nobj  # number of objectives
        self.nfe_max = nfe_max  # maximum number of function evaluations
        self.tolerance = tolerance  # maximum tolerance
        self.lb = lb  # lower bound
        self.ub = ub  # upper bound


    def initial_sampling(self):
        xlimits = np.hstack([np.zeros((self.nv, 1)), np.ones((self.nv, 1))])
        xdoes = NestedLHS(nlevel = 2, xlimits = xlimits, random_state = np.random.randint(0, 1e6))
        x_LF_init_norm, x_HF_init_norm = xdoes(self.ns)
        x_LF_init = self.lb + x_LF_init_norm * (self.ub - self.lb)
        x_HF_init = self.lb + x_HF_init_norm * (self.ub - self.lb)
        return x_HF_init, x_LF_init, x_HF_init_norm, x_LF_init_norm
    
    
    def objective_evaluate(self, x_HF, x_LF, current_nfe, init_channels, depth, epochs_HF, epochs_LF, save_dir):
        # y =  3*(1-x[:, 0:1])**2*np.exp(-(x[:, 0:1]**2) - (x[:, 1:2]+1)**2) - 10*(x[:, 0:1]/5 - x[:, 0:1]**3 - x[:, 1:2]**5)*np.exp(-x[:, 0:1]**2-x[:, 1:2]**2) - 1/3*np.exp(-(x[:, 0:1]+1)**2 - x[:, 1:2]**2)  # PK function
        # y1 = x[:, 0:1]**4 - 10*x[:, 0:1]**2 + x[:, 0:1]*x[:, 1:2] + x[:, 1:2]**4 - (x[:, 0:1]**2)*(x[:, 1:2]**2)
        # y2 = x[:, 1:2]**4 - (x[:, 0:1]**2)*(x[:, 1:2]**2) + x[:, 0:1]**4 + x[:, 0:1]*x[:, 1:2]
        # y = np.hstack([y1, y2])

        # y_HF = ZDT1(x_HF, self.nv)
        # y_LF = ZDT1_LF(x_LF, self.nv)

        # NAS evulation
        y_HF = np.zeros((x_HF.shape[0], self.nobj))
        y_LF = np.zeros((x_LF.shape[0], self.nobj))

        # HF evaluation
        for i in np.arange(x_HF.shape[0]):
            x_inputs_HF = x_HF[i]
            genome_HF = micro_encoding.convert_continuous_variables(x_inputs_HF)

            # PDE
            gpu = 1
            # performance = train_PDE_search.main(x_inputs = x_inputs_HF,
            #                                 genome = genome_HF,
            #                                 search_space = 'micro',
            #                                 gpu = gpu,
            #                                 init_channels = init_channels,
            #                                 depth = depth,
            #                                 epochs = epochs_HF,
            #                                 save = 'arch_{}'.format(current_nfe + i + 1),
            #                                 expr_root = save_dir)
            # y_HF[i, 0] = performance['valid_RMSE']
            # y_HF[i, 1] = performance['flops']

            # CHASE_DB1
            performance = train_chase_search.main(x_inputs = x_inputs_HF,
                                            genome = genome_HF,
                                            search_space = 'micro',
                                            gpu = gpu,
                                            init_channels = init_channels,
                                            depth = depth,
                                            epochs = epochs_HF,
                                            save = 'arch_{}'.format(current_nfe + i + 1),
                                            expr_root = save_dir)
            y_HF[i, 0] = 1 - performance['valid_auc_roc']
            y_HF[i, 1] = performance['flops']



        # LF evaluation
        for i in np.arange(x_LF.shape[0]):
            x_inputs_LF = x_LF[i]
            genome_LF = micro_encoding.convert_continuous_variables(x_inputs_LF)

            # PDE
            gpu = 1
            # performance = train_PDE_search_LF.main(x_inputs = x_inputs_LF,
            #                             genome = genome_LF,
            #                             search_space = 'micro',
            #                             gpu = gpu,
            #                             init_channels = init_channels,
            #                             depth = depth,
            #                             epochs = epochs_LF,
            #                             save = 'arch_{}'.format(0),
            #                             expr_root = save_dir)
            # y_LF[i, 0] = performance['valid_RMSE']
            # y_LF[i, 1] = performance['flops']

            # CHASE_DB1
            performance = train_chase_search_LF.main(x_inputs = x_inputs_LF,
                                        genome = genome_LF,
                                        search_space = 'micro',
                                        gpu = gpu,
                                        init_channels = init_channels,
                                        depth = depth,
                                        epochs = epochs_LF,
                                        save = 'arch_{}'.format(0),
                                        expr_root = save_dir)
            y_LF[i, 0] = 1 - performance['valid_auc_roc']
            y_LF[i, 1] = performance['flops']

        return y_HF, y_LF
    


    def surrogate_prediction(self, x_HF_norm, x_LF_norm, y_HF, y_LF):
        mfk = {}
        for i in np.arange(self.nobj):
            mfk[i] = MFK(poly = 'constant', corr = 'squar_exp', theta0 = self.nv*[0.01], n_start = 50, hyper_opt = 'Cobyla')
            # low-fidelity dataset names being integers from 0 to level-1
            mfk[i].set_training_values(x_LF_norm, y_LF[:, i:i+1], name = 0)
            # high-fidelity dataset without name
            mfk[i].set_training_values(x_HF_norm, y_HF[:, i:i+1])
            # train the model
            mfk[i].train()
        return mfk
    

    def get_EI_pareto_front(self, mfk, y_all_HF):
        problem = Multi_EI_Optimization_MF(mfk, y_all_HF, self.nv, self.nobj)
        algorithm = NSGA2(pop_size = 100, 
                      n_offsprings = 100,
                      sampling = MixedVariableSampling(),
                      mating = MixedVariableMating(eliminate_duplicates = MixedVariableDuplicateElimination()),
                      eliminate_duplicates = MixedVariableDuplicateElimination())
        res = minimize(problem, algorithm, termination=('n_gen', 100))
        return res
    
    def get_parent_population(self, x_all, x_all_norm, y_all):
        pareto = Pareto(x_all.shape[0], y_all)
        pareto.fast_non_dominate_sort()
        pareto.crowd_distance()
        # shuffle
        pop_index_all = []
        for i in np.arange(len(pareto.f)):
            index_pareto = pareto.f[i]
            np.random.shuffle(index_pareto)
            pop_index_all.append(index_pareto)

        pop_index = sum(pop_index_all, [])[0:self.npop]
        x_pp = x_all[pop_index, :]
        x_pp_norm = x_all_norm[pop_index, :]
        y_pp = y_all[pop_index, :]
        return x_pp, x_pp_norm, y_pp
    
    def evolutionary_operator(self, x_pp, x_pp_norm, y_pp):
        # find the best individual
        pareto = Pareto(x_pp.shape[0], y_pp)
        pareto.fast_non_dominate_sort()
        pareto.crowd_distance()
        i_best0 = pareto.f[0]
        i_best = np.random.choice(i_best0)
        
        x_os_rand_1 = np.zeros((x_pp.shape[0], x_pp.shape[1]))
        x_os_best_1 = np.zeros((x_pp.shape[0], x_pp.shape[1]))
        x_os_rand_2 = np.zeros((x_pp.shape[0], x_pp.shape[1]))
        x_os_best_2 = np.zeros((x_pp.shape[0], x_pp.shape[1]))
        x_os_current2rand_1 = np.zeros((x_pp.shape[0], x_pp.shape[1]))
        x_os_current2best_1 = np.zeros((x_pp.shape[0], x_pp.shape[1]))

        # DE/rand/1
        for i in np.arange(x_pp.shape[0]):
            # mutation
            n1 = np.random.choice(x_pp.shape[0], 3, replace = False)
            x_os_rand_1[i] = x_pp[n1[0]] + self.F*(x_pp[n1[1]] - x_pp[n1[2]])
            for j in np.arange(x_pp.shape[1]):
                # crossover
                n_rand = random.random()
                n_cro = np.random.choice(x_pp.shape[1])
                if (n_rand <= self.CR) | (j == n_cro):
                    x_os_rand_1[i, j] = x_os_rand_1[i, j]
                else:
                    x_os_rand_1[i, j] = x_pp[i, j]

            x_os_rand_1[i] = np.minimum(x_os_rand_1[i], self.ub)
            x_os_rand_1[i] = np.maximum(x_os_rand_1[i], self.lb)
        
        # DE/best/1
        for i in np.arange(x_pp.shape[0]):
            # mutation
            n1 = np.random.choice(x_pp.shape[0], 2, replace = False)
            x_os_best_1[i] = x_pp[i_best] + self.F*(x_pp[n1[0]] - x_pp[n1[1]])
            for j in np.arange(x_pp.shape[1]):
                # crossover
                n_rand = random.random()
                n_cro = np.random.choice(x_pp.shape[1])
                if (n_rand <= self.CR) | (j == n_cro):
                    x_os_best_1[i, j] = x_os_best_1[i, j]
                else:
                    x_os_best_1[i, j] = x_pp[i, j]

            x_os_best_1[i] = np.minimum(x_os_best_1[i], self.ub)
            x_os_best_1[i] = np.maximum(x_os_best_1[i], self.lb)

        # DE/rand/2
        for i in np.arange(x_pp.shape[0]):
            # mutation
            n1 = np.random.choice(x_pp.shape[0], 5, replace = False)
            x_os_rand_2[i] = x_pp[n1[0]] + self.F*(x_pp[n1[1]] - x_pp[n1[2]]) + self.F*(x_pp[n1[3]] - x_pp[n1[4]])
            for j in np.arange(x_pp.shape[1]):
                # crossover
                n_rand = random.random()
                n_cro = np.random.choice(x_pp.shape[1])
                if (n_rand <= self.CR) | (j == n_cro):
                    x_os_rand_2[i, j] = x_os_rand_2[i, j]
                else:
                    x_os_rand_2[i, j] = x_pp[i, j]

            x_os_rand_2[i] = np.minimum(x_os_rand_2[i], self.ub)
            x_os_rand_2[i] = np.maximum(x_os_rand_2[i], self.lb)

        # DE/best/2
        for i in np.arange(x_pp.shape[0]):
            # mutation
            n1 = np.random.choice(x_pp.shape[0], 4, replace = False)
            x_os_best_2[i] = x_pp[i_best] + self.F*(x_pp[n1[0]] - x_pp[n1[1]]) + self.F*(x_pp[n1[2]] - x_pp[n1[3]])
            for j in np.arange(x_pp.shape[1]):
                # crossover
                n_rand = random.random()
                n_cro = np.random.choice(x_pp.shape[1])
                if (n_rand <= self.CR) | (j == n_cro):
                    x_os_best_2[i, j] = x_os_best_2[i, j]
                else:
                    x_os_best_2[i, j] = x_pp[i, j]

            x_os_best_2[i] = np.minimum(x_os_best_2[i], self.ub)
            x_os_best_2[i] = np.maximum(x_os_best_2[i], self.lb)

        # DE/current-to-rand/1
        for i in np.arange(x_pp.shape[0]):
            # mutation
            n1 = np.random.choice(x_pp.shape[0], 3, replace = False)
            x_os_current2rand_1[i] = x_pp[i] + self.F*(x_pp[n1[0]] - x_pp[i]) + self.F*(x_pp[n1[1]] - x_pp[n1[2]])
            for j in np.arange(x_pp.shape[1]):
                # crossover
                n_rand = random.random()
                n_cro = np.random.choice(x_pp.shape[1])
                if (n_rand <= self.CR) | (j == n_cro):
                    x_os_current2rand_1[i, j] = x_os_current2rand_1[i, j]
                else:
                    x_os_current2rand_1[i, j] = x_pp[i, j]

            x_os_current2rand_1[i] = np.minimum(x_os_current2rand_1[i], self.ub)
            x_os_current2rand_1[i] = np.maximum(x_os_current2rand_1[i], self.lb)

        # DE/current-to-best/1
        for i in np.arange(x_pp.shape[0]):
            # mutation
            n1 = np.random.choice(x_pp.shape[0], 2, replace = False)
            x_os_current2best_1[i] = x_pp[i] + self.F*(x_pp[i_best] - x_pp[i]) + self.F*(x_pp[n1[0]] - x_pp[n1[1]])
            for j in np.arange(x_pp.shape[1]):
                # crossover
                n_rand = random.random()
                n_cro = np.random.choice(x_pp.shape[1])
                if (n_rand <= self.CR) | (j == n_cro):
                    x_os_current2best_1[i, j] = x_os_current2best_1[i, j]
                else:
                    x_os_current2best_1[i, j] = x_pp[i, j]

            x_os_current2best_1[i] = np.minimum(x_os_current2best_1[i], self.ub)
            x_os_current2best_1[i] = np.maximum(x_os_current2best_1[i], self.lb)

        x_offspring = np.vstack([x_os_rand_1, x_os_best_1, x_os_rand_2, x_os_best_2, x_os_current2rand_1, x_os_current2best_1])
        x_offspring_norm = (x_offspring - self.lb)/(self.ub - self.lb)
        return x_offspring, x_offspring_norm
    

    def find_near_individual(self, x_pp, x_pp_norm, y_pp, x_all_HF, x_all_norm_HF, y_all_HF, x_all_LF, x_all_norm_LF, y_all_LF):
        pareto = Pareto(x_pp.shape[0], y_pp)
        pareto.fast_non_dominate_sort()
        pareto.crowd_distance()

        if (len(pareto.f[0]) == 1) & (len(pareto.f[1]) == 1):
            x_base = np.vstack([x_pp[pareto.f[0]], x_pp[pareto.f[1]], x_pp[np.random.choice(pareto.f[2])]])
        elif (len(pareto.f[0]) == 1) & (len(pareto.f[1]) > 1):
            x_base = np.vstack([x_pp[pareto.f[0]], x_pp[np.random.choice(pareto.f[1], 2, replace = False)]])
        elif len(pareto.f[0]) == 2:
            x_base = np.vstack([x_pp[pareto.f[0]], x_pp[np.random.choice(pareto.f[1])]])
        elif len(pareto.f[0]) > 2:
            x_base = x_pp[np.random.choice(pareto.f[0], 3, replace = False)]
        
        # find near HF individuals
        x_near_HF = np.zeros((0, self.nv))
        x_near_norm_HF = np.zeros((0, self.nv))
        y_near_HF = np.zeros((0, self.nobj))
        for i in np.arange(x_base.shape[0]):
            dis = cdist(x_base[i].reshape(1, -1), x_all_HF, metric = 'euclidean')
            arg_dis = np.argsort(dis)[0][0:self.nnear]
            x_near_HF = np.vstack([x_near_HF, x_all_HF[arg_dis]])
            x_near_norm_HF = np.vstack([x_near_norm_HF, x_all_norm_HF[arg_dis]])
            y_near_HF = np.vstack([y_near_HF, y_all_HF[arg_dis]])
        
        xy_near_HF = np.hstack([x_near_HF, x_near_norm_HF, y_near_HF])
        # xy_near, index1 = np.unique(xy_near, axis = 0, return_index = True)
        xy_near_HF = np.unique(xy_near_HF, axis = 0)

        x_near_HF = xy_near_HF[:, 0:self.nv]
        x_near_norm_HF = xy_near_HF[:, self.nv:2*self.nv]
        y_near_HF = xy_near_HF[:, 2*self.nv:]

        # find near LF individuals
        x_near_LF = np.zeros((0, self.nv))
        x_near_norm_LF = np.zeros((0, self.nv))
        y_near_LF = np.zeros((0, self.nobj))
        for i in np.arange(x_base.shape[0]):
            dis = cdist(x_base[i].reshape(1, -1), x_all_LF, metric = 'euclidean')
            arg_dis = np.argsort(dis)[0][0:2*self.nnear]
            x_near_LF = np.vstack([x_near_LF, x_all_LF[arg_dis]])
            x_near_norm_LF = np.vstack([x_near_norm_LF, x_all_norm_LF[arg_dis]])
            y_near_LF = np.vstack([y_near_LF, y_all_LF[arg_dis]])
        
        xy_near_LF = np.hstack([x_near_LF, x_near_norm_LF, y_near_LF])
        # xy_near, index1 = np.unique(xy_near, axis = 0, return_index = True)
        xy_near_LF = np.unique(xy_near_LF, axis = 0)

        x_near_LF = xy_near_LF[:, 0:self.nv]
        x_near_norm_LF = xy_near_LF[:, self.nv:2*self.nv]
        y_near_LF = xy_near_LF[:, 2*self.nv:]
        return x_near_HF, x_near_norm_HF, y_near_HF, x_near_LF, x_near_norm_LF, y_near_LF
    

    # def EI_function(self, x_norm, x_all, x_all_norm, y_all, gpr):
    #     EI_value = np.zeros((self.nobj, 1))
    #     for i in np.arange(self.nobj):
    #         y_min = np.min(y_all[:, i])
    #         y_predmean, y_predstd = gpr[i].predict(x_norm, return_std = True)
    #         EI_value[i] = (y_min - y_predmean)*norm.cdf((y_min - y_predmean)/y_predstd) + y_predstd*norm.pdf((y_min - y_predmean)/y_predstd)
    #     return EI_value


    def optimization_iteration(self, init_channels, depth, epochs_HF, epochs_LF, save_dir):
        start = time.time()

        current_nfe = 0
        x_init_HF, x_init_LF, x_init_norm_HF, x_init_norm_LF = self.initial_sampling()
        y_init_HF, y_init_LF = self.objective_evaluate(x_init_HF, x_init_LF, current_nfe, init_channels, depth, epochs_HF, epochs_LF, save_dir)

        # train global surrogate
        mfk_global = self.surrogate_prediction(x_init_norm_HF, x_init_norm_LF, y_init_HF, y_init_LF)

        # iteration
        current_nfe = current_nfe + self.ns
        x_all_HF = x_init_HF
        x_all_norm_HF = x_init_norm_HF
        y_all_HF = y_init_HF
        x_all_LF = x_init_LF
        x_all_norm_LF = x_init_norm_LF
        y_all_LF = y_init_LF


        
        # iteration with existing samples (need to change runtime)
        # file1 = np.load('/home/weiz/Astar_Work/NAS/NAS-Unet/results/MF_SADE/CHASE_DB1/CHASE_DB1_MF_SADE_100.npz', allow_pickle = True)
        # x_all_norm_HF, x_all_HF, y_all_HF, x_all_norm_LF, x_all_LF, y_all_LF, mfk_global, runtime_MF_SADE = file1['arr_0'], file1['arr_1'], file1['arr_2'], file1['arr_3'], file1['arr_4'], file1['arr_5'], file1['arr_6'], file1['arr_7']
        # mfk_global = self.surrogate_prediction(x_all_norm_HF, x_all_norm_LF, y_all_HF, y_all_LF)
        # current_nfe = x_all_norm_HF.shape[0]


        while current_nfe < self.nfe_max:
            # parent population
            x_pp, x_pp_norm, y_pp = self.get_parent_population(x_all_HF, x_all_norm_HF, y_all_HF)
            x_offspring, x_offspring_norm = self.evolutionary_operator(x_pp, x_pp_norm, y_pp)

            # prediction for offspring
            y_offspring_predmean = np.zeros((x_offspring.shape[0], self.nobj))
            y_offspring_predmse = np.zeros((x_offspring.shape[0], self.nobj))
            for i in np.arange(self.nobj):
                y_offspring_predmean[:, i:i+1] = mfk_global[i].predict_values(x_offspring_norm)
                y_offspring_predmse[:, i:i+1] = mfk_global[i].predict_variances(x_offspring_norm)

            # get the Pareto front of prediction of multi-objectives
            pareto = Pareto(x_offspring.shape[0], y_offspring_predmean)
            pareto.fast_non_dominate_sort()
            pareto.crowd_distance()

            # select the best individual (HF)  1
            i_best_HF = np.random.choice(pareto.f[0])
            x_best_HF = x_offspring[i_best_HF].reshape(1, -1)
            x_best_norm_HF = x_offspring_norm[i_best_HF].reshape(1, -1)

            # select the best individual (LF)  2
            if len(pareto.f[0]) == 1:
                i_best_LF = pareto.f[0]
                i_best_LF.append(np.random.choice(pareto.f[1]))
            else:
                i_best_LF = np.random.choice(pareto.f[0], 2, replace = False)
            x_best_LF = x_offspring[i_best_LF]
            x_best_norm_LF = x_offspring_norm[i_best_LF]

            y_best_HF, y_best_LF = self.objective_evaluate(x_best_HF, x_best_LF, current_nfe, init_channels, depth, epochs_HF, epochs_LF, save_dir)

            # update database
            x_all_HF = np.vstack([x_all_HF, x_best_HF])
            x_all_norm_HF = np.vstack([x_all_norm_HF, x_best_norm_HF])
            y_all_HF = np.vstack([y_all_HF, y_best_HF])
            x_all_LF = np.vstack([x_all_LF, x_best_LF])
            x_all_norm_LF = np.vstack([x_all_norm_LF, x_best_norm_LF])
            y_all_LF = np.vstack([y_all_LF, y_best_LF])
            current_nfe = current_nfe + 1

            # find near individuals near best individual in parent population
            x_near_HF, x_near_norm_HF, y_near_HF, x_near_LF, x_near_norm_LF, y_near_LF = self.find_near_individual(x_pp, x_pp_norm, y_pp, x_all_HF, x_all_norm_HF, y_all_HF, x_all_LF, x_all_norm_LF, y_all_LF)

            # train local surrogate
            mfk_local = self.surrogate_prediction(x_near_norm_HF, x_near_norm_LF, y_near_HF, y_near_LF)

            # get the Pareto front of EI_1, EI_2, ...
            res_EI = self.get_EI_pareto_front(mfk_local, y_all_HF)

            # plot EI Pareto front
            # PF_data_NSGA_II = res.F
            # PF_data_NSGA_II_sort = PF_data_NSGA_II[np.lexsort(PF_data_NSGA_II.T)]
            # fig = plt.figure(1)
            # # plt.plot(y_all[:, 0], y_all[:, 1], 'o')
            # plt.plot(PF_data_NSGA_II_sort[:, 0], PF_data_NSGA_II_sort[:, 1], '-k*')
            # plt.xlabel('$\mathit{f}_1$', fontsize = 14)
            # plt.ylabel('$\mathit{f}_2$', fontsize = 14)
            # # plt.xscale('log')
            # # plt.yscale('log')
            # plt.xticks(fontsize = 12)
            # plt.yticks(fontsize = 12)
            # plt.legend(fontsize = 12)
            # plt.grid()
            # plt.show()


            # k-means clustering
            x_pareto_front_EI = np.zeros((len(res_EI.X), self.nv))
            for i in range(len(res_EI.X)):
                x0 = res_EI.X[i]
                x_pareto_front_EI[i] = np.array([x0[f"x{k:02}"] for k in range(len(x0))])

            # print(x_pareto_front_EI.shape[0])
            
            # number of HF infill samples
            if x_pareto_front_EI.shape[0] == 1:
                n_clusters = 1
            elif x_pareto_front_EI.shape[0] == 2:
                n_clusters = 2
            else:
                n_clusters = 3

            # number of LF infill samples
            if x_pareto_front_EI.shape[0] <= 6:
                n_add_LF = x_pareto_front_EI.shape[0]
            else:
                n_add_LF = 6

            clustering_results = k_means(x_pareto_front_EI, n_clusters, init = 'k-means++', random_state = 0)

            # select the cluster center for HF infill sampling
            x_add_norm_HF = clustering_results[0]
            x_add_HF = self.lb + x_add_norm_HF * (self.ub - self.lb)

            # select other Pareto set for LF infill sampling
            x_add_norm_LF = x_pareto_front_EI[np.random.choice(np.arange(x_pareto_front_EI.shape[0]), n_add_LF, replace = False)]
            x_add_LF = self.lb + x_add_norm_LF * (self.ub - self.lb)
            y_add_HF, y_add_LF = self.objective_evaluate(x_add_HF, x_add_LF, current_nfe, init_channels, depth, epochs_HF, epochs_LF, save_dir)

            # update database
            x_all_HF = np.vstack([x_all_HF, x_add_HF])
            x_all_norm_HF = np.vstack([x_all_norm_HF, x_add_norm_HF])
            y_all_HF = np.vstack([y_all_HF, y_add_HF])
            x_all_LF = np.vstack([x_all_LF, x_add_LF])
            x_all_norm_LF = np.vstack([x_all_norm_LF, x_add_norm_LF])
            y_all_LF = np.vstack([y_all_LF, y_add_LF])

            current_nfe = current_nfe + n_clusters
            print('current_nfe = {:3d}'.format(current_nfe))
            # print('lambda {:>.3f}: {:>5} {:>8} {:10.6f} {:10.6f} {:10.6f} {:10.6f}'.format(lmbda, i_s, train_iters, loss_all, ic_loss, pde_loss, mse))
            
            # update surrogate
            mfk_global = self.surrogate_prediction(x_all_norm_HF, x_all_norm_LF, y_all_HF, y_all_LF)

        end = time.time()
        runtime = end - start
        # runtime = end - start + runtime_MF_SADE
        return x_all_norm_HF, x_all_HF, y_all_HF, x_all_norm_LF, x_all_LF, y_all_LF, mfk_global, runtime




class Multi_EI_Optimization(Problem):
    def __init__(self, gpr, y_all, nv, nobj):
        variables = dict()
        for i in np.arange(nv):
            variables[f'x{i:02}'] = Real(bounds = (0, 1))

        super().__init__(vars = variables, n_var = nv, n_obj = nobj, n_ieq_constr = 0, n_eq_constr = 0)
        self.gpr = gpr
        self.y_all = y_all
        self.nv = nv
        self.nobj = nobj
    

    def _evaluate(self, x, out, *args, **kwargs):
        objs = np.zeros((x.shape[0], self.nobj))
        for i in np.arange(x.shape[0]):
            x0 = x[i]
            x_inputs = np.array([x0[f"x{k:02}"] for k in range(len(x0))])
            x_inputs = x_inputs.reshape(1, self.nv)
            
            for j in np.arange(self.nobj):
                y_min = np.min(self.y_all[:, j])
                y_predmean, y_predstd = self.gpr[j].predict(x_inputs, return_std = True)
                EI_value = (y_min - y_predmean)*norm.cdf((y_min - y_predmean)/y_predstd) + y_predstd*norm.pdf((y_min - y_predmean)/y_predstd)
                objs[i, j] = -EI_value

        out["F"] = objs




class Multi_EI_Optimization_MF(Problem):
    def __init__(self, mfk, y_all_HF, nv, nobj):
        variables = dict()
        for i in np.arange(nv):
            variables[f'x{i:02}'] = Real(bounds = (0, 1))

        super().__init__(vars = variables, n_var = nv, n_obj = nobj, n_ieq_constr = 0, n_eq_constr = 0)
        self.mfk = mfk
        self.y_all_HF = y_all_HF
        self.nv = nv
        self.nobj = nobj
    

    def _evaluate(self, x, out, *args, **kwargs):
        objs = np.zeros((x.shape[0], self.nobj))
        for i in np.arange(x.shape[0]):
            x0 = x[i]
            x_inputs = np.array([x0[f"x{k:02}"] for k in range(len(x0))])
            x_inputs = x_inputs.reshape(1, self.nv)
            
            for j in np.arange(self.nobj):
                y_min = np.min(self.y_all_HF[:, j])
                y_predmean = self.mfk[j].predict_values(x_inputs)
                y_predmse = self.mfk[j].predict_variances(x_inputs)
                y_predstd = np.sqrt(y_predmse)
                EI_value = (y_min - y_predmean)*norm.cdf((y_min - y_predmean)/y_predstd) + y_predstd*norm.pdf((y_min - y_predmean)/y_predstd)
                objs[i, j] = -EI_value

        out["F"] = objs




class NSGA_II_Optimization(Problem):
    def __init__(self, nv, nobj, lb, ub, current_nfe, init_channels, depth, epochs, save_dir):
        variables = dict()
        for i in np.arange(nv):
            variables[f'x{i:02}'] = Real(bounds = (lb[i], ub[i]))

        super().__init__(vars = variables, n_var = nv, n_obj = nobj, n_ieq_constr = 0, n_eq_constr = 0)
        self.nv = nv
        self.nobj = nobj
        self.current_nfe = current_nfe
        self.init_channels = init_channels
        self.depth = depth
        self.epochs = epochs
        self.save_dir = save_dir
    

    def _evaluate(self, x, out, *args, **kwargs):
        objs = np.full((x.shape[0], self.nobj), np.nan)
        for i in range(x.shape[0]):
            x0 = x[i]
            x_inputs = np.array([x0[f"x{k:02}"] for k in range(len(x0))])
            x_inputs = x_inputs.reshape(1, self.nv)
            y = ZDT1(x_inputs, self.nv)
            objs[i] = y

            # # model evaluation
            # genome = micro_encoding.convert_continuous_variables(x_inputs)
            # performance = train_search.main(x_inputs = x_inputs,
            #                                 genome = genome,
            #                                 search_space = 'micro',
            #                                 init_channels = self.init_channels,
            #                                 depth = self.depth,
            #                                 layers = 11, cutout=False,
            #                                 epochs = self.epochs,
            #                                 save = 'arch_{}'.format(self.current_nfe + i + 1),
            #                                 expr_root = self.save_dir)
            # objs[i, 0] = 100 - performance['valid_acc']
            # objs[i, 1] = performance['flops']
        
        self.current_nfe = self.current_nfe + x.shape[0]

        out["F"] = objs




class NSGA_II(object):

    def __init__(self, nv, nobj, lb, ub, current_nfe, init_channels, depth, epochs, save_dir):

        self.nv = nv  # dimentionality of optimization problem
        self.nobj = nobj  # number of objectives
        self.lb = lb  # lower bound
        self.ub = ub  # upper bound
        self.current_nfe = current_nfe
        self.init_channels = init_channels
        self.depth = depth
        self.epochs = epochs
        self.save_dir = save_dir


    def main(self):
        problem = NSGA_II_Optimization(self.nv, self.nobj, self.lb, self.ub, self.current_nfe, self.init_channels, self.depth, self.epochs, self.save_dir)
        algorithm = NSGA2(pop_size = 50, 
                      n_offsprings = 50,
                      sampling = MixedVariableSampling(),
                      mating = MixedVariableMating(eliminate_duplicates = MixedVariableDuplicateElimination()),
                      eliminate_duplicates = MixedVariableDuplicateElimination())
        res = minimize(problem, algorithm, termination=('n_gen', 6))
        return res




class MO_ACK(object):

    def __init__(self, nv, ns, nobj, nfe_max, lb, ub, tolerance):

        self.nv = nv  # dimentionality of optimization problem
        self.ns = ns  # number of initial samples
        self.nobj = nobj  # number of objectives
        self.nfe_max = nfe_max  # maximum number of function evaluations
        self.tolerance = tolerance  # maximum tolerance
        self.lb = lb  # lower bound
        self.ub = ub  # upper bound


    def initial_sampling(self):
        xlimits = np.hstack([np.zeros((self.nv, 1)), np.ones((self.nv, 1))])
        xdoes = NestedLHS(nlevel = 2, xlimits = xlimits, random_state = np.random.randint(0, 1e6))
        x_LF_init_norm, x_HF_init_norm = xdoes(self.ns)
        x_LF_init = self.lb + x_LF_init_norm * (self.ub - self.lb)
        x_HF_init = self.lb + x_HF_init_norm * (self.ub - self.lb)
        return x_HF_init, x_LF_init, x_HF_init_norm, x_LF_init_norm
    
    
    def objective_evaluate(self, x_HF, x_LF, current_nfe, init_channels, depth, epochs_HF, epochs_LF, save_dir):
        # y =  3*(1-x[:, 0:1])**2*np.exp(-(x[:, 0:1]**2) - (x[:, 1:2]+1)**2) - 10*(x[:, 0:1]/5 - x[:, 0:1]**3 - x[:, 1:2]**5)*np.exp(-x[:, 0:1]**2-x[:, 1:2]**2) - 1/3*np.exp(-(x[:, 0:1]+1)**2 - x[:, 1:2]**2)  # PK function
        # y1 = x[:, 0:1]**4 - 10*x[:, 0:1]**2 + x[:, 0:1]*x[:, 1:2] + x[:, 1:2]**4 - (x[:, 0:1]**2)*(x[:, 1:2]**2)
        # y2 = x[:, 1:2]**4 - (x[:, 0:1]**2)*(x[:, 1:2]**2) + x[:, 0:1]**4 + x[:, 0:1]*x[:, 1:2]
        # y = np.hstack([y1, y2])

        # y_HF = ZDT1(x_HF, self.nv)
        # y_LF = ZDT1_LF(x_LF, self.nv)

        # NAS evulation
        y_HF = np.zeros((x_HF.shape[0], self.nobj))
        y_LF = np.zeros((x_LF.shape[0], self.nobj))

        # HF evaluation
        for i in np.arange(x_HF.shape[0]):
            x_inputs_HF = x_HF[i]
            genome_HF = micro_encoding.convert_continuous_variables(x_inputs_HF)

            # PDE
            gpu = 1
            # performance = train_PDE_search.main(x_inputs = x_inputs_HF,
            #                                 genome = genome_HF,
            #                                 search_space = 'micro',
            #                                 gpu = gpu,
            #                                 init_channels = init_channels,
            #                                 depth = depth,
            #                                 epochs = epochs_HF,
            #                                 save = 'arch_{}'.format(current_nfe + i + 1),
            #                                 expr_root = save_dir)
            # y_HF[i, 0] = performance['valid_RMSE']
            # y_HF[i, 1] = performance['flops']

            # CHASE_DB1
            performance = train_chase_search.main(x_inputs = x_inputs_HF,
                                            genome = genome_HF,
                                            search_space = 'micro',
                                            gpu = gpu,
                                            init_channels = init_channels,
                                            depth = depth,
                                            epochs = epochs_HF,
                                            save = 'arch_{}'.format(current_nfe + i + 1),
                                            expr_root = save_dir)
            y_HF[i, 0] = 1 - performance['valid_auc_roc']
            y_HF[i, 1] = performance['flops']


        # LF evaluation
        for i in np.arange(x_LF.shape[0]):
            x_inputs_LF = x_LF[i]
            genome_LF = micro_encoding.convert_continuous_variables(x_inputs_LF)

            # PDE
            gpu = 1
            # performance = train_PDE_search_LF.main(x_inputs = x_inputs_LF,
            #                             genome = genome_LF,
            #                             search_space = 'micro',
            #                             gpu = gpu,
            #                             init_channels = init_channels,
            #                             depth = depth,
            #                             epochs = epochs_LF,
            #                             save = 'arch_{}'.format(0),
            #                             expr_root = save_dir)
            # y_LF[i, 0] = performance['valid_RMSE']
            # y_LF[i, 1] = performance['flops']

            # CHASE_DB1
            performance = train_chase_search_LF.main(x_inputs = x_inputs_LF,
                                        genome = genome_LF,
                                        search_space = 'micro',
                                        gpu = gpu,
                                        init_channels = init_channels,
                                        depth = depth,
                                        epochs = epochs_LF,
                                        save = 'arch_{}'.format(0),
                                        expr_root = save_dir)
            y_LF[i, 0] = 1 - performance['valid_auc_roc']
            y_LF[i, 1] = performance['flops']

        return y_HF, y_LF
    


    def surrogate_prediction(self, x_HF_norm, x_LF_norm, y_HF, y_LF):
        mfk = {}
        for i in np.arange(self.nobj):
            mfk[i] = MFK(poly = 'constant', corr = 'squar_exp', theta0 = self.nv*[0.01], n_start = 50, hyper_opt = 'Cobyla')
            # low-fidelity dataset names being integers from 0 to level-1
            mfk[i].set_training_values(x_LF_norm, y_LF[:, i:i+1], name = 0)
            # high-fidelity dataset without name
            mfk[i].set_training_values(x_HF_norm, y_HF[:, i:i+1])
            # train the model
            mfk[i].train()
        return mfk
    

    def get_surrogate_pareto_front(self, mfk, y_all_HF):
        problem = Multi_Obj_Optimization_Surrogate(mfk, y_all_HF, self.nv, self.nobj)
        algorithm = NSGA2(pop_size = 100, 
                      n_offsprings = 100,
                      sampling = MixedVariableSampling(),
                      mating = MixedVariableMating(eliminate_duplicates = MixedVariableDuplicateElimination()),
                      eliminate_duplicates = MixedVariableDuplicateElimination())
        res = minimize(problem, algorithm, termination=('n_gen', 100), save_history=True)
        return res
    


    def optimization_iteration(self, init_channels, depth, epochs_HF, epochs_LF, save_dir):
        start = time.time()
        current_nfe = 0
        x_init_HF, x_init_LF, x_init_norm_HF, x_init_norm_LF = self.initial_sampling()
        y_init_HF, y_init_LF = self.objective_evaluate(x_init_HF, x_init_LF, current_nfe, init_channels, depth, epochs_HF, epochs_LF, save_dir)

        # train global surrogate
        mfk_global = self.surrogate_prediction(x_init_norm_HF, x_init_norm_LF, y_init_HF, y_init_LF)

        # iteration
        current_nfe = current_nfe + self.ns
        x_all_HF = x_init_HF
        x_all_norm_HF = x_init_norm_HF
        y_all_HF = y_init_HF
        x_all_LF = x_init_LF
        x_all_norm_LF = x_init_norm_LF
        y_all_LF = y_init_LF



        # # iteration with existing samples (need to change runtime)
        # file1 = np.load('/home/weiz/Astar_Work/NAS/NAS-Unet1/search/DarcyFlow_MO_ACK_depth_4_nfe_100_1.npz', allow_pickle = True)
        # x_all_norm_HF, x_all_HF, y_all_HF, x_all_norm_LF, x_all_LF, y_all_LF, mfk_global = file1['arr_0'], file1['arr_1'], file1['arr_2'], file1['arr_3'], file1['arr_4'], file1['arr_5'], file1['arr_6']
        # mfk_global = self.surrogate_prediction(x_all_norm_HF, x_all_norm_LF, y_all_HF, y_all_LF)
        # current_nfe = x_all_norm_HF.shape[0]


        while current_nfe < self.nfe_max:
            # get the Pareto front of surrogates
            res_surrogate = self.get_surrogate_pareto_front(mfk_global, y_all_HF)

            # k-means clustering
            x_pareto_front_surrogate = np.zeros((len(res_surrogate.X), self.nv))
            for i in range(len(res_surrogate.X)):
                x0 = res_surrogate.X[i]
                x_pareto_front_surrogate[i] = np.array([x0[f"x{k:02}"] for k in range(len(x0))])
            
            if x_pareto_front_surrogate.shape[0] == 1:
                n_clusters = 1
            elif x_pareto_front_surrogate.shape[0] == 2:
                n_clusters = 2
            else:
                n_clusters = 3

            clustering_results = k_means(x_pareto_front_surrogate, n_clusters, init = 'k-means++', random_state = 0)
    
            for i in np.arange(n_clusters):
                i_cluster = np.where(clustering_results[1] == i)
                x_cluster = x_pareto_front_surrogate[i_cluster[0]]
                y_cluster = np.zeros((x_cluster.shape[0], self.nobj))
                for j in np.arange(self.nobj):
                    y_cluster[:, j:j+1] = mfk_global[j].predict_values(x_cluster)

                if x_cluster.shape[0] > 1:
                    G = np.zeros((x_cluster.shape[0], 1))
                    MG = np.zeros((x_cluster.shape[0], 1))
                    for k in np.arange(x_cluster.shape[0]):
                        y_cluster_temp = np.delete(y_cluster, k, axis = 0)
                        y_cluster_delta = np.zeros((x_cluster.shape[0] - 1, self.nobj))
                        for m in np.arange(self.nobj):
                            y_cluster_delta[:, m:m+1] = y_cluster[k, m] - y_cluster_temp[:, m:m+1]
                        
                        G[k] = 1 - np.max(np.min(y_cluster_delta, axis=0))
                        if G[k] < 1:
                            MG[k] = 1 - G[k] + np.max(G)
                        else:
                            MG[k] = G[k]
                    
                    # HF infill sampling
                    i_min_MG = np.argmin(MG)
                    x_add_norm_HF = x_cluster[i_min_MG].reshape(1, -1)
                    x_add_HF = self.lb + x_add_norm_HF * (self.ub - self.lb)
                else:
                    # HF infill sampling
                    x_add_norm_HF = x_cluster.reshape(1, -1)
                    x_add_HF = self.lb + x_add_norm_HF * (self.ub - self.lb)

                # LF infill sampling
                if x_cluster.shape[0] <= 1:
                    x_add_norm_LF = x_cluster
                else:
                    x_add_norm_LF = x_cluster[np.random.choice(np.arange(x_cluster.shape[0]), 2, replace = False)]

                x_add_LF = self.lb + x_add_norm_LF * (self.ub - self.lb)
                y_add_HF, y_add_LF = self.objective_evaluate(x_add_HF, x_add_LF, current_nfe, init_channels, depth, epochs_HF, epochs_LF, save_dir)

                # update database
                x_all_HF = np.vstack([x_all_HF, x_add_HF])
                x_all_norm_HF = np.vstack([x_all_norm_HF, x_add_norm_HF])
                y_all_HF = np.vstack([y_all_HF, y_add_HF])
                x_all_LF = np.vstack([x_all_LF, x_add_LF])
                x_all_norm_LF = np.vstack([x_all_norm_LF, x_add_norm_LF])
                y_all_LF = np.vstack([y_all_LF, y_add_LF])

                current_nfe = current_nfe + 1
                print('current_nfe = {:3d}'.format(current_nfe))
            # print('lambda {:>.3f}: {:>5} {:>8} {:10.6f} {:10.6f} {:10.6f} {:10.6f}'.format(lmbda, i_s, train_iters, loss_all, ic_loss, pde_loss, mse))
            
            # update surrogate
            mfk_global = self.surrogate_prediction(x_all_norm_HF, x_all_norm_LF, y_all_HF, y_all_LF)

        end = time.time()
        runtime = end - start
        # runtime = end - start + runtime_MF_SADE
        return x_all_norm_HF, x_all_HF, y_all_HF, x_all_norm_LF, x_all_LF, y_all_LF, mfk_global, runtime




class Multi_Obj_Optimization_Surrogate(Problem):
    def __init__(self, mfk, y_all, nv, nobj):
        variables = dict()
        for i in np.arange(nv):
            variables[f'x{i:02}'] = Real(bounds = (0, 1))

        super().__init__(vars = variables, n_var = nv, n_obj = nobj, n_ieq_constr = 0, n_eq_constr = 0)
        self.mfk = mfk
        self.y_all = y_all
        self.nv = nv
        self.nobj = nobj
    

    def _evaluate(self, x, out, *args, **kwargs):
        objs = np.zeros((x.shape[0], self.nobj))
        for i in np.arange(x.shape[0]):
            x0 = x[i]
            x_inputs = np.array([x0[f"x{k:02}"] for k in range(len(x0))])
            x_inputs = x_inputs.reshape(1, self.nv)
            
            for j in np.arange(self.nobj):
                y_predmean = self.mfk[j].predict_values(x_inputs)
                y_predmse = self.mfk[j].predict_variances(x_inputs)
                y_predstd = np.sqrt(y_predmse)
                objs[i, j] = y_predmean

        out["F"] = objs




if __name__ == "__main__":
    nv = 12
    ns = 50  # 50
    npop = 50  # int(min(100, 5*nv))
    nnear = 30  # int(min((nv + 1)*(nv + 2)/2, 30))
    F = 0.8
    CR = 0.8
    nobj = 2
    nfe_max = 200
    nfe_max_MF = 80
    # a1 = np.array([0, -10, -10, -10, -10, -10, -10, -10, -10, -10])
    # a2 = np.array([1, 10, 10, 10, 10, 10, 10, 10, 10, 10])
    # lb = a1*np.ones((nv,))
    # ub = a2*np.ones((nv,))
    lb = 0*np.ones((nv,))
    ub = 1*np.ones((nv,))
    tolerance = 0

    init_channels = 0
    depth = 0
    epochs_HF = 0
    epochs_LF = 0
    save_dir = 0
    current_nfe = 0

    # # MF_SADE results
    optimizer = MF_SADE(nv = nv, ns = ns, npop = npop, nnear = nnear, F = F, CR = CR, nobj = nobj, nfe_max = nfe_max_MF, lb = lb, ub = ub, tolerance = tolerance)
    x_all_norm_HF, x_all_HF, y_all_HF, x_all_norm_LF, x_all_LF, y_all_LF, mfk_global, runtime_MF_SADE = optimizer.optimization_iteration(init_channels, depth, epochs_HF, epochs_LF, save_dir)
    pareto = Pareto(x_all_HF.shape[0], y_all_HF)
    pareto.fast_non_dominate_sort()
    pareto.crowd_distance()
    PF_data_HF = np.array([y_all_HF[pareto.f[0], 0], y_all_HF[pareto.f[0], 1]])
    PF_data_HF = PF_data_HF.T
    PF_data_HF_sort = PF_data_HF[np.lexsort(PF_data_HF.T)]

    ref_point = np.array([10, 10])
    HV_MK_SADE = HV(ref_point = ref_point)(y_all_HF)
    PF_true = get_problem('zdt1').pareto_front()
    plt.plot(PF_data_HF_sort[:, 0], PF_data_HF_sort[:, 1], '-b', label = 'MF_SADE Pareto front')
    plt.plot(PF_true[:, 0], PF_true[:, 1], '-gD', label = 'True Pareto Front')
    plt.legend()
    plt.grid()
    plt.show()



    # SADE results
    optimizer = SADE(nv = nv, ns = ns, npop = npop, nnear = nnear, F = F, CR = CR, nobj = nobj, nfe_max = nfe_max, lb = lb, ub = ub, tolerance = tolerance)
    x_all_norm, x_all, y_all, gpr_global, runtime_SADE = optimizer.optimization_iteration(init_channels, depth, epochs_HF, save_dir)
    pareto = Pareto(x_all.shape[0], y_all)
    pareto.fast_non_dominate_sort()
    pareto.crowd_distance()
    PF_data = np.array([y_all[pareto.f[0], 0], y_all[pareto.f[0], 1]])
    PF_data = PF_data.T
    PF_data_sort = PF_data[np.lexsort(PF_data.T)]

    # NSGA-II results
    NSGA_II_optimizer = NSGA_II(nv = nv, nobj = nobj, lb = lb, ub = ub, current_nfe = current_nfe, init_channels = init_channels, depth = depth, epochs = epochs_HF, save_dir = save_dir)
    res = NSGA_II_optimizer.main()
    PF_data_NSGA_II = res.F
    PF_data_NSGA_II_sort = PF_data_NSGA_II[np.lexsort(PF_data_NSGA_II.T)]

    PF_true = get_problem('zdt3').pareto_front()

    # Hypervolume
    ref_point = np.array([10, 10])
    HV_SADE = HV(ref_point = ref_point)(y_all)
    HV_MK_SADE = HV(ref_point = ref_point)(y_all_HF)
    HV_NSGA_II = HV(ref_point = ref_point)(res.F)

    fig = plt.figure(1)
    plt.plot(y_all_HF[:, 0], y_all_HF[:, 1], 'o', label = 'All the HF samples')
    plt.plot(y_all_LF[:, 0], y_all_LF[:, 1], 's', label = 'All the LF samples')
    plt.plot(PF_data_HF_sort[:, 0], PF_data_HF_sort[:, 1], '-b', label = 'MF_SADE Pareto front')
    plt.plot(PF_data_sort[:, 0], PF_data_sort[:, 1], '-rp', label = 'SADE Pareto front')
    plt.plot(PF_data_NSGA_II_sort[:, 0], PF_data_NSGA_II_sort[:, 1], '-k*', label = 'NSGA-II Pareto Front')
    plt.plot(PF_true[:, 0], PF_true[:, 1], '-gD', label = 'True Pareto Front')
    plt.xlabel('$\mathit{f}_1$', fontsize = 14)
    plt.ylabel('$\mathit{f}_2$', fontsize = 14)
    # plt.xscale('log')
    # plt.yscale('log')
    plt.xticks(fontsize = 12)
    plt.yticks(fontsize = 12)
    plt.legend(fontsize = 12)
    plt.grid()
    plt.show()


    


    a
    
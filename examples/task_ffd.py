import os, sys, inspect, time
"""
solve the conflict between paths, and work dir
"""
sys.path.append('../')
#os.chdir('../')
print(sys.path)
print(os.getcwd())

import numpy as np
import torch 
import matplotlib.pyplot as plt
torch.multiprocessing.set_sharing_strategy('file_system')

import lfi
from lfi.utils import discrepancy, visualization
from lfi.algorithms import ABC_algorithms, SMCABC, SMC2ABC, SNLABC, SNL2ABC
from lfi.detsimul import problem_FFD
from lfi.utils import uos, umath




DIR = 'results/FFD' 
IMG_FOLDER = "/home/quinoa/SimpleFace"
OUT_FOLDER = "/home/quinoa/out_folder"
problem = problem_FFD.FFD_Image_Problem(N=500, n=250)
problem.img_folder = IMG_FOLDER
problem.out_img_folder = OUT_FOLDER
problem.true_mean = 0.0
problem.true_var = 10.0
true_theta = problem.get_true_theta()
## The generated images are stored in a folder
## data_obs contains only the deformation parameters
problem.data_obs = problem.simulator(true_theta)
problem.compute_pca(n_components=15)
print(problem.data_obs[0,:],problem.data_obs.shape)
problem.y_obs = problem.statistics(data=problem.data_obs, theta=true_theta)
#uos.save_object(DIR, 'data_obs', problem.data_obs)

true_samples = problem.sample_from_true_posterior()

theta_true = problem.get_true_theta()
ll_true = problem.log_likelihood(theta_true)

theta_wrong = np.array([theta_true[0] + 2,
                        theta_true[1]])
ll_wrong = problem.log_likelihood(theta_wrong)

print(ll_true, ll_wrong)


problem.visualize()
plt.figure(figsize=(5,4))
visualization.plot_likelihood(samples=true_samples,
                              log_likelihood_function=problem.log_likelihood,
                              dimensions=(0,1),
                              bounded = False)#,n_levels=10)
plt.show()
visualization.plot_samples(problem.data_obs)




"""

## Sequential Neural Likelihood
hyperparams = ABC_algorithms.Hyperparams()
hyperparams.save_dir = DIR
#hyperparams.device = 'cuda:1'
hyperparams.device = 'cpu'

hyperparams.num_sim = 4000                       # number of sampling/simulation rounds
hyperparams.L = 2                                # number of learning rounds
#hyperparams.type = 'cnn2d'                       # the network architecture of S(x)
hyperparams.type = 'plain'
hyperparams.stat = 'infomax'                     # statistics function: infomax/moment/score  
hyperparams.estimator = 'DC'                     # MI estimator; JSD (accurate) or DC (fast)
hyperparams.nde = 'MAF'                          # nde; MAF (D>1) or MDN (D=1)
from lfi.utils import discrepancy, visualization

print('\n SNL ABC')
snl_abc = SNL2ABC.SNL2_ABC(problem, discrepancy=discrepancy.eculidean_dist, hyperparams=hyperparams)
snl_abc.run()

JSD_array = []
for l in range(len(snl_abc.nde_array)):
    print('l=', l)
    snl_abc.nde_net = snl_abc.nde_array[l]
    visualization.plot_likelihood(samples=true_samples, log_likelihood_function=snl_abc.log_likelihood, dimensions=(0,1),bounded=False)
    plt.show()
    JSD = discrepancy.JSD(problem.log_likelihood, snl_abc.log_likelihood, true_samples, true_samples, N_grid=30)
    JSD_array.append(JSD)
    print('JSD snl = ', JSD)
uos.save_object(DIR, 'JSD_SNL', JSD_array)
"""
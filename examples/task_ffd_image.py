import os, sys, inspect, time
"""
solve the conflict between paths, and work dir
"""
#sys.path.append('../')
sys.path.append(os.getcwd())
##os.chdir('../')
print(sys.path)
print(os.getcwd())

import numpy as np
import torch 
import matplotlib.pyplot as plt
torch.multiprocessing.set_sharing_strategy('file_system')

import lfi
from lfi.utils import discrepancy, visualization
from lfi.algorithms import ABC_algorithms, SMCABC, SMC2ABC, SNLABC, SNL2ABC
from lfi.detsimul import problem_FFD,problem_FFD_image
from lfi.utils import uos, umath

DIR = 'results/FFD' 
#IMG_FOLDER = "/home/amaranth2/Downloads/SimpleFace"
IMG_FOLDER = "/home/quinoa/SimpleFace"
OUT_FOLDER = "/home/quinoa/out_folder"
problem = problem_FFD_image.FFD_Image_Problem(N=500, n=200)
problem.img_folder = IMG_FOLDER
problem.out_img_folder = OUT_FOLDER
problem.true_mean = 0.0
problem.true_var = 10.0
problem.get_work_imgs()
true_theta = problem.get_true_theta()
## The generated images are stored in a folder
## data_obs contains only the deformation parameters

## The simulator returns the observed mu params and 
## the deformed images
problem.data_obs, problem.imgs_obs = problem.simulator(true_theta)
## Compute PCA over the observed data
problem.compute_pca(n_components=15)
## Pick a mu param
#problem.y_obs, problem.imgs_obs = problem.statistics(data=problem.data_obs,images=problem.imgs_obs, theta=true_theta)
problem.y_obs, _ = problem.statistics(data=problem.data_obs,images=problem.imgs_obs, theta=true_theta)

## Sample thetas from the true posterior
true_samples = problem.sample_from_true_posterior()
## Get the true theta
theta_true = problem.get_true_theta()

## Visualize the likelihood of the samples p(x|theta)
problem.visualize()
plt.figure(figsize=(5,4))
Xs, Ys, Zs = visualization.plot_likelihood(samples=true_samples,
                              log_likelihood_function=problem.log_likelihood,
                              dimensions=(0,1),
                              bounded = False,
                              return_data=True)#,n_levels=10)
np.save(f"Xs_gt.npy", Xs)
np.save(f"Ys_gt.npy", Ys)
np.save(f"Zs_gt.npy", Zs)
plt.show()
visualization.plot_samples(problem.data_obs)
#uos.save_object(DIR, 'true_samples', true_samples)


imgs = problem.imgs_obs
print(imgs.shape)
figs,axes = plt.subplots(1,4)
plt.tight_layout()
for i in range(4):
    axes[i].imshow(imgs[i][:,:,::-1])
    axes[i].set_axis_off()
plt.show()


## Sequential Neural Likelihood
hyperparams = ABC_algorithms.Hyperparams()
hyperparams.save_dir = DIR
hyperparams.device = 'cuda:0'
#hyperparams.device = 'cpu'

hyperparams.num_sim = 800#4000                       # number of sampling/simulation rounds
hyperparams.L = 4                                # number of learning rounds
#hyperparams.type = 'cnn2d'                       # the network architecture of S(x)
hyperparams.type = 'cnnimg'
hyperparams.stat = 'infomax'                     # statistics function: infomax/moment/score  

#Hyperparameters for StatNet
hyperparams.n_neg = 50
hyperparams.bs = 32
hyperparams.lr = 2.0e-4

hyperparams.estimator = 'DC'                     # MI estimator; JSD (accurate) or DC (fast)
hyperparams.nde = 'MAF'                          # nde; MAF (D>1) or MDN (D=1)
from lfi.utils import discrepancy, visualization

print('\n SNL ABC')
print(f"Imgs observed: {problem.imgs_obs.shape}")
problem.n = 2
snl_abc = SNL2ABC.SNL2_ABC_Image(problem, discrepancy=discrepancy.eculidean_dist, hyperparams=hyperparams)
snl_abc.run()

JSD_array = []
for l in range(len(snl_abc.nde_array)):
    print('l=', l)
    snl_abc.nde_net = snl_abc.nde_array[l]
    Xs, Ys, Zs = visualization.plot_likelihood(samples=true_samples,
                                                log_likelihood_function=snl_abc.log_likelihood,
                                                dimensions=(0,1),
                                                bounded=False,
                                                return_data=True)
    np.save(f"Xs_iter{l}.npy", Xs)
    np.save(f"Ys_iter{l}.npy", Ys)
    np.save(f"Zs_iter{l}.npy", Zs)
    plt.show()
    JSD = discrepancy.JSD(problem.log_likelihood, snl_abc.log_likelihood, true_samples, true_samples, N_grid=30)
    JSD_array.append(JSD)
    print('JSD snl = ', JSD)
uos.save_object(DIR, 'JSD_SNL', JSD_array)

from abc import ABCMeta, abstractmethod

import numpy as np
import torch 
import os, sys, time, math
import scipy.stats as stats
import matplotlib.pyplot as plt
from copy import deepcopy

from lfi.utils import umath, uos
from lfi.utils import distributions
from lfi.utils import discrepancy
from lfi.algorithms import ABC_algorithms
from lfi.statnet.statnets import ISN, MSN, SSN , ISN_img
from lfi.neuralde import MAF, MDN
import csv



class SNL2_ABC(ABC_algorithms.Base_ABC):

    '''
    Sequential neural likelihood (with summary stat).
    '''
    
    def __init__(self, problem, discrepancy, hyperparams, **kwargs):
        '''
        Creates an instance of rejection ABC for the given problem.
        Parameters
        ----------
            problem : ABC_Problem instance
                The problem to solve An instance of the ABC_Problem class.
            discrepency: function pointer
                The data discrepency
            hyperparams: 1-D array
                The hyper-parameters [epsilon, num-samples]
            verbose : bool
                If set to true iteration number as well as number of
                simulation calls will be printed.
            save : bool
                If True will save the result to a (possibly exisisting)
                database
        '''
        super(SNL2_ABC, self).__init__(problem, discrepancy, hyperparams, **kwargs)
        
        self.needed_hyperparams = ['epsilon']
        self.epsilon = hyperparams.epsilon
        self.device = torch.device(hyperparams.device)
        self.nde_net = None                             # the learned q(x|theta)
        self.stat_net = None                            # the learned s(x)
        self.nde_array = []                             
        self.stat_array = []     
        self.proposal_array = []                        # the proposal used at each round
        self.hyperparams = hyperparams
 
    def convert_stat(self, x): 
        # no autoencoder, directly return s
        if self.stat_net is None: 
            s = x
            return s
        # convert raw data to summary stat: s = S(x)
        else:
            s = self.stat_net.encode(torch.tensor(x).float())
            return s.detach().cpu().numpy()
            
    def fit_nde(self):
        print('\n > fitting nde')
        all_stats = torch.tensor(self.convert_stat(np.vstack(self.all_stats[0:self.l+1]))).float().to(self.device)
        all_samples = torch.tensor(np.vstack(self.all_samples[0:self.l+1])).float().to(self.device)
        [n, dim] = all_stats.size()
        print('all_stats.size()', all_stats.size())
        if self.hyperparams.nde == 'MAF':
            #net = MAF.MAF(n_blocks=5, n_inputs=dim, n_hidden=50, n_cond_inputs=self.problem.K)
            net = MAF.MAF(n_blocks=5, n_inputs=dim, n_hidden=80, n_cond_inputs=self.problem.K)
        if self.hyperparams.nde == 'MDN':
            net = MDN.MDN(n_in=self.problem.K, n_hidden=50, n_out=dim, K=8)
        if self.nde_net is not None:
            net.load_state_dict(deepcopy(self.nde_net.state_dict()))
        net.train().to(self.device)
        net.learn(inputs=all_stats, cond_inputs=all_samples)
        net = net.eval().cpu()
        self.nde_net = net
        self.nde_array.append(net)

    def learn_stat(self):
        print('\n > fitting encoder')
        all_stats = torch.tensor(np.vstack(self.all_stats[0:self.l+1])).float().to(self.device)
        all_samples = torch.tensor(np.vstack(self.all_samples[0:self.l+1])).float().to(self.device)
        [n, dim] = all_stats.size()
        h = self.problem.K*2
        print('summary statistic dim =', h, 'original dim =', dim)
        #architecture = [dim] + [100, 100, h]
        architecture = [dim] + [100, 150,100, h]    
        print('architecture', architecture)
        if self.hyperparams.stat == 'infomax':
            net = ISN(architecture, dim_y=self.problem.K, hyperparams=self.hyperparams)
        if self.hyperparams.stat == 'moment':
            net = MSN(architecture, dim_y=self.problem.K, hyperparams=self.hyperparams)
        if self.hyperparams.stat == 'score':
            net = SSN(architecture, dim_y=self.problem.K, hyperparams=self.hyperparams)
        if self.stat_net is not None:
            net.load_state_dict(deepcopy(self.stat_net.state_dict()))
        net.train().to(self.device)
        net.learn(x=all_stats, y=all_samples)
        net = net.eval().cpu()
        self.stat_net = net
        self.stat_array.append(net)

    def sample_from_nde(self):
        net = self.nde_net
        net.eval()
        # pilot run for rej sampling
        if self.max_ll is None:
            self.max_ll = -math.inf
            for j in range(10000):
                theta = self.problem.sample_from_prior()
                ll = self.log_likelihood(theta)
                if ll > self.max_ll: self.max_ll = ll
        # rejection sampling
        while True:
            theta = self.problem.sample_from_prior()
            prob_accept = self.log_likelihood(theta) - self.max_ll
            u = distributions.uniform.draw_samples(0, 1, 1)[0]
            if np.log(u) < prob_accept: break
        return theta
        
    def log_likelihood(self, theta, use_ratio=False):
        if not use_ratio:
            '''
                log p(theta|x_o) = log q(x_o|theta)     (note: uniform prior)
            '''
            net = self.nde_net
            net.eval()
            y_obs, theta = self.convert_stat(self.whiten(self.y_obs)), theta
            y_obs, theta = torch.tensor(y_obs).float(), torch.tensor(theta).float().view(1, -1)
            log_probs = net.log_probs(inputs=y_obs, cond_inputs=theta)
            return log_probs.item()
        else:
            '''
            log p(theta|x_o) = log r(x_o, theta) + C(x_o)   (note: uniform prior. Here r(x, theta) = p(x, theta)/p(x)p(theta))
            '''
            net = self.stat_net
            net.eval()
            y_obs, theta = self.y_obs, theta
            y_obs, theta = torch.tensor(y_obs).float(), torch.tensor(theta).float().view(1, -1)
            log_probs = net.log_likelihood(y_obs, theta)
            return log_probs.view(-1).item()
        
    def set(self, l=0):
        self.l = l
        self.stat_net = self.stat_array[l]
        self.nde_net = self.nde_array[l]

    def run(self, all_stats=None, all_samples=None):
        '''
            main pipeline for the algorithm
        '''
        # initialization
        self.prior = self.problem.sample_from_prior
        
        # iterations
        L = self.hyperparams.L
        total_num_sim = self.num_sim 
        self.num_sim = int(total_num_sim/L)
        self.all_stats = []
        self.all_samples = []
        for l in range(L):
            print('iteration ', l)
            self.l = l
            self.max_ll = None
            if all_stats is None:
                self.simulate()
                self.all_stats.append(self.stats)
                self.all_samples.append(self.samples)
            else:
                self.all_stats = all_stats
                self.all_samples = all_samples
            self.learn_stat()
            self.fit_nde()
            self.prior = self.sample_from_nde
            print('\n')
        self.num_sim = total_num_sim
        
        # return
        self.save_results()

        

class SNL2_ABC_Image(ABC_algorithms.Base_ABC_Image):

    '''
    Sequential neural likelihood (with learned summary stat S(x)).
    '''
    
    def __init__(self, problem, discrepancy, hyperparams, **kwargs):
        '''
        Creates an instance of rejection ABC for the given problem.
        Parameters
        ----------
            problem : ABC_Problem instance
                The problem to solve An instance of the ABC_Problem class.
            discrepency: function pointer
                The data discrepency
            hyperparams: 1-D array
                The hyper-parameters [epsilon, num-samples]
            verbose : bool
                If set to true iteration number as well as number of
                simulation calls will be printed.
            save : bool
                If True will save the result to a (possibly exisisting)
                database
        '''
        super(SNL2_ABC_Image, self).__init__(problem, discrepancy, hyperparams, **kwargs)
        
        self.needed_hyperparams = ['epsilon']
        self.epsilon = hyperparams.epsilon
        self.device = torch.device(hyperparams.device)
        self.nde_net = None                             # the learned q(x|theta)
        self.stat_net = None                            # the learned s(x)
        self.nde_array = []                             
        self.stat_array = []     
        self.proposal_array = []                        # the proposal used at each round
        self.hyperparams = hyperparams
        self.sample_keep = 200
        
        self.csv_logger = f"mi_times_{self.hyperparams.estimator}.csv"
        
        with open(self.csv_logger, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["times"])
    def convert_stat(self, x): 
        # no autoencoder, directly return s
        if self.stat_net is None: 
            s = x
            return s #torch.tensor(x).float()#s
        # convert raw data to summary stat: s = S(x)
        else:
            s = self.stat_net.encode(torch.tensor(x).float())
            return s.detach().cpu().numpy()
        #x = torch.tensor(x).float().to(self.device)
        #with torch.no_grad():
        #    s = self.stat_net.encode(torch.tensor(x).float())
        #return s  # stays on GPU
            
    def fit_nde(self):
        print('\n > fitting nde')
        allstatsnp = self.convert_stat(np.concatenate(self.all_stats[0:self.l+1],axis=0))
        idx = np.random.choice(allstatsnp.shape[0], size=allstatsnp.shape[0], replace=False)
        all_stats = torch.tensor(allstatsnp[idx]).float().to(self.device)
        print(f"learn stat, all_stats: {all_stats.shape}")
        # All sampled thetas
        
        #all_samples = torch.tensor(np.vstack(self.all_samples[0:self.l+1])).float().to(self.device)
        allsampnp = np.vstack(self.all_samples[0:self.l+1])
        all_samples = torch.tensor(allsampnp[idx]).float().to(self.device)
        
        #all_stats = torch.tensor(self.convert_stat(np.concat(self.all_stats[0:self.l+1],axis=0))).float().to(self.device)
        #all_stats = self.convert_stat(np.concat(self.all_stats[0:self.l+1],axis=0)) #.float().to(self.device)
        #all_samples = torch.tensor(np.vstack(self.all_samples[0:self.l+1])).float().to(self.device)
        [n, dim] = all_stats.size()
        print('all_stats.size()', all_stats.size())
        if self.hyperparams.nde == 'MAF':
            net = MAF.MAF(n_blocks=5, n_inputs=dim, n_hidden=50, n_cond_inputs=self.problem.K,bs=32,lr=5e-5)
        if self.hyperparams.nde == 'MDN':
            net = MDN.MDN(n_in=self.problem.K, n_hidden=50, n_out=dim, K=8)
        if self.nde_net is not None:
            print("> Loading NDENet weights ...")
            net.load_state_dict(deepcopy(self.nde_net.state_dict()))
        #print("NDE net arch: \n",net)
        net.train().to(self.device)
        net.learn(inputs=all_stats, cond_inputs=all_samples)
        net = net.eval().cpu()
        self.nde_net = net
        self.nde_array.append(net)

    def learn_stat(self):
        """
            Train the neural statistic
        """
        print('\n > fitting encoder')
        # All simulated images
        allstatsnp = np.concatenate(self.all_stats[0:self.l+1],axis=0)
        idx = np.random.choice(allstatsnp.shape[0], size=allstatsnp.shape[0], replace=False)
        all_stats = torch.tensor(allstatsnp[idx]).float().to(self.device)
        print(f"learn stat, all_stats: {all_stats.shape}")
        # All sampled thetas
        
        #all_samples = torch.tensor(np.vstack(self.all_samples[0:self.l+1])).float().to(self.device)
        allsampnp = np.vstack(self.all_samples[0:self.l+1])
        all_samples = torch.tensor(allsampnp[idx]).float().to(self.device)

        #[n, dim] = all_stats.size()
        n,dim = all_stats.size()[0],72
        h = self.problem.K*2
        print('summary statistic dim =', h, 'original dim =', dim)
        architecture = [dim] + [100, 100, h] ## <<< Change this stats architecture
        print('architecture', architecture)
        
        ## Select the statistic network
        if self.hyperparams.stat == 'infomax':
            net = ISN_img(architecture, dim_y=self.problem.K, hyperparams=self.hyperparams)
        if self.hyperparams.stat == 'moment':
            net = MSN(architecture, dim_y=self.problem.K, hyperparams=self.hyperparams)
        if self.hyperparams.stat == 'score':
            net = SSN(architecture, dim_y=self.problem.K, hyperparams=self.hyperparams)
        if self.stat_net is not None:
            print("> Loading StatNet weights ...")
            net.load_state_dict(deepcopy(self.stat_net.state_dict()))
        #print("Summary statistics arch: \n",net)
        if self.hyperparams.stat == 'infomax':
            net.csv_logger = self.csv_logger
        net.train().to(self.device)
        net.learn(x=all_stats, y=all_samples)
        net = net.eval().cpu()
        self.stat_net = net
        self.stat_array.append(net)

    """
    def sample_from_nde(self):
        #print(">Sampling from NDE")
        net = self.nde_net
        net.eval()
        # pilot run for rej sampling
        if self.max_ll is None:
            self.max_ll = -math.inf
            for j in range(10000):
                theta = self.problem.sample_from_prior()
                ll = self.log_likelihood(theta)
                if ll > self.max_ll: self.max_ll = ll
        # rejection sampling
        cnt = 0
        while True:
            cnt +=1
            theta = self.problem.sample_from_prior()
            prob_accept = self.log_likelihood(theta) - self.max_ll
            u = distributions.uniform.draw_samples(0, 1, 1)[0]
            if np.log(u) < prob_accept: break
        return theta
    """
    
    def sample_from_nde(self, batch_size=512):
        """
        Draw a single theta ~ prior(θ) weighted by NDE likelihood, 
        using batched rejection sampling.
        """
        device = self.device
        net = self.nde_net.to(device).eval()

        # --- 1) Pilot run for max log-likelihood (once) ---
        if self.max_ll is None:
            num_pilot = 10000
            thetas_pilot = []
            for _ in range(num_pilot):
                theta = self.problem.sample_from_prior()
                thetas_pilot.append(theta)
            thetas_pilot = np.array(thetas_pilot)
            thetas_pilot = torch.tensor(thetas_pilot, dtype=torch.float32, device=device)  # (N, K)

            with torch.no_grad():
                ll_pilot = self.log_likelihood(thetas_pilot)  # (N,)

            self.max_ll = ll_pilot.max().item()
            # optional: print("max_ll =", self.max_ll)

        # --- 2) Batched rejection sampling ---
        while True:
            # sample a batch of candidates from prior
            thetas = []
            for _ in range(batch_size):
                theta = self.problem.sample_from_prior()
                thetas.append(theta)
            thetas = np.array(theta)
            thetas = torch.tensor(thetas, dtype=torch.float32, device=device)  # (B, K)

            with torch.no_grad():
                log_lik = self.log_likelihood(thetas)  # (B,)

            # log-acceptance probs
            log_accept = log_lik - self.max_ll   # (B,)

            # sample uniforms in log-space
            log_u = torch.log(torch.rand_like(log_accept))

            # mask of accepted indices
            mask = log_u < log_accept
            if mask.any():
                # pick first accepted sample
                idx = mask.nonzero(as_tuple=False)[0, 0]
                theta_accept = thetas[idx].detach().cpu().numpy()
                return theta_accept
    
    
    def log_likelihood(self, theta, use_ratio=False):
        if not use_ratio:
            '''
                log p(theta|x_o) = log q(x_o|theta)     (note: uniform prior)
            '''
            net = self.nde_net
            net.eval()
            #y_obs, theta = self.convert_stat(self.whiten(self.y_obs)), theta
            #print(f"SNL LOGLIKE imgs: {self.img_obs.shape}")
            y_obs, theta = self.convert_stat(self.img_obs), theta
            print(f"nde log like: {y_obs.device}")
            #y_obs = y_obs.float().to(self.device)
            #theta = torch.tensor(theta).float().view(1, -1).to(self.device)
            y_obs, theta = torch.tensor(y_obs).float(), torch.tensor(theta).float().view(1, -1)
            log_probs = net.log_probs(inputs=y_obs, cond_inputs=theta)
            return log_probs.item()
        else:
            '''
            log p(theta|x_o) = log r(x_o, theta) + C(x_o)   (note: uniform prior. Here r(x, theta) = p(x, theta)/p(x)p(theta))
            '''
            net = self.stat_net
            net.eval()
            #y_obs, theta = self.y_obs, theta
            y_obs, theta = self.img_obs, theta
            #y_obs, theta = torch.tensor(y_obs).float(), torch.tensor(theta).float().view(1, -1)
            y_obs, theta = torch.tensor(y_obs).float(), torch.tensor(theta).float()
            log_probs = net.log_likelihood(y_obs, theta)
            return log_probs.view(-1).item()
        
    def set(self, l=0):
        self.l = l
        self.stat_net = self.stat_array[l]
        self.nde_net = self.nde_array[l]

    def run(self, all_stats=None, all_samples=None):
        '''
            main pipeline for the algorithm
        '''
        # initialization
        self.prior = self.problem.sample_from_prior
        
        # iterations
        L = self.hyperparams.L
        total_num_sim = self.num_sim 
        self.num_sim = int(total_num_sim/L)
        self.all_stats = []
        self.all_samples = []
        for l in range(L):
            print('iteration ', l)
            self.l = l
            self.max_ll = None
            if all_stats is None:
                # simulate() is different from problem.simulator(),
                # simulate() obtains low-dim summaries from a group of 
                # high-dim samples (i.e. a Dataset-> vector summary)
                self.simulate() 
                # Enlarging the Dataset
                self.all_stats.append(self.stats) # Add more raw stat samples (images)
                self.all_samples.append(self.samples) # Add more theta samples
            else:
                self.all_stats = all_stats
                self.all_samples = all_samples
            self.learn_stat() # Train the Stat Net with: (Raw images,thetas) 
            self.fit_nde() # Train the Neural Density Estimator p(theta|S(X_o))
            
            ### Just debugging the JSD discrepancy (comment if unnecessary)
            ### ----------------------------------------------------------------
            true_samples = self.problem.sample_from_true_posterior()
            JSD = discrepancy.JSD(self.problem.log_likelihood, self.log_likelihood, true_samples, true_samples, N_grid=30)
            print(f"[DEBUG] -- JSD: {JSD}")
            ### ----------------------------------------------------------------
            
            stats_all   = np.vstack(self.all_stats)       # (N_total, ... flattened later inside learn_stat)
            samples_all = np.vstack(self.all_samples)     # (N_total, K)

            sample_keep = self.sample_keep
            N_total = stats_all.shape[0]
            keep = min(sample_keep, N_total)
            idx = np.random.choice(N_total, size=keep, replace=False)

            self.all_stats   = [stats_all[idx]]
            self.all_samples = [samples_all[idx]]
            #selsamples = np.random.randint(low=0,high=len(self.all_stats),size=200)
            #self.all_stats = list(self.all_stats[selsamples])
            #self.all_samples = list(self.all_samples[selsamples])
            self.prior = self.sample_from_nde # Sample a new theta theta~p(theta|X_o)
            print('\n')
        self.num_sim = total_num_sim
        
        # return
        self.save_results()

        
        
        

import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.distributions as distribution
import math
import numpy as np
import time
from copy import deepcopy

from lfi.utils import optimizer

class MDN(nn.Module):
    """ 
        Mixture density network 
    """
    def __init__(self, n_in, n_hidden, n_out, K=1,bs=200,lr=5e-4):
        super(MDN, self).__init__()
        self.bs = bs
        self.lr = lr
        self.wd = 1e-5
        self.main = nn.Sequential(
            nn.Linear(n_in, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
        )
        self.K = K
        self.dim = n_out
        self.coeff_layer = CoeffLayer(n_hidden, K)
        self.mean_layer = nn.ModuleList([MeanLayer(n_hidden, n_out) for i in range(K)])
        self.cov_layer = nn.ModuleList([CovLayer(n_hidden, n_out) for i in range(K)])
        
    def forward(self, cond_inputs):
        # nn(x) = {coeff}, {mu}, {cov}
        h = self.main(cond_inputs)
        mu_array, C_array, log_det_array = [], [], []
        for k in range(self.K):
            # > mu
            mu = self.mean_layer[k](h)
            mu_array.append(mu)
            # > cov
            C, log_det = self.cov_layer[k](h)
            C_array.append(C)
            log_det_array.append(log_det)
        coeff = self.coeff_layer(h)
        return coeff, mu_array, C_array, log_det_array
    
    def sample(self, cond_inputs, n=1):
        device = cond_inputs.device
        coeff, mu_array, C_array, log_det_array = self.forward(cond_inputs)
        coeff = torch.softmax(coeff,dim=1)
        categorical = distribution.Categorical(coeff)
        samples = []
        for i in range(n):
            k = categorical.sample()    # pick a component
            assert k.shape[0] == 1, print("Batch ===>> 1")
            mu, C = mu_array[k][0], C_array[k][0].inverse()
            V = C.mm(C.t())
            normal = distribution.MultivariateNormal(mu, V)
            x = normal.sample()        
            samples.append(x)
        return torch.cat(samples, dim=0)
            
    """
    def log_probs(self, inputs, cond_inputs):
        # pdf = \sum coeff[k] * N(x; mu[k], cov[k])
        coeff, mu_array, C_array, log_det_array = self.forward(cond_inputs)
        prob = torch.zeros(len(inputs)).to(inputs.device)
        normal = distribution.Normal(torch.tensor([0.0]).to(inputs.device), torch.tensor([1.0]).to(inputs.device))
        for k in range(self.K):   # <- pdf for each Gaussian component
            mu, C, log_det = mu_array[k], C_array[k], log_det_array[k]
            z = (inputs - mu).unsqueeze(dim=1)
            C_T = C.transpose(dim0=1, dim1=2)
            z = z.bmm(C_T)
            z = z.squeeze(dim=1)
            log_base_prob = normal.log_prob(z).sum(dim=1)
            log_prob = log_base_prob + log_det
            prob += coeff[:,k] * log_prob.exp() 
        return (prob + 1e-16).log()
    """
    
    def log_probs(self, inputs, cond_inputs):
        coeff, mu_array, C_array, log_det_array = self.forward(cond_inputs)
        log_coeff = torch.log_softmax(coeff,dim=1)
        # enforce valid mixture weights
        #log_coeff = (coeff + 1e-12).log()  # or better: have forward return logits and do log_softmax

        # collect component log-probs: shape [B, K]
        comps = []
        for k in range(self.K):
            mu, C, log_det = mu_array[k], C_array[k], log_det_array[k]

            z = (inputs - mu).unsqueeze(1)          # [B,1,D]
            z = z.bmm(C.transpose(1, 2)).squeeze(1) # [B,D]

            log_base = (-0.5 * (z ** 2) - 0.5 * torch.log(torch.tensor(2.0 * torch.pi, device=z.device))).sum(dim=1)
            log_pk = log_base + log_det             # <-- ensure log_det matches your transform convention

            comps.append(log_coeff[:, k] + log_pk)

        comps = torch.stack(comps, dim=1)           # [B,K]
        return torch.logsumexp(comps, dim=1)        # [B]
    
    def objective_func(self, inputs, cond_inputs):
        return self.log_probs(inputs, cond_inputs).mean()
    
    def learn(self, inputs, cond_inputs):
        loss_value = optimizer.NNOptimizer.learn(self, inputs, cond_inputs)
        return loss_value
    
    
        
class CoeffLayer(nn.Module):
    def __init__(self, n_in, K):
        super(CoeffLayer, self).__init__()
        self.n_in = n_in
        self.K = K
        self.linear = nn.Linear(n_in, K)
        
    def forward(self, h):
        m, d = h.size()
        out = self.linear(h)
        #s = out.exp()
        #coeff = s/s.sum(dim=1, keepdim=True) 
        return out #coeff
    
        
class MeanLayer(nn.Module): 
    def __init__(self, n_in, n_out):
        super(MeanLayer, self).__init__()
        self.n_in = n_in
        self.n_out = n_out
        self.linear = nn.Linear(n_in, n_out)
        
    def forward(self, h):
        m, d = h.size()
        out = self.linear(h)
        mean = out.view(m, self.n_out)
        return mean
        

class CovLayer(nn.Module):
    def __init__(self, n_in, n_out):
        super(CovLayer, self).__init__()
        self.n_in = n_in
        self.n_out = n_out
        self.linear = nn.Linear(n_in, n_out*n_out)
   
    def mask(self, h):
        n = len(h)
        ones = 1 + torch.zeros(self.n_out, self.n_out)
        ltri_mask = torch.tril(ones, diagonal=-1).expand(n, self.n_out, self.n_out)
        diag_mask = torch.eye(self.n_out).expand(n, self.n_out, self.n_out)
        return ltri_mask.to(h.device), diag_mask.to(h.device)
        
    def forward(self, h):
        n, d = h.size()
        out = self.linear(h)
        out = out.view(n, self.n_out, self.n_out)
        ltri_mask, diag_mask = self.mask(h)
        ltri, diag = out*ltri_mask, (out.exp()*diag_mask)
        C = ltri + diag                    
        log_det = (out*diag_mask).sum(dim=2).sum(dim=1)
        return C, log_det   # x = C^{-1}z, z = Cx, det|C| = -det|dx/dz|  C^{-1}C^{-T} = Sigma
    
    
        
        
        
        

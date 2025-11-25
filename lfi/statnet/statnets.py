import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.autograd as autograd
import numpy as np
import scipy
import math
import time
from copy import deepcopy

from utils import optimizer
from baselayer import CriticLayer,ScoreLayer


class ISN(nn.Module):
    """ 
        Information-theoretic Statistics Network
    """
    def __init__(self, architecture, dim_y, hyperparams):
        super().__init__()

        # default hyperparameters
        self.estimator = 'JSD' if not hasattr(hyperparams, 'estimator') else hyperparams.estimator  
        self.bs = 200 if not hasattr(hyperparams, 'bs') else hyperparams.bs 
        self.lr = 5e-4 if not hasattr(hyperparams, 'lr') else hyperparams.lr
        self.wd = 0e-5 if not hasattr(hyperparams, 'wd') else hyperparams.wd
        self.n_neg = 25 if not hasattr(hyperparams, 'n_neg') else hyperparams.n_neg
        
        self.encode_y = True if not hasattr(hyperparams, 'encode_y') else hyperparams.encode_y
        self.encode_layer = EncodeLayer(architecture, dim_y, hyperparams)
        self.encode2_layer = EncodeLayer([dim_y] + architecture[1:], dim_y, None)
        self.critic_layer = CriticLayer(architecture, architecture[-1], hyperparams)
    
    def encode(self, x):
        # s = s(x), get the summary statistic of x
        return self.encode_layer(x)
    
    def encode2(self, y):
        # theta = h(y), get the representation of y
        return self.encode2_layer(y)
        
    def MI(self, z, y, n=10):
        # [A]. Jensen-shannon divergence (DeepInfoMax, ICLR'19)
        if self.estimator == 'JSD':
            m, d = z.size()
            z, y = self.encode(z), self.encode2(y) if self.encode_y else y
            idx_pos = []
            idx_neg = []
            for i in range(n): 
                idx_pos = idx_pos + np.linspace(0, m-1, m).tolist()
                idx_neg = idx_neg + torch.randperm(m).cpu().numpy().tolist()
            f_pos = self.critic_layer(z, y)
            f_neg = self.critic_layer(z[idx_pos], y[idx_neg])
            A, B = -F.softplus(-f_pos), F.softplus(f_neg)
            mi = A.mean() - B.mean()
        # [B]. Distance correlation (Annals of Statistics'07)
        if self.estimator == 'DC':
            m, d = z.size()
            z, y = self.encode(z), self.encode2(y) if self.encode_y else y
            A = torch.cdist(z, z, p=2)
            B = torch.cdist(y, y, p=2)
            A_row_sum, A_col_sum = A.sum(dim=0, keepdim=True), A.sum(dim=1, keepdim=True)
            B_row_sum, B_col_sum = B.sum(dim=0, keepdim=True), B.sum(dim=1, keepdim=True)
            a = A - A_row_sum/(m-2) - A_col_sum/(m-2) + A.sum()/((m-1)*(m-2))
            b = B - B_row_sum/(m-2) - B_col_sum/(m-2) + B.sum()/((m-1)*(m-2))
            AB, AA, BB = (a*b).sum()/(m*(m-3)), (a*a).sum()/(m*(m-3)), (b*b).sum()/(m*(m-3))
            mi = AB**0.5/(AA**0.5 * BB**0.5)**0.5
        # [C]. Donsker-Varadhan Representation (MINE, ICML'18)
        if self.estimator == 'DV':
            m, d = z.size()
            z, y = self.encode(z), self.encode2(y)
            idx_pos = []
            idx_neg = []
            for i in range(n):
                idx_pos = idx_pos + np.linspace(0, m-1, m).tolist()
                idx_neg = idx_neg + torch.randperm(m).cpu().numpy().tolist()
            f_pos = self.critic_layer(z, y)
            f_neg = self.critic_layer(z[idx_pos], y[idx_neg])
            mi = f_pos.mean() - f_neg.exp().mean().log()
        # [D]. Wasserstein dependency measure (WPC, NIPS'19)
        if self.estimator == 'WD':
            z, y = self.encode(z), self.encode2(y)
            m, d, K = z.size()[0], z.size()[1], y.size()[1] 
            idx_pos = []
            idx_neg = []
            for i in range(n):
                idx_pos = idx_pos + np.linspace(0, m-1, m).tolist()
                idx_neg = idx_neg + torch.randperm(m).cpu().numpy().tolist()
            f_pos = self.critic_layer(z, y)
            f_neg = self.critic_layer(z[idx_pos], y[idx_neg])
            mi = f_pos.mean() - f_neg.mean()
        return mi
    
    def objective_func(self, x, y):
        return self.MI(x, y, n=self.n_neg)
    
    def learn(self, x, y):
        loss_value = optimizer.NNOptimizer.learn(self, x, y)
        return loss_value


class MSN(nn.Module):
    """ 
        Moment statistic network
    """
    def __init__(self, architecture, dim_y, hyperparams):
        super().__init__()
        self.bs = 400 if not hasattr(hyperparams, 'bs') else hyperparams.bs 
        self.lr = 1e-3 if not hasattr(hyperparams, 'lr') else hyperparams.lr
        self.wd = 0e-5 if not hasattr(hyperparams, 'wd') else hyperparams.wd
        self.type = 'plain' if not hasattr(hyperparams, 'type') else hyperparams.type 
        self.dropout = False if not hasattr(hyperparams, 'dropout') else hyperparams.dropout 
        
        self.encode_layer = EncodeLayer(architecture, dim_y, hyperparams)

    def forward(self, x):
        return self.encode(x)

    def encode(self, x):
        return self.encode_layer(x)
    
    def objective_func(self, x, y):
        yy = self.forward(x)
        J = torch.norm(yy-y, dim=1)**2
        return -J.mean()
    
    def learn(self, x, y):
        loss_value = optimizer.NNOptimizer.learn(self, x, y)
        return loss_value


class SSN(nn.Module):
    """ 
        Score-matching Statistic Network
    """
    def __init__(self, architecture, dim_y, hyperparams):
        super().__init__()
        self.bs = 400 if not hasattr(hyperparams, 'bs') else hyperparams.bs 
        self.lr = 1e-3 if not hasattr(hyperparams, 'lr') else hyperparams.lr
        self.wd = 0e-5 if not hasattr(hyperparams, 'wd') else hyperparams.wd
        self.type = 'plain' if not hasattr(hyperparams, 'type') else hyperparams.type 
        self.dropout = False if not hasattr(hyperparams, 'dropout') else hyperparams.dropout 
        
        self.score_layer = ScoreLayer(architecture[-1], dim_y, 100, n_layer=1)
        self.encode_layer = EncodeLayer(architecture, dim_y, hyperparams)

    def encode(self, x):
        # s = s(x), get the summary statistic of x
        return self.encode_layer(x)
        
    def SM(self, x, y):
        n, dim = y.size()
        x = x.clone().requires_grad_(True)
        y = y.clone().requires_grad_(True)
        log_energy = self.score_layer(x, y)                                    # log f(y|x), R^d                m*1
        score = autograd.grad(log_energy.sum(), y, create_graph=True)[0]       # d_y log f(y|x),                m*d
        loss1 = 0.5*torch.norm(score, dim=1) ** 2                              # |d_y|^2                        m*1
        loss2 = torch.zeros(n, device=x.device)                                # trace d_yy log f(y|x)          m*1
        for d in range(dim): 
            loss2 += autograd.grad(score[:, d].sum(), y, create_graph=True, retain_graph=True)[0][:, d] 
        loss = loss1 + loss2
        return loss.mean()
    
    def objective_func(self, x, y):
        loss = self.SM(self.encode(x), y)
        return -loss
        
    def learn(self, x, y):
        loss_value = optimizer.NNOptimizer.learn(self, x, y)
        return loss_value   

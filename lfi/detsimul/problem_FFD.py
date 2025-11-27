import numpy as np
import math
import scipy.stats as stats
from abc import ABCMeta, abstractmethod
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import time
import os

from lfi.utils import distributions 
from lfi.utils import umath
from lfi.detsimul import ABC_problems
from glob import glob
import cv2


class FFD_Image_Problem(ABC_problems.ABC_Problem):
    """
    The problem consists of drawing FFD deformation parameters
    from a Gaussian Distribution and applying them to a group
    of face images.
    """
    def __init__(self,N=500,n=100,image_folder="",out_folder=""):
        self.simulator_args = ['mean', 'var']
        self.prior = [distributions.uniform, distributions.uniform]
        self.prior_args = np.array([[-5.0, 5.0], [0.001, 100.0]])
        self.true_mean = 0.0
        self.true_var = 1.0
        
        self.N = N
        self.n = n
        self.mudim = 72
        self.K = 2
        self.img_folder = image_folder
        self.out_img_folder = out_folder
        self.ctrl_pnts = 6
    
    def get_true_theta(self):
        return np.array([self.true_mean, self.true_var])
    
    def basis_function(self,t):
        """
        B-spline basis function [FFD deformation]
        """
        t2 = t * t
        t3 = t * t2

        B0 = (1.0 - t)**3 / 6.0
        B1 = (3.0 * t3 - 6.0 * t2 + 4.0) / 6.0
        B2 = (-3.0 * t3 + 3.0 * t2 + 3.0 * t + 1.0) / 6.0
        B3 = t3 / 6.0
        # Stack to create (N_points, 4) shape
        return np.stack([B0, B1, B2, B3], axis=-1)


    def apply_ffd_to_points(self,points, control_points_displacements, N_CTRL_X, N_CTRL_Y, img_w, img_h):
        T_x = np.zeros(points[:, 0].shape)
        T_y = np.zeros(points[:, 1].shape)
        N_points = points.shape[0]

        # 1. Map physical coordinates (x, y) to parametric coordinates (u, v)
        u_norm = points[:, 0] * (N_CTRL_X - 1) / img_w
        v_norm = points[:, 1] * (N_CTRL_Y - 1) / img_h

        # 2. Find the starting integer indices (k, l)
        k = np.floor(u_norm).astype(int)
        l = np.floor(v_norm).astype(int)

        k = np.clip(k, 1, N_CTRL_X - 3)
        l = np.clip(l, 1, N_CTRL_Y - 3)

        # 3. Find the local parameter (t_u, t_v)
        t_u = u_norm - k
        t_v = v_norm - l

        # 4. Pre-calculate all basis function values
        B_u_all = self.basis_function(t_u) # Shape: (N_points, 4)
        B_v_all = self.basis_function(t_v) # Shape: (N_points, 4)

        # 5. Calculate the total deformation vector T(u, v)

        for i in range(4): # B-spline basis index i (0, 1, 2, 3)
            # B_u is the weight for the i-th basis function across all points
            B_u = B_u_all[:, i] # Shape: (N_points,)
            cp_i = k - 1 + i    # Control point row index (e.g., k-1, k, k+1, k+2)

            for j in range(4): # B-spline basis index j (0, 1, 2, 3)
                # B_v is the weight for the j-th basis function across all points
                B_v = B_v_all[:, j] # Shape: (N_points,)
                cp_j = l - 1 + j    # Control point column index

                # 5a. Calculate the total scalar weight for this control point (cp_i, cp_j)
                weight = B_u * B_v # Shape: (N_points,)

                # 5b. Fetch the displacement values. This uses advanced indexing.
                # This operation MUST return an array of shape (N_points,)
                delta_P_x = control_points_displacements[cp_i, cp_j, 0]
                delta_P_y = control_points_displacements[cp_i, cp_j, 1]

                # 5c. Accumulate the deformation
                T_x += weight * np.float64(delta_P_x)
                T_y += weight * np.float64(delta_P_y)

        # 6. Calculate the new point position
        new_points = np.stack((points[:, 0] + T_x, points[:, 1] + T_y), axis=-1)

        return new_points
    
    def ffd_image_warp(self,image, N_CTRL_X, N_CTRL_Y, displacements):
        # Get image dimensions
        h, w = image.shape[:2]
        disp_reshaped = displacements.reshape(N_CTRL_X,N_CTRL_Y,2)

        # 1. Create the grid of original pixel coordinates
        # We create two 2D arrays (X, Y) representing the coordinates (x, y) of every pixel.
        x_map, y_map = np.meshgrid(np.arange(w), np.arange(h))

        # Flatten the maps into a list of (x, y) points
        original_points = np.stack((x_map.flatten(), y_map.flatten()), axis=-1)

        # 2. Apply the FFD to the grid points
        deformed_points = self.apply_ffd_to_points(
            original_points,
            disp_reshaped,
            N_CTRL_X, N_CTRL_Y,
            w, h
        )

        # 3. Prepare the maps for cv2.remap
        map_x = deformed_points[:, 0].reshape((h, w)).astype(np.float32)
        map_y = deformed_points[:, 1].reshape((h, w)).astype(np.float32)

        # 4. Perform the image warping using OpenCV
        # We use INTER_CUBIC for high-quality interpolation, which is appropriate for FFD's smooth deformation.
        warped_image = cv2.remap(
            image,
            map_x,
            map_y,
            interpolation=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REFLECT # Or BORDER_CONSTANT for black border
        )
        return warped_image


    def simulator(self,theta):
        """
        Returns deformed images

        Args:
            theta (_type_): _description_
        """
        mean = theta[0]
        var = theta[1]
        cov = np.diag(var*np.ones(self.mudim))
        nmean = mean*np.ones(self.mudim)
        MU = distributions.normal_nd.draw_samples(nmean,cov,self.n)
        # Convert mu to x (images)
        #X = self.Z2X(MU)
        return MU
    
    def Z2X(self, Z):
        ndisp = Z.shape[0]
        print(f"==== Warping===>{ndisp} samples")
        cnt = 1
        ffds = []
        for fname in glob(self.img_folder+"/*.jpg"):
            img = cv2.imread(fname,1)
            for d in range(ndisp):
                warped = self.ffd_image_warp(img,self.ctrl_pnts,self.ctrl_pnts,Z[d,:])
                warped = cv2.resize(warped,(256,256),interpolation=cv2.INTER_LINEAR)
                cv2.imwrite(os.path.join(self.out_img_folder,f"{cnt:05d}.jpg"),warped)
                cnt += 1
        return self.out_img_folder
    
    def log_likelihood(self, theta):
        '''
        Calculate the log_likelihood of theta given y_obs.
        log p(x_obs|theta)
        
        ----------
        Parameters
        ----------
        theta : array
            The array of parameters
        Returns
        -------
        output : float
            The likelihood value L(theta)
        '''
        mean = theta[0]*np.ones(self.mudim)
        covar = theta[1]*np.diag(np.ones(self.mudim))
        ## "Observed" Densities according to the original theta
        densities = distributions.normal_nd.pdf(self.data_obs,mean,covar)
        ll = np.log(densities+1e-10)
        return ll.sum()
    
    def log_pdf(self,data, theta):
        '''
        Calculate the log_likelihood of theta given y_obs.
        log p(x_obs|theta)
        
        ----------
        Parameters
        ----------
        theta : array
            The array of parameters
        Returns
        -------
        output : float
            The likelihood value L(theta)
        '''
        mean = theta[0]*np.ones(self.mudim)
        covar = theta[1]*np.diag(np.ones(self.mudim))
        ## "Observed" Densities according to the original theta
        densities = distributions.normal_nd.pdf(data,mean,covar)
        ll = np.log(densities+1e-10)
        return ll.sum()

    def sample_from_prior(self):
        '''
        Sample one value from the prior.
        ----------
        Returns
        -------
        output : array
            one sample from the prior
        '''
        sample_mean = self.prior[0].draw_samples(self.prior_args[0,0],self.prior_args[0,1],1)[0]
        sample_var = self.prior[1].draw_samples(self.prior_args[1,0],self.prior_args[1,1],1)[0]
        return np.array([sample_mean, sample_var])
    
    def visualize(self):
        print('visualizing p(x|theta) in PCA space')
        
        # -------------------------------
        # 1. Prepare samples / PCA
        # -------------------------------
        samples = self.data_obs                       # (m, 72)
        m, dim = samples.shape

        # Compute PCA
        data_mean = samples.mean(axis=0)
        Xc = samples - data_mean                      # center
        C = Xc.T @ Xc / (m - 1)                       # covariance
        evals, evecs = np.linalg.eigh(C)              # symmetric eigendecomp
        
        # top-2 principal components
        idx = np.argsort(evals)[::-1]
        W = evecs[:, idx[:2]]                         # (72, 2)
        
        # project observed samples to 2D PCA space
        Z_samples = Xc @ W                             # (m, 2)

        # -------------------------------
        # 2. Determine plotting range
        # -------------------------------
        min_values = Z_samples.min(axis=0)
        max_values = Z_samples.max(axis=0)

        N_grid = 200
        ranges = []
        for k in range(2):                             # only 2D now
            r = np.linspace(min_values[k], max_values[k], N_grid)
            ranges.append(r)
        X, Y = np.meshgrid(*ranges)
        
        # Build grid points R in PCA space: shape (N_grid*N_grid, 2)
        R = np.stack([X.ravel(), Y.ravel()], axis=1)

        # -------------------------------
        # 3. Evaluate p(x | theta) in PCA space
        # -------------------------------
        theta = self.get_true_theta()
        mean_full = np.full(dim, theta[0])             # 72-D mean
        var = theta[1]                                 # variance
        
        # Project model mean to PCA space
        mean_pca = (mean_full - data_mean) @ W         # (2,)

        # Covariance in PCA space is var * I_2 due to isotropic model + orthonormal PCA basis
        cov_pca = var * np.eye(2)
        
        # Multivariate normal pdf in PCA space
        from scipy.stats import multivariate_normal
        rv = multivariate_normal(mean=mean_pca, cov=cov_pca)
        
        pdf = rv.pdf(R)                                # (N_grid*N_grid,)
        Z_pdf = pdf.reshape(X.shape)                   # (N_grid, N_grid)

        # -------------------------------
        # 4. Make contour plot
        # -------------------------------
        plt.figure(figsize=(6, 6))
        plt.contour(X, Y, Z_pdf, 15, cmap='jet', linewidths=0.75)

        # overlay sample scatter
        plt.scatter(Z_samples[:,0], Z_samples[:,1], s=8, color='black', alpha=0.25)

        plt.title(r"$p(x|\theta)$ projected onto first 2 PCA components")
        plt.xlabel("PC 1")
        plt.ylabel("PC 2")
        plt.tight_layout()
        plt.show()
        
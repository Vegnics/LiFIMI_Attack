import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import time

from lfi.utils import umath, uos
from lfi.utils import discrepancy


#"""
def plot_contour(X, Y, p_XY, region, title, xlabel, ylabel, bounded=True):
    plt.contour(X, Y, p_XY, 40, cmap='jet', linewidths=0.75)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    xmin = region[0, 0]
    xmax = region[0, 1]
    ymin = region[1, 0]
    ymax = region[1, 1]
    if bounded:
        plt.xlim((xmin, xmax))
        plt.ylim((ymin, ymax))
#"""

def plot_axline(x,y):
    plt.axvline(x=x,color='k', linewidth=0.75)
    plt.axhline(y=y,color='k', linewidth=0.75)

#"""
def plot_likelihood(samples,
                    log_likelihood_function,
                    dimensions=(0,1),
                    bounded = True,
                    return_data=False,
                    xmin = None,
                    xmax = None,
                    ymin = None,
                    ymax = None): 
    # Compute log-likelihood values
    n, d = samples.shape
    if d == 1:
        X, P = umath.log_likelihood_1D(samples, log_likelihood_function)
        plt.figure(time.time()*100, figsize=(5, 5))
        plt.plot(X,P)
        plt.xlabel(r'$\theta$')
        plt.ylabel(r'$\log p(\theta|x_o)$')
        return
    elif d == 2:
        X, Y, P = umath.log_likelihood_2D(samples, log_likelihood_function,xmin,xmax,ymin,ymax)
    else:
        X, Y, P = umath.log_likelihood_3D(samples, log_likelihood_function, dimensions)
        
    # Determine the visualize region
    visualize_samples = samples[:, dimensions]
    (mean1, std1) = visualize_samples[:,0].mean(), visualize_samples[:,0].std()
    (mean2, std2) = visualize_samples[:,1].mean(), visualize_samples[:,1].std()
    print('mean-parma1 = ', mean1, '     mean-param2 = ', mean2)
    F = 1.0
    R = np.array([[mean1-F*std1, mean1+F*std1], [mean2-F*std2, mean2+F*std2]])
    #R = np.array([[mean1-8.0*std1, mean1+8.0*std1], [mean2-8.0*std2, mean2+8.0*std2]])
    # Visualize contour
    fig = plt.figure(time.time()*100, figsize=(5, 5))
    #fig.set_tight_layout(True)
    #plt.axis('off')
    #plt.figure()
    #plt.imshow(P,cmap="jet")
    plot_contour(X, Y, P, R, r'Plot likelihood p($\theta|S_{ll}(X_o))$', r'$\theta_{}$'.format(dimensions[0]), r'$\theta_{}$'.format(dimensions[1]),bounded)
    if return_data:
        return X, Y, P
#"""

"""
def plot_likelihood(samples, log_likelihood_function, dimensions=(0, 1), 
                    bounded=True, n_levels=10, q_low=0.01, q_high=0.99):
    
    n, d = samples.shape

    if d == 1:
        X, P = umath.log_likelihood_1D(samples, log_likelihood_function)
        plt.figure(figsize=(5, 5))
        plt.plot(X, P)
        plt.xlabel(r'$\theta$')
        plt.ylabel(r'$\log p(\theta|x_o)$')
        return

    elif d == 2:
        X, Y, P = umath.log_likelihood_2D(samples, log_likelihood_function)
    else:
        X, Y, P = umath.log_likelihood_3D(samples, log_likelihood_function, dimensions)

    # ---- choose visualization region from samples via quantiles ----
    visualize_samples = samples[:, dimensions]

    x_samp = visualize_samples[:, 0]
    y_samp = visualize_samples[:, 1]

    x_low, x_high = np.quantile(x_samp, [q_low, q_high])
    y_low, y_high = np.quantile(y_samp, [q_low, q_high])

    # small margin so contours don't touch the borders
    mx = 0.1 * (x_high - x_low)
    my = 0.1 * (y_high - y_low)

    xmin = x_low - mx
    xmax = x_high + mx
    ymin = y_low - my
    ymax = y_high + my

    # also respect the grid’s limits
    xmin = max(xmin, X.min())
    xmax = min(xmax, X.max())
    ymin = max(ymin, Y.min())
    ymax = min(ymax, Y.max())

    region = np.array([[xmin, xmax], [ymin, ymax]])

    print(f"x-range: [{xmin:.3f}, {xmax:.3f}], y-range: [{ymin:.3f}, {ymax:.3f}]")

    # ---- plot contour ----
    fig = plt.figure(figsize=(5, 5))
    plot_contour(X, Y, P, region,
                 r'Plot likelihood p($\theta|S_{ll}(X_o)$)',
                 r'$\theta_{}$'.format(dimensions[0]),
                 r'$\theta_{}$'.format(dimensions[1]),
                 bounded=bounded,
                 n_levels=n_levels)
"""

"""
def plot_contour(X, Y, p_XY, region, title, xlabel, ylabel,
                 bounded=True, n_levels=20, delta_log=100.0):
    # crop to region first (optional but recommended)
    x_grid = X[0, :]
    y_grid = Y[:, 0]
    xmin, xmax = region[0]
    ymin, ymax = region[1]

    ix = (x_grid >= xmin) & (x_grid <= xmax)
    iy = (y_grid >= ymin) & (y_grid <= ymax)

    X_sub, Y_sub = np.meshgrid(x_grid[ix], y_grid[iy])
    P_sub = p_XY[iy][:, ix]

    # ---- cut off very low log-likelihood values ----
    P_max = P_sub.max()
    vmin = P_max - delta_log          # only keep logL >= P_max - delta_log
    P_sub_clipped = np.clip(P_sub, vmin, P_max)

    # contour levels between vmin and max
    levels = np.linspace(vmin, P_max, n_levels)

    plt.contour(X_sub, Y_sub, P_sub_clipped,
                levels=levels, cmap='jet', linewidths=0.75)

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    if bounded:
        plt.xlim(xmin, xmax)
        plt.ylim(ymin, ymax)
"""


def plot_samples(samples, dimensions=(0,1)):
    visualize_samples = samples[:, dimensions]
    plt.figure(time.time()*100, figsize=(5, 4))
    plt.scatter(visualize_samples[:,0], visualize_samples[:,1], s=8, edgecolors='k', marker='o', facecolors='none')
    
    
def compare_contours(samples, true_likelihood, approx_likelihood, dimensions=(0,1)):
    # Compute log-likelihood values
    n, d = samples.shape
    if d == 2:
        X, Y, P1 = umath.log_likelihood_2D(samples, true_likelihood)
        X, Y, P2 = umath.log_likelihood_2D(samples, approx_likelihood)
    else:
        X, Y, P1 = umath.log_likelihood_3D(samples, true_likelihood, dimensions)
        X, Y, P2 = umath.log_likelihood_3D(samples, approx_likelihood, dimensions)
        
    # Determine the visualize region
    visualize_samples = samples[:, dimensions]
    (mean1, std1) = visualize_samples[:,0].mean(), visualize_samples[:,0].std()
    (mean2, std2) = visualize_samples[:,1].mean(), visualize_samples[:,1].std()
    print('mean-parma1 = ', mean1, '     mean-param2 = ', mean2)
    R = np.array([[mean1-3.0*std1, mean1+3.0*std1], [mean2-3.0*std2, mean2+3.0*std2]])
    
    # Visualize contour
    plt.figure(time.time()*100, figsize=(5, 4))
    C1 = plt.contour(X, Y, P1, 10, colors='k', linewidths=0.75)
    C2 = plt.contour(X, Y, P2, 10, colors='r', linewidths=0.75, linestyles='dashed', alpha=0.85)
    C1.collections[0].set_label('true posterior')
    C2.collections[0].set_label('approx posterior')
    plt.legend(bbox_to_anchor=(-0.00,1.02,1.00,0.2), loc="lower left",
                mode="expand", borderaxespad=0, ncol=2)
    plt.title('')
    plt.xlabel(r'$\theta_{}$'.format(dimensions[0]))
    plt.ylabel(r'$\theta_{}$'.format(dimensions[1]))
    xmin = R[0, 0]
    xmax = R[0, 1]
    ymin = R[1, 0]
    ymax = R[1, 1]
    plt.xlim((xmin, xmax))
    plt.ylim((ymin, ymax))
    plt.savefig('contours_compare.png')
    
    

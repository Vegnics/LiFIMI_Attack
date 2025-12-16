import numpy as np
from matplotlib import pyplot as plt
import os


def kl_divergence(p, q, eps=1e-100):
    """
    Compute the Kullback-Leibler (KL) divergence D_KL(P || Q) for two 
    discrete probability distributions.

    Args:
        p (np.ndarray): The 'true' probability distribution (P).
        q (np.ndarray): The reference probability distribution (Q).

    Returns:
        float: The computed KL divergence.
    """
    p = p / (p.sum() + eps)
    q = q / (q.sum() + eps)
    # avoid log(0)
    p = np.clip(p, eps, None)
    q = np.clip(q, eps, None)
    KLD = np.sum(p * np.log(p / q))
    return KLD

def JS_divergence(p, q, eps=1e-100):
    # Normalize
    p = p / (p.sum() + eps)
    q = q / (q.sum() + eps)

    m = 0.5 * (p + q)

    # avoid log(0)
    p = np.clip(p, eps, None)
    q = np.clip(q, eps, None)
    m = np.clip(m, eps, None)

    KL_pm = np.sum(p * np.log(p / m))
    KL_qm = np.sum(q * np.log(q / m))

    return 0.5 * KL_pm + 0.5 * KL_qm

"""
def JS_divergence(p, q):
    rows = p.shape[0]
    cols = p.shape[1]
    KL_PM = 0
    KL_QM = 0
    prob_p = p/(np.sum(p))
    prob_q = q/(np.sum(q))
    prob_pq = (prob_p + prob_q)/2
    for i in range(rows):
        for j in range(cols):
            if prob_p[i,j] < 1e-20:
                KL_PM += 0
            else:
                KL_PM += prob_p[i,j] * np.log(prob_p[i,j]/prob_pq[i,j])

            if prob_q[i,j] < 1e-20:
                KL_QM += 0
            else:
                KL_QM += prob_q[i,j] * np.log(prob_q[i,j]/prob_pq[i,j])
    return KL_PM/2 + KL_QM/2
"""

FOLDER = ""
MI_EST = "JSD"
scale = 5
#dist_path = f"/home/amaranth2/Downloads/exp1_1208/task_FFD_image_MAF_mi_{MI_EST}_scale_{scale:0.1f}"
#dist_path = f"/home/amaranth2/Downloads/task_FFD_image_MAF_mi_{MI_EST}_scale_4.0/task_FFD_image_MAF_mi_{MI_EST}_scale_4.0"
#dist_path = "/home/amaranth2/Downloads/task_FFD_image_MAF_mi_DV_scale_4.0/task_FFD_image_MAF_mi_DV_scale_4.0"
dist_path = f"/home/amaranth2/Downloads/task_FFD_MAF_mi_{MI_EST}_scale_{scale:0.1f}/task_FFD_MAF_mi_{MI_EST}_scale_{scale:0.1f}"

#niter = 2
gt_file = "/home/amaranth2/Downloads/Zs_gt.npy"
for niter in range(10):
    dist_file = os.path.join(dist_path,f"Zs_iter{niter}_MAF_mi_{MI_EST}_scale_{scale:0.1f}.npy")
    Zs_gt = np.load(gt_file)
    Zs_gt = Zs_gt/np.sum(Zs_gt)
    Zs_nde = np.load(dist_file)
    Zs_nde = Zs_nde/np.sum(Zs_nde)
    
    Xs = np.load(os.path.join(dist_path,f"Xs_iter{niter}_MAF_mi_{MI_EST}_scale_{scale:0.1f}.npy"))
    Ys = np.load(os.path.join(dist_path,f"Ys_iter{niter}_MAF_mi_{MI_EST}_scale_{scale:0.1f}.npy"))
    NDE_P = np.transpose(np.load(dist_file))[::-1,:]
    GT_P = np.transpose(np.load(gt_file))[::-1,:]
    JSD = JS_divergence(NDE_P,GT_P)
    #KLD = kl_divergence(NDE_P,GT_P)
    KLD = kl_divergence(GT_P,NDE_P)
    expect_xnde = np.sum(Xs*Zs_nde)
    expect_ynde = np.sum(Ys*Zs_nde)
    expect_xgt = np.sum(Xs*Zs_gt)
    expect_ygt = np.sum(Ys*Zs_gt)
    delta_x = np.abs(expect_xgt-expect_xnde)
    delta_y = np.abs(expect_ygt-expect_ynde)
    MSE = (0.5*(delta_x**2+delta_y**2))**0.5 
    print(f"Iteration: {niter+1} ->> JSD: {JSD:0.3f},  KLD: {KLD:0.3f}, (dx={delta_x:0.3f},dy={delta_y:0.3f}), MSE: {MSE:0.3f}")
    print(expect_xgt,expect_ygt)
    #print(GT_P.shape)
    #print(dist_file)
    plt.imshow(NDE_P,cmap="jet")
    plt.show()
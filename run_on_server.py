import pandas as pd
from time import time
import numpy as np
from typing import Any, Union
from skimage.data import shepp_logan_phantom
from skimage import transform
from scipy.fft import fft2, ifft2
from pywt import dwt2, idwt2
from utils import logspace, allcombinations, add_standard_attributes, run_local_df_experiment, repeat
import matplotlib.pyplot as plt
from scipy.fft import fft2, ifft2

def square_haar(wav:tuple):
    (cA, part) = wav
    (cH, cV, cD) = part
    top = np.concatenate((cA, cV), axis=1)
    bottom = np.concatenate((cH, cD), axis=1)
    return np.concatenate((top, bottom), axis=0)

def soft_threshold(x, lambda_):
    """Soft thresholding operator"""
    return np.sign(x) * np.maximum(np.abs(x) - lambda_, 0)

def SF_func(indices):
    return lambda x : fft2(x, norm="ortho").flat[indices]

def S_transpose(indices, m,n):
    S = np.zeros((m,n)) 
    for (i, idx) in enumerate(indices):
        S[i, idx] = 1
    return np.transpose(S)

def SF_transpose_map(n, m, indices):
    sqt_n = int(n**0.5)
    S_conj = S_transpose(indices, m, n)
    return lambda x : ifft2((S_conj @ x.flat).reshape((sqt_n, sqt_n)), norm="ortho")

def wr_prob_vec(loc_co):
    return loc_co**2/np.sum(loc_co**2)

def Bernoulli_prob_weights(loc_co, m, n):
    assert (loc_co > 0).all()
    assert m < n
    sorted_co = np.sort(loc_co)
    J = n
    def Lsqrd(j, m, n, sorted_co):
        return m * np.sum(sorted_co[:j]**2)/ (m - (n-j))
    for j in range(n, n-m-1, -1):
        if m*sorted_co[j-1]**2 < Lsqrd(j, m, n, sorted_co):
            J = j
            break
    Lsqr = Lsqrd(J, m, n, sorted_co)
    return np.clip(m * loc_co**2 / Lsqr, 0, 1), J, Lsqr
def unsquare_haar(mat):
    midpoint = mat.shape[0]//2
    cA= mat[:midpoint, :midpoint]
    cV=mat[midpoint:, :midpoint]
    cD=mat[midpoint:, midpoint:]
    cH=mat[:midpoint, midpoint:]
    return (cA, (cV,cH, cD))

def lasso_wr_Fourier(n,m,indices, b:np.ndarray, x:np.ndarray, lambda_, D=None, max_iter=1000, tol=1e-5):
    """LASSO solver using Iterative Soft Thresholding Algorithm (ISTA)
    If no resampling, set L to 2
    """
    # L is the Lipshitz constant of the gradient, which scales with the operator norm of S^TS, which is the max number of repeated measurements.
    L = 2* np.max(np.unique(indices, return_counts=True)[1]) * n/m
    if D == None:
        D = np.ones(n)
    D_tld=D[indices]
    x_old = np.zeros_like(x)
    SF=SF_func(indices)
    SF_transpose=SF_transpose_map(n, m, indices)
    fits = np.zeros(max_iter)
    regs = np.zeros(max_iter)
    for ind in range(max_iter):
        x_old[:,:] = x
        residual = (b - np.sqrt(n/m)*SF(x))
        fits[ind] = np.linalg.norm(residual)**2
        regs[ind] = np.sum(np.abs(x))
        gradient = -2*np.real(np.sqrt(n/m)SF_transpose(D_tld**2*residual))
        x = soft_threshold(x - gradient / L, lambda_)
        if np.linalg.norm(x - x_old) < tol:
            break
    return x, fits, regs

def make_local_df():
    sqt_n=32 #original size
    df= allcombinations({'sqt_n': sqt_n,'m': sqt_n**2*2, 'lambda': 0.000001, 'sigma': 0.0000})
    df = repeat(df, 2)
    df['seed'] = [np.random.randint(0, 2**32 - 1) for _ in range(len(df))]
    add_standard_attributes(df)
    return df

# local_df =  make_local_df()
# result = run_local_df_experiment(local_df, experiment)
# plt.figure()
# plt.plot(result['fits'][0])
# plt.plot(result['regs'][0])
# plt.yscale('log')
# np.max(result['rre'])

def make_df():
    seed = 1234
    np.random.seed(seed)
    sqt_n=100
    df= allcombinations({'sqt_n': sqt_n,'m': logspace(1,sqt_n**2, 8, integer=True), 'sigma': [0,0.0001, 0.001, 0.01, 1.0], 'lambda':0.001, 'tol':0.0001})
    df = repeat(df, 3)
    df['seed'] = [np.random.randint(0, 2**32 - 1) for _ in range(len(df))]
    df.attrs={"label":"unif_recov_curve_fixed", "global_seed":seed}
    add_standard_attributes(df)
    return df

def experiment(data:Union[dict, pd.Series])-> dict[str, Any]:
    np.random.seed(data['seed'])
    start_time = time()
    sqt_n= int(data['sqt_n'])
    n=sqt_n**2
    m = int(data['m'])
    true_signal = transform.resize(shepp_logan_phantom(), (sqt_n, sqt_n))
    features=square_haar(dwt2(true_signal, "haar"))
    indices = np.random.choice(n,m,replace=True)
    sigma = data['sigma']
    SF = SF_func(indices)
    b = np.sqrt(n/m)*SF(features) + sigma * np.random.normal(0, 1/np.sqrt(m), m)
    x_hat, fits, regs = lasso_wr_Fourier(n, m , indices, b, np.zeros([sqt_n, sqt_n], dtype=np.float32), data['lambda'], max_iter=1000, tol = data['tol'])
    recovered_img = idwt2(unsquare_haar(x_hat), "haar")
    dist = np.linalg.norm(x_hat - features)
    rre = np.linalg.norm(x_hat - features)/ np.linalg.norm(features)
    run_time = time()-start_time
    return {'rre': rre, 'runtime_seconds': run_time, 'fits': fits, 'regs': regs, 'dist': dist, 'indices': indices}

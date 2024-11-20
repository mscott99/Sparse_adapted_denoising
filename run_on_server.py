import pandas as pd
import numpy as np
from typing import Any, Union
from skimage.data import shepp_logan_phantom
from skimage import transform
from scipy.fft import fft2, ifft2
from pywt import dwt2, idwt2
from utils import logspace, allcombinations, add_standard_attributes, run_local_df_experiment
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

def lasso_wr_Fourier(n,m,indices, b:np.ndarray, x:np.ndarray, lambda_, D=None, max_iter=100, tol=1e-4):
    """LASSO solver using Iterative Soft Thresholding Algorithm (ISTA)"""
    L = 2
    if D == None:
        D = np.ones(n)
    D_tld=D[indices]
    x_old = np.zeros_like(x)
    SF = SF_func(indices)
    SF_transpose=SF_transpose_map(n, m, indices)
    for _ in range(max_iter):
        x_old[:,:] = x
        residual = (b - SF(x))
        gradient = SF_transpose(D_tld**2*residual).astype(np.float32)
        x = soft_threshold(x + gradient / L, lambda_ / L)
        if np.linalg.norm(x - x_old) < tol:
            break
    return x

def make_df():
    sqt_n=100
    df= allcombinations({'sqt_n': sqt_n,'m': logspace(1,sqt_n**2-1, 4, integer=True), 'sigma': [0.0001, 0.01, 1.0]})
    df.attrs={"label":"figure_out_test"}
    add_standard_attributes(df)
    return df

def experiment(data:Union[dict, pd.Series])-> dict[str, Any]:
    np.random.seed(data['seed'])
    sqt_n= int(data['sqt_n'])
    n=sqt_n**2
    m = int(data['m'])
    true_signal = transform.resize(shepp_logan_phantom(), (sqt_n, sqt_n))
    features=square_haar(dwt2(true_signal, "haar"))
    indices = np.random.choice(n,m,replace=True)
    sigma = data['sigma']
    SF = SF_func(indices)
    b = SF(features) + sigma * np.random.normal(0, 1, m)
    x_hat = lasso_wr_Fourier(n, m , indices, b, np.zeros([sqt_n, sqt_n], dtype=np.float32), 0.001, max_iter=100)
    recovered_img = idwt2(unsquare_haar(x_hat), "haar")
    rre = np.linalg.norm(x_hat - features)/ np.linalg.norm(features)
    return {'recovered_img':recovered_img,'rre': rre}

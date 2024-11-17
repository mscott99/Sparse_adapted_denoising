from itertools import chain
import seaborn as sns
import numpy as np
from skimage.data import shepp_logan_phantom
from skimage import transform
from scipy.fft import fft2, ifft2
import matplotlib.pyplot as plt
from pywt import dwt2, idwt2
from utils import logspace, allcombinations
import matplotlib.pyplot as plt
from scipy.fft import fft2, ifft2

# def l1_wavelet(wav):
#     sum = np.sum(np.abs(wav[0]))
#     for elt in wav[1:]:
#         for sub_mat in elt:
#             sum+= np.sum(np.abs(sub_mat))
#     return sum

# def flatten_haar(wav:tuple):
#     return np.concatenate((wav[0].ravel(), *(w.ravel() for w in chain(*wav[1:]))))

# def shape_up_haar(arr:np.ndarray):
#     size = arr.shape[0]
#     midpoint = size // 4
#     square_size = int(midpoint**0.5)
#     cA = arr[:midpoint].reshape((square_size, square_size))
#     cH = arr[midpoint:(midpoint*2)].reshape((square_size, square_size))
#     cV = arr[(midpoint*2):(midpoint*3)].reshape((square_size, square_size))
#     cD = arr[(midpoint*3):(midpoint*4)].reshape((square_size, square_size))
#     return (cA, (cH, cV, cD))

def square_haar(wav:tuple):
    (cA, part) = wav
    (cH, cV, cD) = part
    top = np.concatenate((cA, cV), axis=1)
    bottom = np.concatenate((cH, cD), axis=1)
    return np.concatenate((top, bottom), axis=0)

def unsquare_haar(mat):
    midpoint = mat.shape[0]//2
    cA= mat[:midpoint, :midpoint]
    cV=mat[midpoint:, :midpoint]
    cD=mat[midpoint:, midpoint:]
    cH=mat[:midpoint, midpoint:]
    return (cA, (cV,cH, cD))

test= np.random.randn(4,4)
assert np.allclose(square_haar(unsquare_haar(square_haar(dwt2(test, "haar")))), square_haar(dwt2(test, "haar")))

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
    sorted_co = np.sort(loc_co)
    J = n
    def Lsqrd(j):
        return m * np.sum(sorted_co[:j]**2)/ (m - (n-j))
    for j in range(n-1, 0, -1):
        if m* sorted_co[j]**2 < Lsqrd(j):
            J = j
            break
    Lsqr = Lsqrd(J)
    return np.clip(m * loc_co**2 / Lsqr, 0, 1), J, Lsqr

n = 10
m = 6
loc_co = np.random.rand(10)
sns.barplot(Bernoulli_prob_weights(loc_co, m, n)[0])
sns.barplot(loc_co)

np.sum(Bernoulli_prob_weights(loc_co, m, n)[0])


# Try to make this not too terrible.
# Model is minimize \|D_tld(b- SFx_hat)\|_2^2 + lambda \|x_hat\|_1 with prox descent
def lasso_wr_Fourier(n,m,indices, b:np.ndarray, x:np.ndarray, lambda_, D=np.ones(n), max_iter=100, tol=1e-4):
    """LASSO solver using Iterative Soft Thresholding Algorithm (ISTA)"""
    L = 2
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

sqt_n=32
true_signal = transform.resize(shepp_logan_phantom(), (sqt_n, sqt_n))
features=square_haar(dwt2(true_signal, "haar"))
plt.imshow(features)

n=sqt_n**2
m = n // 5
indices = np.random.choice(n,m)

sigma = 0.000
SF = SF_func(indices)
b = SF(features) + sigma * np.random.normal(0, 1, m)

x_hat = lasso_wr_Fourier(n, m , indices, b, np.zeros((sqt_n, sqt_n), dtype=np.float32), 0.001, max_iter=100)
plt.imshow(np.concatenate((true_signal, idwt2(unsquare_haar(x_hat), "haar")), axis=1))

np.linalg.norm(x_hat - features)/ np.linalg.norm(features)

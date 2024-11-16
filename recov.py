from itertools import chain
from numbers import Complex
from tokenize import Floatnumber
import numpy as np
from pandas import Float64Dtype
from skimage.data import shepp_logan_phantom
from skimage import transform
from scipy.fft import dct, idct, fft2, ifft2
import matplotlib.pyplot as plt
from pywt import dwt2, idwt2, wavedec2 
import pywt
from utils import logspace, allcombinations
from sklearn.linear_model import Lasso
from sklearn import linear_model
import matplotlib.pyplot as plt
from scipy.fft import dctn, fft2, ifft2
from scipy.sparse import bmat
# from scipy.linalg import hadamard


# def experiment(n=64, m=None, sigma=0.01, sparsity=0.1):
# if m is None:
#     m = n // 4


def l1_wavelet(wav):
    sum = np.sum(np.abs(wav[0]))
    for elt in wav[1:]:
        for sub_mat in elt:
            sum+= np.sum(np.abs(sub_mat))
    return sum

def flatten_haar(wav:tuple):
    return np.concatenate((wav[0].ravel(), *(w.ravel() for w in chain(*wav[1:]))))

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

def shape_up_haar(arr:np.ndarray):
    size = arr.shape[0]
    midpoint = size // 4
    square_size = int(midpoint**0.5)
    cA = arr[:midpoint].reshape((square_size, square_size))
    cH = arr[midpoint:(midpoint*2)].reshape((square_size, square_size))
    cV = arr[(midpoint*2):(midpoint*3)].reshape((square_size, square_size))
    cD = arr[(midpoint*3):(midpoint*4)].reshape((square_size, square_size))
    return (cA, (cH, cV, cD))

assert np.allclose(true_signal, idwt2(shape_up_haar(flatten_haar(dwt2(true_signal, "haar"))), "haar"))

def soft_threshold(x, lambda_):
    """Soft thresholding operator"""
    return np.sign(x) * np.maximum(np.abs(x) - lambda_, 0)

def SF_func(indices):
    return lambda x : fft2(x, norm="ortho").flat[indices]

def ST(indices, m,n):
    S = np.zeros((m,n)) 
    for (i, idx) in enumerate(indices):
        S[i, idx] = 1
    return np.transpose(S)

indices = np.random.choice(100,5)
ST(indices, 5, 100)

def SF_transpose_map(n, m, indices):
    sqt_n = int(n**0.5)
    S_transpose = ST(indices, m, n)
    return lambda x : ifft2((S_transpose @ x.flat).reshape((sqt_n, sqt_n)), norm="ortho")

# Try to make this not too terrible.
def lasso_wr_Fourier(n,m,indices, b:np.ndarray[np.complex64], x:np.ndarray[np.float32], lambda_, max_iter=100, tol=1e-4):
    """LASSO solver using Iterative Soft Thresholding Algorithm (ISTA)"""
    L = 2
    x_old = np.zeros_like(x)
    map = SF_func(indices)
    transpose_map=SF_transpose_map(n, m, indices)
    for _ in range(max_iter):
        x_old[:,:] = x
        residual = b - map(x)
        gradient = transpose_map(residual).astype(np.float32)
        x = soft_threshold(x + gradient / L, lambda_ / L)
        if np.linalg.norm(x - x_old) < tol:
            break
    return x

test=flatten_haar(dwt2(true_signal, "haar"))
assert np.array_equal(test, flatten_haar(shape_up_haar(test)))

sqt_n=128
true_signal = transform.resize(shepp_logan_phantom(), (sqt_n, sqt_n))

features=square_haar(dwt2(true_signal, "haar"))
plt.imshow(features)

n=sqt_n**2
m = n // 5
indices = np.random.choice(n,m)

# G = np.random.randn(m,n)
# identity = np.eye(n, n).reshape((sqt_n,sqt_n,sqt_n,sqt_n))
# stored_fft2 = dctn(identity, axes= (1,2), type=2).reshape((n,n))

sigma = 0.000
SF = SF_func(indices)
b = SF(features) + sigma * np.random.normal(0, 1, m)

x_hat = lasso_wr_Fourier(n, m , indices, b, np.zeros((sqt_n, sqt_n), dtype=np.float32), 0.001, max_iter=100)
plt.imshow(np.concatenate((true_signal, idwt2(unsquare_haar(x_hat), "haar")), axis=1))

np.linalg.norm(x_hat - features)/ np.linalg.norm(features)

# lasso = Lasso(0.001)
# result = lasso.fit(A, b)


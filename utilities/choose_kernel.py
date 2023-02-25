import numpy as np
from scipy.spatial.distance import cdist
from time import perf_counter
from scipy import spatial

def choose_kernel(kernelType = "gaussianKernel"):
    d = kernelType
    if kernelType == "gaussianKernel":
        return GaussianKernel()
    else:
        print("No such kernel is implemented yet")

class GaussianKernel():
    def __init__(self):
        return

    def kernel_operator(self, X2, bandwidth, factor=None):
        return lambda x:  (1 / len(X2)) * self.calcKernel(x, X2, bandwidth, factor=factor)

    def calcKernel(self, X1, X2, bandwidth, factor=None):
        self.bandwidth = bandwidth
        if factor != None:
            bandwidth = factor * bandwidth

        D = cdist(X1, X2, metric = 'sqeuclidean')
        D = (-1 / (2*bandwidth ** 2)) * D
        return np.exp(D)

    def prediction(self, Xts, Xtr, coef):
        Knn = self.calcKernel(Xts, Xtr, self.bandwidth)
        return np.dot(Knn, coef)

    def calc_derivative(self, alpha, X1, X2):

        d = (X1-X2)
        dd = (X2-X1)
        m = np.multiply(d, alpha)
        mm = np.multiply(dd, alpha)
        return -1/(self.bandwidth)**2 * np.dot(self.calcKernel(X1, X2, self.bandwidth), np.multiply((X1-X2), alpha))


# Radial kernel
def get_nn_indices(x, y, r):
    """Return nn indices for x"""
    tree = spatial.KDTree(x)
    indices = tree.query_ball_point(y, r)
    nn_indices = {i: set(idx) for i, idx in enumerate(indices)}
    nn_vector = [item for sublist in indices for item in sublist]
    return nn_indices, nn_vector

def radial_kernel(x, r, n):
    nn_indices, _ = get_nn_indices(x, x, r)
    W = np.zeros((n, n))
    for i in range(0, n):
        W[i, list(nn_indices[i])] = 1
    return W

def gaussian_kernel(x1, x2, bandwidth, factor=None):
        if factor != None:
            bandwidth = factor * bandwidth
        D = cdist(x1, x2, metric = 'sqeuclidean')
        D = (-1 / (2*bandwidth ** 2)) * D
        return np.exp(D)
from sklearn.neighbors import NearestNeighbors
import numpy as np
from utilities.util import timer_func
from scipy.spatial.distance import cdist

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

class Knn():
    def __init__(self, config):
        self.config = config
        self.knn = NearestNeighbors(radius=self.config['knn']['radius'])
        self.xtr = None
        self.ytr = None
        self.num_nn = None

    def predict(self, x, timeit=False):
        """Returns prediction time if timeit == True"""
        if timeit:
            return self.predict_with_timeit(x)
        else:
            y_pred, pred_time = self.predict_with_timeit(x)
            return y_pred

    @timer_func
    def predict_with_timeit(self, xts):
        self.knn.fit(self.xtr)
        nearest_neighbour_indices = self.knn.kneighbors(xts, self.num_nn, return_distance=False)
        nn_ytr = self.ytr[nearest_neighbour_indices]
        return np.mean(nn_ytr, axis=1)

    def add_model(self, model, num_nn):
        self.num_nn = num_nn
        self.xtr = model['x']
        self.ytr = model['y']

    def clear(self):
        self.xtr = None
        self.ytr = None
        self.num_nn = None

class KnnApf(Knn):
    def __init__(self, config):
        super().__init__(config)

    def add_model(self, model, num_nn):
        self.num_nn = num_nn
        self.xtr = model['apf']
        self.ytr = model['y']



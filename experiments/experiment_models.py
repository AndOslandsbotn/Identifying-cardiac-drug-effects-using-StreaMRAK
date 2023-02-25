from solvers.falkon import Falkon
from solvers.streamrak import Streamrak
from solvers.knn import Knn, KnnApf

from utilities.util import list_to_ndarray
from utilities.stats import calc_stats
from utilities.data_io import load_npz_data, load_json_data
from pathlib import Path


import os
import numpy as np
from definitions import ROOT_DIR

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

class ExperimentBase():
    def __init__(self, model_name, config_solver, config_predictor):
        self.config = config_predictor
        self.model_datasizes = self.config['general']['model_datasizes']
        self.model_name = model_name
        self.model = self.init_model(model_name, config_solver)
        self.time_usage = {}

        self.model_datasize = None
        self.test_data = None
        self.test_data_filename = None

    def init_model(self, model_name, config_solver):
        if model_name == 'falkon':
            return Falkon(config_solver)
        elif model_name == 'streamrak':
            return Streamrak(config_solver)
        elif model_name == 'euc-knn':
            return Knn(config_solver)
        elif model_name == 'apf-knn':
            return KnnApf(config_solver)

    def add_model(self, model_datasize, num_nn):
        model_data = self.load_training_data(model_datasize)
        self.model.add_model(model_data, num_nn)

    def add_existing_model(self, datasize):
        """Loads existing model of that was trained on datasize
        :param datasize: Size of training data that model was trained on
        """
        model_folder = os.path.join(ROOT_DIR, self.config['files']['model_folder']
                                      +f"_idx{self.config['general']['indices_param']}", self.model_name)
        model_filename = f'model_N{datasize}.json'
        trained_model = load_json_data(os.path.join(model_folder, model_filename))
        trained_model = list_to_ndarray(trained_model['model'])
        self.model.add_trained_model(trained_model)

    def load_test_data(self, test_data_filename):
        self.test_data_filename= test_data_filename
        indices_param = self.config['general']['indices_param']
        test_data_folder = os.path.join(ROOT_DIR, f"{self.config['files']['test_data_folder']}{indices_param}")
        self.test_data = load_npz_data(test_data_folder, test_data_filename+'.npz')

    def load_training_data(self, datasize):
        indices_param = self.config['general']['indices_param']
        data_folder = os.path.join(ROOT_DIR, f"{self.config['files']['data_folder']}{indices_param}")
        data_filename = f"{self.config['files']['trdata_filename']}{datasize}.npz"
        return load_npz_data(data_folder, data_filename)

class DomainExperiment(ExperimentBase):
    def __init__(self, model_name, config_solver, config_predictor):
        super().__init__(model_name, config_solver, config_predictor)

    def predict(self):
        for model_datasize in self.model_datasizes:
            if self.model_name in ['falkon', 'streamrak']:
                self.add_existing_model(model_datasize)
                self.make_prediction()
                self.save(model_datasize, self.model_name)
                self.model.clear()
            elif self.model_name in ['apf-knn', 'euc-knn']:
                for num_nn in self.config['general']['num_nn']:
                    self.add_model(model_datasize, num_nn)
                    self.make_prediction()
                    self.save(model_datasize, self.model_name+f'{num_nn}')
                    self.model.clear()

    def make_prediction(self):
        if self.model_name == 'apf-knn':
            self.xts, self.yts = self.test_data['apf'], self.test_data['y']
        else:
            self.xts, self.yts = self.test_data['x'], self.test_data['y']
        self.yts_pred, self.prediction_time = self.model.predict(self.xts, timeit=True)
        self.nmse, self.nstd, self.tnmse, self.tnstd = calc_stats(self.yts, self.yts_pred)

    def save(self, model_datasize, model_name):
        results_folder = os.path.join(ROOT_DIR, self.config['files']['results_folder']
                                      +f"_idx{self.config['general']['indices_param']}", model_name)
        Path(results_folder).mkdir(parents=True, exist_ok=True)

        filename = f"{self.test_data_filename}_pred_N{model_datasize}"
        np.savez(os.path.join(results_folder, filename),
                 xts=self.xts, yts=self.yts, yts_pred=self.yts_pred,
                 nmse = self.nmse, nstd = self.nstd, tnmse = self.tnmse, tnstd = self.tnstd,
                 predtime = self.prediction_time)
        return

class PredictionTimeExperiment(ExperimentBase):
    def __init__(self, model_name, config_solver, config_predictor):
        super().__init__(model_name, config_solver, config_predictor)
        self.config_predictor = config_predictor
        self.test_data_sizes = config_predictor['prediction_time_exp']['test_data_sizes'].copy()
        self.model_datasize = config_predictor['prediction_time_exp']['model_datasize']

    def predict(self):
        for test_data_size in self.test_data_sizes:
            print("test_data_size: ", test_data_size)
            if self.model_name in ['falkon', 'streamrak']:
                self.add_existing_model(self.model_datasize)
                self.make_prediction(test_data_size)
                self.save(test_data_size, self.model_name)
                self.model.clear()
            elif self.model_name in ['apf-knn', 'euc-knn']:
                for num_nn in self.config['general']['num_nn']:
                    self.add_model(self.model_datasize, num_nn)
                    self.make_prediction(test_data_size)
                    self.save(test_data_size, self.model_name+f'{num_nn}')
                    self.model.clear()

    def make_prediction(self, test_data_size):
        if self.model_name == 'apf-knn':
            self.xts, self.yts = self.test_data['apf'][:test_data_size, :], self.test_data['y'][:test_data_size, :]
        else:
            self.xts, self.yts = self.test_data['x'][:test_data_size, :], self.test_data['y'][:test_data_size, :]
        self.yts_pred, self.prediction_time = self.model.predict(self.xts, timeit=True)
        self.nmse, self.nstd, self.tnmse, self.tnstd = calc_stats(self.yts, self.yts_pred)

    def save(self, test_data_size, model_name):
        results_folder = os.path.join(ROOT_DIR, self.config['files']['results_folder']
                                      +f"_idx{self.config['general']['indices_param']}", model_name)
        Path(results_folder).mkdir(parents=True, exist_ok=True)

        filename = f"{self.test_data_filename}_pred_Nts{test_data_size}"
        np.savez(os.path.join(results_folder, filename),
                 xts=self.xts, yts=self.yts, yts_pred=self.yts_pred,
                 nmse = self.nmse, nstd = self.nstd, tnmse = self.tnmse, tnstd = self.tnstd,
                 predtime = self.prediction_time)
        return




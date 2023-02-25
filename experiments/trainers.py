from solvers.streamrak import Streamrak
from solvers.falkon import Falkon
from solvers.cover_tree import estimate_span

from utilities.util import timer_func, ndarray_to_list
from utilities.data_io import load_npz_data
from pathlib import Path
from tqdm import tqdm

import os
import json
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from definitions import ROOT_DIR

class TrainerBase():
    def __init__(self, trainer_type, config_trainer):
        self.config = config_trainer
        self.trainer_type = trainer_type

    def load_training_data(self, datasize):
        indices_param = self.config['general']['indices_param']
        data_folder = os.path.join(ROOT_DIR, f"{self.config['files']['data_folder']}{indices_param}")
        data_filename = f"{self.config['files']['trdata_filename']}{datasize}.npz"
        return load_npz_data(data_folder, data_filename)

    def save_model(self, model, datasize):
        results_folder = os.path.join(ROOT_DIR, self.config['files']['results_folder']
                                      +f"_idx{self.config['general']['indices_param']}", self.trainer_type)
        Path(results_folder).mkdir(parents=True, exist_ok=True)

        model_new = {}
        model_new['model'] = ndarray_to_list(model)
        model_new['time'] = self.time_usage

        model_filename = f'model_N{datasize}.json'
        with open(os.path.join(results_folder, model_filename), 'w') as f:
            f.write(json.dumps(model_new))
        return


class StreamrakTrainer(TrainerBase):
    def __init__(self, config_solver, config_trainer):
        self.trainer_type = 'streamrak'
        super().__init__(self.trainer_type, config_trainer)

        self.config = config_trainer
        self.streamrak = Streamrak(config_solver)
        self.time_usage = {}

    def train(self):
        for datasize in tqdm(self.config['general']['training_data_sizes'], desc='Loop data sizes'):
            training_data = self.load_training_data(datasize)

            # Initializa cover-tree
            init_radius = estimate_span(training_data['x'])
            self.streamrak.initialize(init_radius)

            # Train streamrak
            _, train_time = self.streamrak.train(training_data['x'], training_data['y'], timeit=True)
            model = self.streamrak.get_model()
            self.time_usage['train_time'] = train_time

            self.save_model(model, datasize)
            self.streamrak.clear()
        return


class FalkonTrainer(TrainerBase):
    def __init__(self, config_solver, config_trainer):
        self.trainer_type = 'falkon'
        super().__init__(self.trainer_type, config_trainer)

        self.config = config_trainer
        self.falkon = Falkon(config_solver)
        self.time_usage = {}

    @timer_func
    def find_optimal_bw(self, training_data):
        bwmax, bwmin, size = self.config['falkon']['bw_grid']
        bw_grid = np.logspace(bwmax, bwmin, size)
        xtr, xval, ytr, yval = train_test_split(training_data['x'], training_data['y'], test_size=0.3, random_state=42)
        mse = []
        for bw in bw_grid:
            self.falkon.set_bw(bw)
            self.falkon.train(xtr, ytr)
            yval_pred = self.falkon.predict(xval)
            mse.append(mean_squared_error(yval, yval_pred))
        idx = np.argmin(mse)
        return bw_grid[idx]

    def train(self):
        for datasize in tqdm(self.config['general']['training_data_sizes'], desc='Loop data sizes'):
            training_data = self.load_training_data(datasize)

            # Select landmarks
            self.falkon.select_landmarks(training_data['x'])

            # Find bandwidth
            opt_bw, find_bw_time = self.find_optimal_bw(training_data)
            self.time_usage['find_bw_time'] = find_bw_time

            # Train falkon
            self.falkon.set_bw(opt_bw)
            print("Optimal bandwidth: ", opt_bw)
            _, train_time = self.falkon.train(training_data['x'], training_data['y'], timeit=True)
            model = self.falkon.get_model()
            self.time_usage['train_time'] = train_time

            self.save_model(model, datasize)
            self.falkon.clear()
        return

from definitions import CONFIG_SOLVERS_PATH, CONFIG_TRAINER_PATH
from config.yaml_functions import yaml_loader
config_solver = yaml_loader(CONFIG_SOLVERS_PATH)
config_training = yaml_loader(CONFIG_TRAINER_PATH)

if __name__ == '__main__':
    falkon_trainer = FalkonTrainer(config_solver, config_training)
    falkon_trainer.train()

    falkon_trainer = StreamrakTrainer(config_solver, config_training)
    falkon_trainer.train()


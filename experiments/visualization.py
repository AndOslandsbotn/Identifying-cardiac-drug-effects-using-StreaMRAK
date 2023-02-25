from utilities.data_io import load_npz_data, load_json_data, write_json
from utilities.plotstyle_maps import get_algo_names_map, get_color_map_models, get_dash_map_models, get_marker_map_models, get_color_map_selection

import os
import numpy as np
from definitions import ROOT_DIR
from pathlib import Path

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from matplotlib import ticker, cm

class VisualizeBase():
    """Base class for the experiment visualization"""
    def __init__(self, config):
        self.config = config
        self.indices_param = self.config['general']['indices_param']
        self.model_names = self.config['general']['model_names']

        self.pred_filenames = self.config['files']['pred_data_filenames']
        self.pred_data_folder = self.config['files']['pred_data_folder']

        self.figures_folder = os.path.join(ROOT_DIR, self.config['files']['figures_folder']+f"_idx{self.indices_param}")
        self.color_map_models = get_color_map_models()
        self.dash_map_models = get_dash_map_models()
        self.marker_map_models = get_marker_map_models()
        self.color_map_selection = get_color_map_selection()
        self.algo_names_map = get_algo_names_map()
        self.pred_data_dict = {}

    def load_training_data(self, datasize):
        indices_param = self.config['general']['indices_param']
        data_folder = os.path.join(ROOT_DIR, f"{self.config['files']['data_folder']}{indices_param}")
        data_filename = f"{self.config['files']['trdata_filename']}{datasize}.npz"
        return load_npz_data(data_folder, data_filename)

    def load_pred_data(self):
        for model_name in self.model_names:
            pred_data_folder = os.path.join(ROOT_DIR, f"{self.pred_data_folder}_idx{self.indices_param}", model_name)
            for pred_filename in self.pred_filenames:
                pred_data = load_npz_data(pred_data_folder, pred_filename+'.npz')
                if not model_name in self.pred_data_dict:
                    self.pred_data_dict[model_name] = {}
                    self.pred_data_dict[model_name][pred_filename] = pred_data
                else:
                    self.pred_data_dict[model_name][pred_filename] = pred_data

    def save_figure(self, fig, filename, transparent=True):
        Path(self.figures_folder).mkdir(parents=True, exist_ok=True)
        fig.savefig(os.path.join(self.figures_folder, filename+'.eps'))
        fig.savefig(os.path.join(self.figures_folder, filename+'.pdf'))
        fig.savefig(os.path.join(self.figures_folder, filename+'.svg'), format='svg', dpi=1200, transparent=transparent)


class VisualizeDomainPred(VisualizeBase):
    def __init__(self, config):
        super().__init__(config)
        self.training_data_sizes = np.array(config['general']['training_data_sizes'])
        self.model_folder = config['files']['model_folder']
        self.training_time_dict = {}
        self.load_pred_data()
        self.load_training_time_data()
        self.avg_apf_construction_time = config['training_time_experiment']['avg_apf_construct_time'] #0.00127
        self.avg_ode_solver_time = config['training_time_experiment']['avg_ode_solver_time']
        self.cutoff = config['domain_error_experiment']['cutoff']
        self.is_logscale = config['domain_error_experiment']['is_logscale']
        return

    def organize_mse(self, model_name, param_idx):
        data = []
        for pred_filename in self.pred_filenames:
            data.append(self.pred_data_dict[model_name][pred_filename]['nmse'][param_idx])
        return data

    def organize_training_time(self, model_name):
        training_time = []
        for training_data_size in self.training_data_sizes:
            if model_name == 'falkon':
                training_time.append(self.training_time_dict[model_name][training_data_size]['find_bw_time']
                                     + self.training_time_dict[model_name][training_data_size]['train_time'])
            elif model_name == 'streamrak':
                training_time.append(self.training_time_dict[model_name][training_data_size]['train_time'])
        return training_time

    def organize_pred_time(self, model_name):
        avg_pred_time = []
        for pred_filename in self.pred_filenames:
            xts = self.pred_data_dict[model_name][pred_filename]['xts']
            num_ts_samples = len(xts)
            avg_pred_time.append(self.pred_data_dict[model_name][pred_filename]['predtime']/num_ts_samples)
        return avg_pred_time

    def load_training_time_data(self):
        for model_name in self.model_names:
            if model_name in ['falkon', 'streamrak']:
                model_folder = os.path.join(ROOT_DIR, f"{self.model_folder}_idx{self.indices_param}", model_name)
                for training_data_size in self.training_data_sizes:
                    model_dict = load_json_data(os.path.join(model_folder, f'model_N{training_data_size}'+'.json'))
                    if not model_name in self.training_time_dict:
                        self.training_time_dict[model_name] = {}
                        self.training_time_dict[model_name][training_data_size] = model_dict['time']
                    else:
                        self.training_time_dict[model_name][training_data_size] = model_dict['time']

    def plot_mse_vs_number_of_samples(self):
        for param_idx in self.indices_param:
            fig, ax = plt.subplots()
            for model_name in self.model_names:
                mse = self.organize_mse(model_name, param_idx)
                plt.plot(self.training_data_sizes, mse,
                         label=f'{self.algo_names_map[model_name]}',
                         color = self.color_map_models[model_name],
                         dashes = self.dash_map_models[model_name],
                         marker= self.marker_map_models[model_name])
            plt.xlabel('Number of training samples')
            if param_idx == 1:
                plt.ylabel(r'NMSE (Sodium current $(g_{Na})$)')
                plt.yticks([1.e-1, 1.e-2, 1.e-3, 1.e-4, 1.e-5])
                plt.ylim([1.e-5, 3.e-1])
                plt.legend(ncol=2)
            elif param_idx == 3:
                plt.ylabel(r'NMSE (Slow inward current $(g_{s})$)')
                plt.yticks([1.e-2, 1.e-3, 1.e-4, 1.e-5])
                plt.ylim([1.e-5, 3.e-2])
            else:
                plt.ylabel(r'NMSE')
            plt.yscale('log')

            filename = f'mse_param{param_idx}'
            self.save_figure(fig, filename)

    def plot_training_time_vs_number_of_samples(self):
        fig, ax = plt.subplots()
        for model_name in self.model_names:
            if model_name in ['falkon', 'streamrak']:
                training_time = self.organize_training_time(model_name)
                plt.plot(self.training_data_sizes, training_time,
                         label=f'{self.algo_names_map[model_name]}',
                         color = self.color_map_models[model_name],
                         dashes = self.dash_map_models[model_name],
                         marker= self.marker_map_models[model_name])

        plt.plot(self.training_data_sizes, self.avg_apf_construction_time*self.training_data_sizes,
                 label=f'Apf-construction',
                 color=self.color_map_models['apf-knn1'],
                 dashes = self.dash_map_models['apf-knn1'],
                 marker= self.marker_map_models['apf-knn1'])
        plt.plot(self.training_data_sizes, self.avg_ode_solver_time*self.training_data_sizes,
                 label=f'Avg. time Beeler-Reuter',
                 color=self.color_map_models['reference'],
                 dashes=self.dash_map_models['reference'],
                 marker=self.marker_map_models['reference'])
        plt.yscale('log')
        plt.legend()
        plt.ylim([1e-2, 1.e5])
        plt.xlabel('Number of training samples')
        plt.ylabel('Training time')
        filename = f'training_time'
        self.save_figure(fig, filename)

    def plot_pred_time_vs_model_size(self):
        fig, ax = plt.subplots()
        for model_name in self.model_names:
            avg_pred_time = self.organize_pred_time(model_name)
            plt.plot(self.training_data_sizes, avg_pred_time,
                         label=f'{self.algo_names_map[model_name]}',
                         color = self.color_map_models[model_name],
                         dashes = self.dash_map_models[model_name],
                         marker= self.marker_map_models[model_name])
        plt.ylim([1.e-6, 1.e-2])
        plt.yticks([1.e-2, 1.e-3, 1.e-4, 1.e-5, 1.e-6])
        plt.yscale('log')
        plt.legend(ncol=2)
        plt.xlabel('Model size (In number of training samples)')
        plt.ylabel('Prediction time')
        filename = f'pred_time'
        self.save_figure(fig, filename)


    def plot_domain_errors(self):
        labels = ['A', 'B', 'C', 'D', 'E', 'F']
        pred_data_filename = self.config['domain_error_experiment']['pred_data_filename']

        training_data = self.load_training_data(self.config['domain_error_experiment']['datasize'])
        tr_param = training_data['y']

        idx1 = self.indices_param[0]
        idx2 = self.indices_param[1]

        xticks = [0.2, 0.5, 0.8, 1.1, 1.4, 1.7, 2]
        yticks = [0.2, 0.5, 0.8, 1.1, 1.4, 1.7, 2]

        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        j = 0
        for i, model_name in enumerate(self.model_names):

            if i > 2:
                j = 1
            ax = axes[j, i % 3]
            ax.set_xticks(xticks)
            ax.set_yticks(yticks)
            ax.set_ylim([0.2, 2])
            ax.set_xlim([0.2, 2])

            data = self.pred_data_dict[model_name][pred_data_filename]
            point_vise_error = abs(np.sum(data['yts'] - data['yts_pred'], axis=1))
            idx = np.where(point_vise_error <= self.cutoff)[0]
            point_vise_error = point_vise_error[idx]
            param = data['yts'][idx, :]

            if self.is_logscale:
                sc = ax.tricontourf(param[:, idx1], param[:, idx2], abs(point_vise_error + 10**(-14)), locator=ticker.LogLocator(), cmap=cm.PuBu_r)
            else:
                sc = ax.tricontourf(param[:, idx1], param[:, idx2], abs(point_vise_error), cmap=cm.PuBu_r, vmin=1.e-5, vmax=0.1)


            if i == 0 and j == 0:
                ax.legend()

            # Share x and y labels
            if i % 3 == 0:
                ax.set_ylabel('Slow inward current $(g_s)$')
            if j == 1:
                ax.set_xlabel(r'Sodium current $(g_{Na})$')

            if j == 0:
                ax.text(0.3, -0.25, f'({labels[i]}) {self.algo_names_map[model_name]}', transform=ax.transAxes, size=15)
            elif j == 1:
                ax.text(0.3, -0.35, f'({labels[i]}) {self.algo_names_map[model_name]}', transform=ax.transAxes, size=15)

        fig.subplots_adjust(top=0.99, bottom=0.2, wspace=0.3, hspace=0.4, left=0.1, right=0.8)
        cbar_ax = fig.add_axes([0.85, 0.25, 0.05, 0.7])
        clb = fig.colorbar(sc, cax=cbar_ax)
        clb.ax.set_ylabel(f'Normalized MSE', position=(1.6, 0.5), labelpad=20, rotation=270)

        if self.is_logscale:
            filename = f'domain_error_combined_log'
        else:
            filename = f'domain_error_combined'
        self.save_figure(fig, filename)

    def plot_domain_errors_individual_plots(self):
        pred_data_filename = self.config['domain_error_experiment']['pred_data_filename']
        for model_name in self.model_names:
            data = self.pred_data_dict[model_name][pred_data_filename]
            point_vise_error = abs(np.sum(data['yts'] - data['yts_pred'], axis=1))
            idx = np.where(point_vise_error <= self.cutoff)[0]
            point_vise_error = point_vise_error[idx]
            param = data['yts'][idx, :]

            fig = plt.figure()
            ax = fig.add_subplot()
            idx1 = self.indices_param[0]
            idx2 = self.indices_param[1]
            if self.is_logscale:
                sc = plt.tricontourf(param[:, idx1], param[:, idx2], abs(point_vise_error + 10**(-14)), locator=ticker.LogLocator(), cmap=cm.PuBu_r)
            else:
                sc = plt.tricontourf(param[:, idx1], param[:, idx2], abs(point_vise_error), cmap=cm.PuBu_r, vmin=0, vmax=0.1)
                plt.clim(0, 0.1)

            ax.set_xlabel(r'Sodium current $(g_{Na})$')
            ax.set_ylabel('Slow invard current $(g_s)$')

            clb = plt.colorbar(sc)
            #clb.ax.tick_params(labelsize=8)
            #clb.ax.set_title(f'MSE {model_name}', position=(1, 10))
            clb.ax.get_yaxis().labelpad = 21
            clb.ax.set_ylabel(f'MSE {self.algo_names_map[model_name]}', rotation=270)

            if self.is_logscale:
                filename = f'domain_error_{model_name}_log'
            else:
                filename = f'domain_error_{model_name}'
            self.save_figure(fig, filename)



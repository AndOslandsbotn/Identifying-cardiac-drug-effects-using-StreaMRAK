from action_potential_model import beeler_reuter_model
from action_potential_model.beeler_reuter_model import beelerReuterTimeGrid, rescale_constants
import numpy as np
import itertools
import os
from pathlib import Path
from definitions import ROOT_DIR
from utilities.data_io import load_npz_data, write_json
from utilities.util import normalize, compute_cost_terms_traces, cart2pol, pol2cart
from scipy.io import savemat

def generate_voltage_curves(parameters):
    """This function takes parameter vector and generates a voltage curve using the beeler retuer model
    :param parameters: An n x 10 numpy array of paramters
    """
    states, parameters = beeler_reuter_model.sample(parameters, [])
    voltages = []
    for state in states:
        v = state[beeler_reuter_model.state_indices("V")]
        voltages.append(v)
    return np.array(voltages), np.array(parameters)

class DataGenerator():
    def __init__(self, config):
        self.config = config
        self.param_indices = config['general']['indices_param']
        self.num_samples_per_param = config['general']['num_samples_per_param']
        self.init_param = np.array(config['general']['init_param']).astype(float)
        self.param_range = config['general']['param_range']

        self.data_folder = config['files']['data_folder']
        self.data_filename = config['files']['data_filename']

    def generate_voltage_curves(self, parameters):
        """This function takes parameter vector and generates a voltage curve using the beeler retuer model
        :param parameters: An n x 10 numpy array of paramters
        """
        states, parameters = beeler_reuter_model.sample(parameters, [])
        voltages = []
        for state in states:
            v = state[beeler_reuter_model.state_indices("V")]
            voltages.append(v)
        return np.array(voltages), np.array(parameters)

    def uniformly_sample_from_domain(self):
        param = np.random.uniform(low=self.param_range[0],
                                  high=self.param_range[1],
                                  size=(self.num_samples_per_param**len(self.param_indices), len(self.param_indices)))
        parameters = np.tile(self.init_param.reshape(-1, 1), self.num_samples_per_param ** len(self.param_indices)).transpose()
        parameters[:, self.param_indices] = param
        return parameters

    def generate_dataset(self):
        parameters = self.uniformly_sample_from_domain()
        self.voltages, self.parameters = self.generate_voltage_curves(parameters)
        self.save()

    def save(self):
        data_folder = os.path.join(ROOT_DIR, self.config['files']['data_folder'] + f"_idx{self.param_indices}")
        Path(data_folder).mkdir(parents=True, exist_ok=True)

        filename = f"{self.data_filename}_N{self.num_samples_per_param**len(self.param_indices)}"
        np.savez(os.path.join(data_folder, filename),
                 voltages=self.voltages, parameters=self.parameters)


class TestTrainDataGenerator(DataGenerator):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.data = load_npz_data(self.config['files']['data_folder']+f"_idx{self.config['general']['indices_param']}",
                                  self.config['files']['data_filename']
                                  + f"_N{self.config['general']['num_samples_per_param']**len(self.config['general']['indices_param'])}.npz")
        self.time_grid = beelerReuterTimeGrid()
        self.xtr, self.xts, self.ytr, self.yts = self.test_train_splitt()

    def test_train_splitt(self):
        x = self.data['voltages']
        y = self.data['parameters']

        n, d = y.shape
        indices = np.arange(n)
        np.random.seed(0)
        np.random.shuffle(indices)
        idx_tr = indices[0:int(n * 0.7)]
        idx_ts = indices[int(n * 0.7):]
        xtr = x[idx_tr]
        xts = x[idx_ts]

        ytr = y[idx_tr]
        yts = y[idx_ts]
        return xtr, xts, ytr, yts

    def generate_features(self, x):
        x_norm = normalize(x)
        apf, apf_time = compute_cost_terms_traces(x, x, self.time_grid)
        apf = apf[:, 0:30]
        apf_norm = normalize(apf)
        return x_norm, apf, apf_norm, apf_time

    def extract_samples(self, xtr, ytr, num_training_samples):
        n, d = xtr.shape
        idx = np.random.choice(np.arange(0, n), num_training_samples, replace=False)
        return xtr[idx], ytr[idx]

    def generate_training_samples(self):
        for num_training_samples in self.config['domain_experiment']['num_train_samples']:
            x, y = self.extract_samples(self.xtr, self.ytr, num_training_samples)
            x_norm, apf, apf_norm, apf_time = self.generate_features(x)
            filename = f"TrData_N{num_training_samples}.npz"
            self.save(filename, x, y, x_norm, apf, apf_norm, apf_time)

    def generate_test_samples(self):
        x_norm, apf, apf_norm, apf_time = self.generate_features(self.xts)
        filename = f"TsData.npz"
        self.save(filename, self.xts, self.yts, x_norm, apf, apf_norm, apf_time)

    def save(self, filename, x, y, x_norm, apf, apf_norm, apf_time):
        data_folder = os.path.join(ROOT_DIR, self.config['files']['data_folder'] + f"_idx{self.param_indices}")
        Path(data_folder).mkdir(parents=True, exist_ok=True)
        np.savez(os.path.join(data_folder, filename),
                 x=x, y=y, apf = apf, xnorm = x_norm,
                 apfnorm = apf_norm, apftime = apf_time)


class PerturbationDataGenerator(DataGenerator):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.radius_arr = self.config['perturbation_experiment']['radius']
        self.param_center = np.array(self.config['perturbation_experiment']['param_center']).astype(float).reshape(1, -1)
        self.sample_sizes = self.config['perturbation_experiment']['sample_size']
        self.desired_angles = np.array(self.config['selected_ap_curves_experiment']['desired_angles'])

    def generate_pert_circle_angles(self, radius):
        sample_size = int(100 * 2 * np.pi * radius)
        if self.config['perturbation_experiment']['is_grid'] == True:
            angles = 2 * np.pi * np.linspace(0, 1, sample_size)
        else:
            angles = 2 * np.pi * np.random.uniform(low=0.0, high=1.0, size=sample_size)
        return angles

    def expand(self, x, y):
        parameters = np.repeat(self.param_center, repeats=len(x), axis=0).transpose()
        parameters[1, :] += x
        parameters[3, :] += y
        return parameters.transpose()

    def reduce(self, parameters):
        return parameters[:, self.param_indices[0]], parameters[:, self.param_indices[1]]

    def find_polar_from_param(self, paramters):
        x, y = self.reduce(paramters)
        x0 = self.param_center[0, self.param_indices[0]]
        y0 = self.param_center[0, self.param_indices[1]]
        radius, angles = cart2pol(x - x0, y - y0)
        angles = np.degrees(angles)
        return radius, angles

    def generate_center(self):
        v_center, param_center = self.generate_voltage_curves(self.param_center)
        apf_center, _ = compute_cost_terms_traces(v_center, v_center, beelerReuterTimeGrid())
        return param_center, v_center[:, 0:30], apf_center[:, 0:30]

    def generate_selected_ap_curves(self, radius):
        param_selected_ap = np.array(pol2cart(np.radians(self.desired_angles), radius))
        param_selected_ap = self.expand(param_selected_ap[0, :], param_selected_ap[1, :])
        v_selected_ap, param_selected_ap = self.generate_voltage_curves(param_selected_ap)
        apf_selected_ap, _ = compute_cost_terms_traces(v_selected_ap, v_selected_ap, beelerReuterTimeGrid())

        if len(param_selected_ap) < len(self.desired_angles):
            #  If the ODE solver has failed at producing some of the curves, we need to manually calculate the angles
            #  for the curves that was produced successfully
            radius, angles = self.find_polar_from_param(param_selected_ap)
        else:
            angles = self.desired_angles
        return param_selected_ap, v_selected_ap, apf_selected_ap, angles

    def generate_pert_circle(self, radius):
        desired_angles = self.generate_pert_circle_angles(radius)
        param_pert_circle = np.array(pol2cart(desired_angles, radius))
        param_pert_circle = self.expand(param_pert_circle[0, :], param_pert_circle[1, :])
        v_pert_circle, param_pert_circle = self.generate_voltage_curves(param_pert_circle)
        apf_pert_circle, _ = compute_cost_terms_traces(v_pert_circle, v_pert_circle, beelerReuterTimeGrid())

        if len(param_pert_circle) < len(desired_angles):
            #  If the ODE solver has failed at producing some of the curves, we need to manually calculate the angles
            #  for the curves that was produced successfully
            radius, angles = self.find_polar_from_param(param_pert_circle)
        else:
            angles = desired_angles
        return param_pert_circle, v_pert_circle, apf_pert_circle, angles

    def generate_dataset(self):
        param_center, v_center, apf_center = self.generate_center()
        for circle_nr, radius in enumerate(self.radius_arr):
            param_pert_circle, v_pert_circle, apf_pert_circle, angles_pert_circle = self.generate_pert_circle(radius)
            param_selected_ap, v_selected_ap, apf_selected_ap, angles_selected_ap = self.generate_selected_ap_curves(radius)

            filename = f'drugeff_c1pert{circle_nr}.npz'
            self.save_data(filename, param_center, param_pert_circle, v_center,
                           v_pert_circle, apf_center, apf_pert_circle,
                           param_selected_ap, v_selected_ap, apf_selected_ap, angles_selected_ap)

    def save_data(self, filename, param_center, param_pert_circle, v_center, v_pert_circle, apf_center,
                  apf_pert_circle, param_selected_ap, v_selected_ap, apf_selected_ap, angles_selected_ap):
        results_folder = os.path.join(ROOT_DIR, self.config['perturbation_experiment']['results_folder'] \
                         + f"_idx{self.config['general']['indices_param']}")
        Path(results_folder).mkdir(parents=True, exist_ok=True)
        np.savez(os.path.join(results_folder, filename), param_center=param_center, param_pert_circle=param_pert_circle,
                 v_center=v_center, v_pert_circle=v_pert_circle, apf_center=apf_center, apf_pert_circle=apf_pert_circle,
                 param_selected_ap=param_selected_ap, v_selected_ap=v_selected_ap,
                 apf_selected_ap=apf_selected_ap, angles_selected_ap=angles_selected_ap)


class SelectedAPCurves():
    def __init__(self, config):
        self.config = config
        self.angle_directions = self.config['selected_ap_curves_experiment']['desired_angles']
        self.radius_interval = self.config['selected_ap_curves_experiment']['radius_interval']
        self.num_samples_along_direction = self.config['selected_ap_curves_experiment']['num_samples_along_direction']
        self.radiuss = np.linspace(self.radius_interval[0], self.radius_interval[1], self.num_samples_along_direction)

        self.p_center = np.ones((1, 10))
        self.v_center, _ = self.generate_voltage_curves(self.p_center)

        self.drug_directions = self.config['selected_drug_exp']['drug_directions']
        self.drug_pert_radius = self.config['selected_drug_exp']['drug_pert_radius']
        self.drug_noise = self.config['selected_drug_exp']['drug_noise']
        self.num_samples_per_drug = self.config['selected_drug_exp']['num_samples_per_drug']
        return

    def expand(self, x, y):
        parameters = np.repeat(self.p_center, repeats=len(x), axis=0).transpose()
        parameters[1, :] += x
        parameters[3, :] += y
        return parameters.transpose()

    def generate_voltage_curves(self, parameters):
        """This function takes parameter vector and generates a voltage curve using the beeler retuer model
        :param parameters: An n x 10 numpy array of paramters
        """
        states, parameters = beeler_reuter_model.sample(parameters, [])
        voltages = []
        for state in states:
            v = state[beeler_reuter_model.state_indices("V")]
            voltages.append(v)
        return np.array(voltages), np.array(parameters)

    def generate_parameters_along_direction(self, angle):
        angles = angle*np.ones(len(self.radiuss))
        x, y = pol2cart(np.radians(angles), self.radiuss)
        parameters = self.expand(x, y)
        return parameters

    def generate_drug_perturbations(self, angle):
        radius = self.drug_pert_radius + np.random.normal(0, self.drug_noise, self.num_samples_per_drug)
        radius = list(radius)
        radius.append(self.drug_pert_radius)
        radius = np.array(radius)
        angles = angle * np.ones(len(radius))
        x, y = pol2cart(np.radians(angles), radius)
        parameters = self.expand(x, y)
        return parameters

    def generate_selected_ap_along_directions(self):
        for angle in self.angle_directions:
            radius_list = []
            print("Direction: ", angle)
            param = self.generate_parameters_along_direction(angle)
            voltages, parameters = self.generate_voltage_curves(param)
            apf, _ = compute_cost_terms_traces(voltages, voltages, beelerReuterTimeGrid())
            apf = apf[:, 0:30]

            for param in parameters:
                radius, _ = cart2pol(param[1]-1, param[3]-1)
                radius_list.append(radius)

            radius_array = np.array(radius_list)
            filename = f'selected_ap_along_direction{angle}'
            self.save_data(filename, voltages, apf, parameters, radius_array)

    def generate_selected_drug_perturbations(self):
        for angle in self.drug_directions:
            radius_list = []
            print("Direction: ", angle)
            param = self.generate_drug_perturbations(angle)
            voltages, parameters = self.generate_voltage_curves(param)
            apf, _ = compute_cost_terms_traces(voltages, voltages, beelerReuterTimeGrid())

            for param in parameters:
                radius, _ = cart2pol(param[1] - 1, param[3] - 1)
                radius_list.append(radius)

            radius_array = np.array(radius_list)
            filename = f'drug_direction{angle}'

            results_folder = os.path.join(ROOT_DIR, self.config['selected_drug_exp']['results_folder'] \
                                          + f"_idx{self.config['general']['indices_param']}")
            Path(results_folder).mkdir(parents=True, exist_ok=True)

            np.savez(os.path.join(results_folder, filename),
                     parameters=parameters,
                     voltages=voltages,
                     apf = apf
                     )
        return


    def save_data(self, filename, voltages, apf, parameters, radius_array):
        results_folder = os.path.join(ROOT_DIR, self.config['selected_ap_curves_experiment']['results_folder'] \
                         + f"_idx{self.config['general']['indices_param']}")
        Path(results_folder).mkdir(parents=True, exist_ok=True)

        np.savez(os.path.join(results_folder, filename),
                 radiuss=radius_array,
                 p_center=self.p_center,
                 v_center=self.v_center,
                 parameters=parameters,
                 voltages=voltages,
                 apf=apf
                 )
        return

class GenerateCurrentMatrix():
    def __init__(self):
        self.proportionality_constant = rescale_constants()[2] / rescale_constants()[1]

        self.default_parameters = np.ones(10).reshape(1, -1)
        self.states, self.parameters = beeler_reuter_model.sample(self.default_parameters, [])
        self.parameters = self.parameters[0]
        self.states = self.states[0]
        self.m = self.states[beeler_reuter_model.state_indices("m")]
        self.h  = self.states[beeler_reuter_model.state_indices("h")]
        self.j  = self.states[beeler_reuter_model.state_indices("j")]
        self.Cai = self.states[beeler_reuter_model.state_indices("Cai")]
        self.d  = self.states[beeler_reuter_model.state_indices("d")]
        self.f  = self.states[beeler_reuter_model.state_indices("f")]
        self.x1 = self.states[beeler_reuter_model.state_indices("x1")]
        self.V = self.states[beeler_reuter_model.state_indices("V")]

        self.i_Na = None
        self.i_inn = None
        self.i_out = None

        self.u = None
        self.v = None
        self.s = None
        self.s_norm = None

        return

    def generate_voltage_curves(self, parameters):
        """This function takes parameter vector and generates a voltage curve using the beeler retuer model
        :param parameters: An n x 10 numpy array of paramters
        """
        states, parameters = beeler_reuter_model.sample(parameters, [])
        voltages = []
        for state in states:
            v = state[beeler_reuter_model.state_indices("V")]
            voltages.append(v)
        return np.array(voltages), np.array(parameters)

    def sample_currents(self):
        self.generate_currents()
        self.save_currents()

    def generate_currents(self):
        E_Na = self.parameters[0]
        E_s = -82.3 - 13.0287 * np.log(0.001 * self.Cai)

        # Sodium current
        self.i_Na = self.parameters[1]*(self.proportionality_constant + self.m*self.m*self.m*self.h*self.j)*(self.V-E_Na)

        # Slow inward current
        self.i_inn = self.parameters[3]*(self.V-E_s)*self.d*self.f

        # Outward current
        self.i_out = 0.0019727757115328517* (21.75840239619708 * np.exp(0.04 * self.V) - 1) * np.exp(-0.04 * self.V) * self.x1

    def generate_perturbations_along_eigen_directions(self):
        num_samples_along_direction = 9
        perturbation_grid = np.linspace(-0.5, 0.5, num_samples_along_direction)

        for direction_nr in range(0, 2):
            perturbation_grid_projection = np.outer(self.v[direction_nr, :], perturbation_grid).transpose()

            perturb_direction = np.tile(self.default_parameters.reshape(-1, 1), num_samples_along_direction).transpose()
            perturb_direction[:, [1, 3]] += perturbation_grid_projection
            voltage_perturb_direction, param_perturb_direction = self.generate_voltage_curves(perturb_direction)
            apf_perturb_direction, _ = compute_cost_terms_traces(voltage_perturb_direction,
                                                                 voltage_perturb_direction,
                                                                 beelerReuterTimeGrid())

            filename = f'pert_direction{direction_nr}.npz'
            self.save_perturbations(filename,
                                    voltage_perturb_direction,
                                    apf_perturb_direction,
                                    param_perturb_direction,
                                    perturbation_grid, self.v[direction_nr, :])

    def spectral_analysis(self):
        A = np.array([self.i_Na, self.i_inn, self.i_out]).transpose()
        #A = np.array([self.i_Na, self.i_inn]).transpose()
        self.u, self.s, vh = np.linalg.svd(A)
        self.s_norm = self.s/np.max(self.s)
        self.v = vh.transpose()
        self.v = self.v/np.sum(abs(self.v), axis=0)
        self.save_spectral_info()
        return

    def save_spectral_info(self):
        folder = os.path.join(ROOT_DIR, 'Data', 'Identifiability_analysis')
        Path(folder).mkdir(parents=True, exist_ok=True)

        path = os.path.join(folder, 'svd_current_matrix.npz')
        np.savez(path,
                 u=self.u,
                 v=self.v,
                 s=self.s,
                 s_norm=self.s_norm,
                 )
        dictionary = {}
        dictionary['v'] = self.v.tolist()
        dictionary['s'] = self.s.tolist()
        dictionary['s_norm'] = self.s_norm.tolist()
        path = os.path.join(folder, 'svd_current_matrix.json')
        write_json(path, dictionary)

    def save_currents(self):
        current_dictionary = {}
        current_dictionary['i_Na'] = self.i_Na
        current_dictionary['i_inn'] = self.i_inn
        current_dictionary['i_out'] = self.i_out

        folder = os.path.join(ROOT_DIR, 'Data', 'Identifiability_analysis')
        Path(folder).mkdir(parents=True, exist_ok=True)

        path = os.path.join(folder, 'currents_unpert_param.mat')
        savemat(path, current_dictionary)

        currents_array = np.array([self.i_Na, self.i_inn, self.i_out])
        path = os.path.join(folder, 'currents_unpert_param.npz')
        np.savez(path, currents_array=currents_array)

    def save_perturbations(self, filename, voltages, apf, parameters, perturbations, direction):
        results_folder = os.path.join(ROOT_DIR, 'Data', 'Identifiability_analysis')
        Path(results_folder).mkdir(parents=True, exist_ok=True)

        np.savez(os.path.join(results_folder, filename),
                 direction = direction,
                 perturbations=perturbations,
                 parameters=parameters,
                 voltages=voltages,
                 apf=apf
                 )


from definitions import CONFIG_GENERATE_DATA
from config.yaml_functions import yaml_loader
config_generate_data = yaml_loader(CONFIG_GENERATE_DATA)

if __name__ == '__main__':
    #dataGenerator = DataGenerator(config_generate_data)
    #dataGenerator.generate_dataset()

    #pertDataGenerator = PerturbationDataGenerator(config_generate_data)
    #pertDataGenerator.generate_dataset()

    selectedApCurves = SelectedAPCurves(config_generate_data)
    selectedApCurves.generate_selected_ap_along_directions()
    #selectedApCurves.generate_selected_drug_perturbations()

    #generateCurrentMatrix = GenerateCurrentMatrix()
    #generateCurrentMatrix.sample_currents()
    #generateCurrentMatrix.spectral_analysis()
    #generateCurrentMatrix.generate_perturbations_along_eigen_directions()
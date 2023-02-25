from experiments.visualization import VisualizeDomainPred

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from definitions import CONFIG_MATPLOTSTYLE_PATH
plt.style.use(CONFIG_MATPLOTSTYLE_PATH)

from definitions import CONFIG_VIZ_DOMAIN_PRED_PATH
from config.yaml_functions import yaml_loader
config_viz_domain_pred = yaml_loader(CONFIG_VIZ_DOMAIN_PRED_PATH)

if __name__ == '__main__':
    domain_predictions = VisualizeDomainPred(config_viz_domain_pred)
    domain_predictions.plot_mse_vs_number_of_samples()
    domain_predictions.plot_pred_time_vs_model_size()
    domain_predictions.plot_training_time_vs_number_of_samples()
    domain_predictions.plot_domain_errors()
    plt.show()
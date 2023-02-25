import os
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))  # Project Root
CONFIG_GENERATE_DATA = os.path.join(ROOT_DIR, 'config', 'config_generate_data.yaml')
CONFIG_MATPLOTSTYLE_PATH = os.path.join(ROOT_DIR, 'config', 'stylesheet.mplstyle')
CONFIG_SOLVERS_PATH = os.path.join(ROOT_DIR, 'config', 'config_solvers.yaml')
CONFIG_TRAINER_PATH = os.path.join(ROOT_DIR, 'config', 'config_training.yaml')
CONFIG_DOMAIN_PRED_PATH = os.path.join(ROOT_DIR, 'config', 'config_domain_pred.yaml')
CONFIG_VIZ_DOMAIN_PRED_PATH = os.path.join(ROOT_DIR, 'config', 'config_viz_domain_pred.yaml')
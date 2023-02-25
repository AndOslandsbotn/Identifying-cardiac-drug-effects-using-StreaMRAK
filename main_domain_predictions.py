from experiments.experiment_models import DomainExperiment, PredictionTimeExperiment
from definitions import CONFIG_SOLVERS_PATH, CONFIG_DOMAIN_PRED_PATH
from config.yaml_functions import yaml_loader

config_solver = yaml_loader(CONFIG_SOLVERS_PATH)
config_domain_pred = yaml_loader(CONFIG_DOMAIN_PRED_PATH)

if __name__ == '__main__':
    model_names = ["streamrak", "falkon", "euc-knn", "apf-knn"]
    for model_name in model_names:
        print(model_name)
        if model_name == "euc-knn1":
            d = 3
        domainExperiment = DomainExperiment(model_name, config_solver, config_domain_pred)
        test_data_filename = f"{config_domain_pred['files']['test_data_filenames']}"
        domainExperiment.load_test_data(test_data_filename)
        domainExperiment.predict()

        predtimeExperiment = PredictionTimeExperiment(model_name, config_solver, config_domain_pred)
        test_data_filename = f"{config_domain_pred['prediction_time_exp']['test_data_filename']}"
        predtimeExperiment.load_test_data(test_data_filename)
        predtimeExperiment.predict()
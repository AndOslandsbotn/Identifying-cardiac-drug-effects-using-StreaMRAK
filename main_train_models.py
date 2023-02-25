from experiments.trainers import FalkonTrainer, StreamrakTrainer

from definitions import CONFIG_SOLVERS_PATH, CONFIG_TRAINER_PATH
from config.yaml_functions import yaml_loader
config_solver = yaml_loader(CONFIG_SOLVERS_PATH)
config_training = yaml_loader(CONFIG_TRAINER_PATH)

if __name__ == '__main__':
    falkon_trainer = FalkonTrainer(config_solver, config_training)
    falkon_trainer.train()

    streamrak_trainer = StreamrakTrainer(config_solver, config_training)
    streamrak_trainer.train()

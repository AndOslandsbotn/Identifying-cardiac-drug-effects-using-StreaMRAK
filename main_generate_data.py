from utilities.generate_data import DataGenerator, TestTrainDataGenerator
from definitions import CONFIG_GENERATE_DATA
from config.yaml_functions import yaml_loader
config_generate_data = yaml_loader(CONFIG_GENERATE_DATA)

if __name__ == '__main__':
    #data_generator = DataGenerator(config_generate_data)
    #data_generator.generate_dataset()

    test_train_generator = TestTrainDataGenerator(config_generate_data)
    test_train_generator.generate_training_samples()
    test_train_generator.generate_test_samples()

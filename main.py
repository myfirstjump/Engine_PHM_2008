from py_module.config import Configuration
from py_module.data_reader import DataReader
from py_module.data_preprocessing import DataProprocessing
from py_module.data_exploration import DataExploration
from py_module.learning_definition import LearningDefinition
from py_module.data_training import DataTraining
from py_module.plot_module import PlotDesign
from py_module.data_evaluation import DataEvaluation
from py_module.data_training_tf18 import DataTrainingTF18

import os

class EngineCycleTraining(object):

    """
    Main Flow:
    1. data loading
    2. data preprocessing
    3. data training
    """

    # Preprocessing
    #   Define RUL
    #   Standardization
    # Feature Extraction
    #   AE
    # RUL Prediction

    def __init__(self):
        self.config_obj = Configuration()
        self.reader_obj = DataReader()
        self.data_preprocessing_obj = DataProprocessing()
        self.learning_define_obj = LearningDefinition()
        self.data_exploration_obj = DataExploration()
        self.learing_def_obj = LearningDefinition()
        self.training_obj = DataTraining()
        self.plotting_obj = PlotDesign()
        self.evaluation_obj = DataEvaluation()
        self.training_tf18_obj = DataTrainingTF18()

    def data_loading(self):
        file_path = os.path.join(self.config_obj.data_folder, self.config_obj.file_name)
        data = self.reader_obj.read_csv_data(file_path)
        

        test_file_path = os.path.join(self.config_obj.test_data_folder, self.config_obj.test_file_name)
        testing_data = self.reader_obj.read_csv_data(test_file_path)
        return data, testing_data

    def data_exploration(self, data):

        self.data_exploration_obj.data_exploration_2008_PHM_Engine_data(data)

    def data_preprocessing(self, data):
        
        data = self.data_preprocessing_obj.data_preprocessing_2008_PHM_Engine_data(data, self.config_obj.features_name)
        data = self.data_preprocessing_obj.features_standardization(data, self.config_obj.standardization_features)

        return data
    
    def learning_define(self, data):
        
        ### PHM 2008 Engine, 

        new_data = self.learning_def_obj.learning_define_2008_PHM_Engine_data(data)        

        return new_data

    def model_training(self, data):

        my_history = self.training_obj.training_2008_PHM_Engine_data(data, epochs=30, load_model = False)
    #     hyperparameters = {'batch_szie':[8,16,32,64, 128], "epochs":[10, 30, 50, 100, 200, 300], }
    #     self.training_obj.grid_search_with_cross_validation(hyperparameters, "data", "cv_fold", "preprocessing_function", "training_function", "scoring_metric")
        
        return my_history

    def plotting_function(self, obj):

        self.plotting_obj.learning_curve(obj)

    def data_evaluation(self, test_data):
        
        self.evaluation_obj.data_evaluation_2008_PHM_Engine_data(test_data)

    def data_training_tf18(self, data):

        self.training_tf18_obj.training_PHM_2008_Engine_data(data, model_string="RNN")



def main_flow():
    
    main_obj = EngineCycleTraining()
    
    data, testing_data = main_obj.data_loading()
    data = main_obj.data_preprocessing(data)
    testing_data = main_obj.data_preprocessing(testing_data)
    # main_obj.data_exploration(data)
    print(data)
    # Training
    my_history = main_obj.model_training(data)
    main_obj.plotting_function(my_history)

    ### Evaluation
    # while(True):
    #     main_obj.data_evaluation(testing_data)
    main_obj.data_evaluation(testing_data)

    ### Code Test
    # main_obj.data_training_tf18(data)


if __name__ == "__main__":
    main_flow()


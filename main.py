from py_module.config import Configuration
from py_module.data_reader import DataReader
from py_module.data_preprocessing import DataProprocessing
<<<<<<< HEAD
from py_module.learning_definition import LearningDefinition

import os

class EngineCycleTraining(object):

    """
    Main Flow:
    1. data loading
    2. data preprocessing
    3. data training
    """
=======
from py_module.data_exploration import DataExploration

import os


    # 小波函數(wavelet) 可以去除雜訊, 諧波基函數, EMD(empirical mode decomposition)???
    # ARIMA + NN
    # DLSTM
    # 模型結構撰寫，擷取AE前半段Encoder部分加入模型

class EngineCyclePrediction(object):
>>>>>>> master

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
<<<<<<< HEAD
        self.learning_define_obj = LearningDefinition()
=======
        self.data_exploration_obj = DataExploration()
>>>>>>> master

    def data_loading(self):
        file_path = os.path.join(self.config_obj.data_folder, self.config_obj.file_name)
        data = self.reader_obj.read_csv_data(file_path)
<<<<<<< HEAD
        
=======

        test_file_path = os.path.join(self.config_obj.test_data_folder, self.config_obj.test_file_name)
        testing_data = self.reader_obj.read_csv_data(test_file_path)
>>>>>>> master
        return data

    def data_preprocessing(self, data):
        
<<<<<<< HEAD
        data = self.data_preprocessing_obj.data_col_rename(data, self.config_obj.features_name)
        data = self.data_preprocessing_obj.data_col_remove(data, ['sensor_22', "sensor_23"])

        return(data)
    
    def learning_define(self, data):
        
        ### PHM 2008 Engine, 
        # define RUL which is true y. 
        data = self.learning_define_obj.learning_define_2008_PHM_Engine_data(data)        
        # train, test split
        '''
        將引擎unit分為五份，一份當作測試集，其他四份為訓練集。
        '''
        train_data, test_data = self.learning_define_obj.train_test_split_2008_PHM_Engine_data(data)

        return train_data, test_data

    def data_queue(self, data):
        '''
        function:
            基於PHM2008引擎資料有218個引擎，在餵入RNN時，以每個引擎為單位，採隨機抽樣。
            此函數每次隨機抽出一個引擎unit資料，並且進行pre timestep加工後輸出。
        '''

        # 1. 抽出unit
        


=======
        data = self.data_preprocessing_obj.data_preprocessing_2008_PHM_Engine_data(data, self.config_obj.features_name)
        data = self.data_preprocessing_obj.features_standardization(data, self.config_obj.standardization_features)
        
        return(data)
>>>>>>> master

    def data_exploration(self, data):
        
        self.data_exploration_obj.data_exploration_2008_PHM_Engine_data(data)

def main_flow():
    
    main_obj = EngineCycleTraining()
    data = main_obj.data_loading()
    data = main_obj.data_preprocessing(data)
<<<<<<< HEAD
    train_data, test_data = main_obj.learning_define(data)

    train_data_unit = main_obj.data_queue(train_data)

    # print(data)
=======
    main_obj.data_exploration(data)
    print(data)
>>>>>>> master

if __name__ == "__main__":
    main_flow()


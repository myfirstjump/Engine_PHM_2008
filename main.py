from py_module.config import Configuration
from py_module.data_reader import DataReader
from py_module.data_preprocessing import DataProprocessing
from py_module.learning_definition import LearningDefinition

import os

class EngineCycleTraining(object):

    """
    Main Flow:
    1. data loading
    2. data preprocessing
    3. data training
    """

    def __init__(self):
        self.config_obj = Configuration()
        self.reader_obj = DataReader()
        self.data_preprocessing_obj = DataProprocessing()
        self.learning_define_obj = LearningDefinition()

    def data_loading(self):
        file_path = os.path.join(self.config_obj.data_folder, self.config_obj.file_name)
        data = self.reader_obj.read_csv_data(file_path)
        
        return data

    def data_preprocessing(self, data):
        
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
        




def main_flow():
    
    main_obj = EngineCycleTraining()
    data = main_obj.data_loading()
    data = main_obj.data_preprocessing(data)
    train_data, test_data = main_obj.learning_define(data)

    train_data_unit = main_obj.data_queue(train_data)

    # print(data)

if __name__ == "__main__":
    main_flow()


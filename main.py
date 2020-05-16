from py_module.config import Configuration
from py_module.data_reader import DataReader

import os

class EngineCyclePrediction(object):

    def __init__(self):
        self.config_obj = Configuration()
        self.reader_obj = DataReader()

    def data_loading(self):
        file_path = os.path.join(self.config_obj.data_folder, self.config_obj.file_name)
        data = self.reader_obj.read_csv_data(file_path)
        return data



def main_flow():
    
    main_obj = EngineCyclePrediction()
    data = main_obj.data_loading()
    print(data)

if __name__ == "__main__":
    main_flow()


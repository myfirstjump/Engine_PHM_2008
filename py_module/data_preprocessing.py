import pandas as pd

class DataProprocessing(object):

    def __init__(self):
        pass

    def data_preprocessing_2008_PHM_Engine_data(self, data, new_col_name):

        data = self.data_col_rename(data, new_col_name)
        data = data.drop(labels=['sensor_22', 'sensor_23'], axis='columns')
        print(data.describe())

        return data

    def data_col_rename(self, data, new_col_name):

        data.columns = new_col_name

        return data
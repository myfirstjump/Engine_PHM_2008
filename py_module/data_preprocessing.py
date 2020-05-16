import pandas as pd

class DataProprocessing(object):

    def __init__(self):
        pass

    def data_col_rename(self, data, new_col_name):

        data.columns = new_col_name

        return data
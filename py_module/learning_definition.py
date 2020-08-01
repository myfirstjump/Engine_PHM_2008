from py_module.config import Configuration

import pandas as pd
import numpy as np
import random
from sklearn import model_selection

class LearningDefinition(object):

    def __init__(self):
        self.config_obj = Configuration()

    def learning_define_2008_PHM_Engine_data(self, data):

        engine_num = np.max(data.unit)

        # Build RUL column for each unit
        RUL_ = list()
        for each_unit in range(1, engine_num+1):

            sub_data = data[data.unit == each_unit]
            cycles = np.max(sub_data.cycles)
            print('Unit:', each_unit, 'has cycles:', cycles)
            rul = [i for i in range(1, cycles+1)]
            rul.reverse()
            RUL_ = RUL_ + rul
        data['RUL'] = RUL_

        return data

    def train_test_split_2008_PHM_Engine_data(self, data):

        engine_num = np.max(data.unit)

        units = [i for i in range(1, engine_num+1)]

        train_units, test_units = model_selection.train_test_split(units, train_size=0.8)
        # print('Train units:', train_units)
        # print('Test units:', test_units)

        print('Data shape:', data.shape)

        train_data = pd.DataFrame(columns=data.columns)
        test_data = pd.DataFrame(columns=data.columns)
        for each_unit in train_units:
            train_data = pd.concat([train_data, data[data.unit==each_unit]])
        for each_unit in test_units:
            test_data = pd.concat([test_data, data[data.unit==each_unit]])

        print('Train shape:', train_data.shape)
        print('Test shape:', test_data.shape)

        return train_data, test_data

    def build_pre_timestep_supervised


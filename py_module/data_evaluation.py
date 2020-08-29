import random

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras

from py_module.config import Configuration
from py_module.plot_module import PlotDesign
from py_module.learning_definition import LearningDefinition

class DataEvaluation(object):

    def __init__(self):
        self.config_obj = Configuration()
        self.plotting_obj = PlotDesign()
        self.learing_def_obj = LearningDefinition()

    def data_evaluation_2008_PHM_Engine_data(self, data):


        h5_path = self.config_obj.keras_model_path
        model = keras.models.load_model(h5_path)

        def yield_unit_data(data, train_valid_units, epochs):
            cnt = 0
            while cnt < epochs:
                which_unit = random.choice(train_valid_units)
                unit_data = data[data['unit'] == which_unit]
                cnt += 1
                yield which_unit, unit_data
        test_unit_num, test_data = [(test_unit_num, test_data) for (test_unit_num, test_data) in yield_unit_data(data, [i+1 for i in range(self.config_obj.test_engine_number)], 1)][0]

        test_data = self.learing_def_obj.learning_define_2008_PHM_Engine_data(test_data)
        print("以引擎 unit: {} 做為testing data.".format(test_unit_num))

        test_x = test_data.values[:,:-1]
        test_y = test_data.values[:, -1]

        test_x = test_x.reshape((test_x.shape[0], self.config_obj.previous_p_times + 1, self.config_obj.features_num))
        predict_y = model.predict(test_x)
        
        # plotting

        self.plotting_obj.plot_RUL_prediction(pred_y=predict_y, true_y=test_y)
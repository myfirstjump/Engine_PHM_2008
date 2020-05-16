

class Configuration(object):
    
    def __init__(self):

        self.data_folder = "C:\\Users\\edward chen\\Documents\\Data_2008_PHM"
        self.file_name = "train.txt"

        self.features_name = ['unit', 'time_in_cycles', 'op_setting_1', 'op_setting_2', 'op_setting_3',] + ['sensor_' + str(i) for i in range(1, 24)]
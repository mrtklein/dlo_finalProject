import os
import sys


class Utils:

    def __init__(self):
        self.project_dir = os.path.abspath(os.getcwd())
        self.is_windows = sys.platform.startswith('win')

    def getInputPath(self):
        if self.is_windows:
            data_dir = self.project_dir + '\\input\\'
        else:
            data_dir = self.project_dir + '/input/'

        return data_dir

    def getLogPath(self):
        if self.is_windows:
            log_dir = self.project_dir + '\\logs\\'
        else:
            log_dir = self.project_dir + '/logs/'

        return log_dir

# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
import glob
import os
import pandas as pd
import numpy as np
from IPython.display import display
from sklearn import preprocessing
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import LogNorm
import seaborn as sns
from itertools import product
from sklearn.model_selection import GridSearchCV
from joblib import Parallel, delayed
import multiprocessing
import concurrent.futures
import time
from sklearn.feature_selection import RFE
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.colors as mcolors

from SVM_Trainer import SVMX
from DataLoader import load_bigData

'''
G1 = CIC2118
G2 = CIC2105
G3 = CIC2106
G4 = CIC2116
G5 = CIC2117
B1 = CIC2109
B2 = CIC2110
B3 = CIC2134
B4 = CIC2141
'''
def SVM_main():
    bigData = load_bigData()
    pd.set_option('display.width', 4000)
    pd.set_option('display.max_columns', 50)
    pd.set_option('display.max_rows', 100)
    np.set_printoptions(edgeitems=300, linewidth=100000)

    test_goodDevice_length, test_badDevice_length, svc_predict, test_accuracyScore = SVMX(train_goodDevice1=bigData[6],
                                                                                          train_goodDevice2=bigData[16],
                                                                                          train_goodDevice3=bigData[17],
                                                                                          train_goodDevice4=bigData[18],
                                                                                          train_badDevice5=bigData[10],
                                                                                          test_bad_device=bigData[41],
                                                                                          test_good_device=bigData[5],
                                                                                          gamma=0.8, cvalue=1.2e-2,
                                                                                          kernelType="rbf", degree=2,
                                                                                          class_weight="balanced",
                                                                                          scaler=preprocessing.MinMaxScaler())

    print(svc_predict[-test_badDevice_length:])
    print(svc_predict[:test_goodDevice_length])

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    SVM_main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/

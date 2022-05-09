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

def SVMX(train_goodDevice1, train_goodDevice2, train_goodDevice3, train_goodDevice4, train_badDevice5,
               test_bad_device, test_good_device, gamma, cvalue, degree, class_weight, scaler, kernelType):
    train_goodData = pd.concat([train_goodDevice1, train_goodDevice2, train_goodDevice3, train_goodDevice4],
                               ignore_index=True)
    train_goodData_target = pd.DataFrame(np.zeros([len(train_goodData), 1]), columns=["Target"])
    test_minDeviceNumber = min(len(test_bad_device), len(test_good_device))
    test_badDevice_length = len(test_bad_device)
    test_goodDevice_length = len(test_good_device)

    train_badDeviceSelection = train_badDevice5
    train_badDeviceSelection_target = pd.DataFrame(np.ones([len(train_badDeviceSelection), 1]), columns=["Target"])
    train_selectedData = pd.concat([train_goodData, train_badDeviceSelection], ignore_index=True)
    train_selectedData_target = pd.concat([train_goodData_target, train_badDeviceSelection_target],
                                              ignore_index=True)

    test_badDevice_selection = test_bad_device
    test_goodDevice_selection = test_good_device
    test_devices_combined = pd.concat([test_goodDevice_selection, test_badDevice_selection], ignore_index=True)
    test_targetNPL1 = np.zeros([len(test_goodDevice_selection), 1])
    test_targetNPL2 = np.ones([len(test_badDevice_selection), 1])
    test_targetDF1 = pd.DataFrame(test_targetNPL1, columns=['Target'])
    test_targetDF2 = pd.DataFrame(test_targetNPL2, columns=['Target'])
    test_selectedData_target = pd.concat([test_targetDF1, test_targetDF2], ignore_index=True)

    # Grid Search
    svc = make_pipeline(scaler, SVC(kernel=kernelType, cache_size=1950, class_weight=class_weight, gamma=gamma,
                                    C=cvalue, degree=degree))
    # svc = SVC(kernel="poly",cache_size=1900, class_weight='balanced', gamma=gamma, C=cvalue)
    svc.fit((train_selectedData), np.ravel(train_selectedData_target))
    svc_predict = svc.predict(test_devices_combined)
    test_accuracyScore = accuracy_score(y_true=test_selectedData_target, y_pred=svc_predict)

    return test_goodDevice_length, test_badDevice_length, svc_predict, test_accuracyScore


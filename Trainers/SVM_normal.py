import glob
import os
import pandas as pd
import numpy as np
from IPython.display import display
from sklearn import preprocessing
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
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
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import *
from multiprocessing import Pool
from itertools import product
from joblib import parallel_backend





def SVM_normal(train_goodDevice1, train_goodDevice2, train_goodDevice3, train_goodDevice4, train_badDevice5,
                test_bad_device, test_good_device, gamma, cvalue, degree, class_weight, scaler, kernelType):
    train_goodData = pd.concat([train_goodDevice1,train_goodDevice2,train_goodDevice3,train_goodDevice4], ignore_index=True)
    train_goodData_target = pd.DataFrame(np.zeros([len(train_goodData),1]), columns=["Target"])
    test_minDeviceNumber = min(len(test_bad_device), len(test_good_device))
    test_badDevice_length = len(test_bad_device)

    accuracyMatrix = 2 * np.ones([len(train_badDevice5), test_minDeviceNumber])

    for train_index in (np.arange(len(train_badDevice5) - 1, -1, -1)):
        train_badDeviceSelection = train_badDevice5[len(train_badDevice5) - train_index - 1:]
        train_badDeviceSelection_target =  pd.DataFrame(np.ones([len(train_badDeviceSelection),1]), columns=["Target"])
        train_selectedData = pd.concat([train_goodData, train_badDeviceSelection],ignore_index=True)
        train_selectedData_target = pd.concat([train_goodData_target, train_badDeviceSelection_target],ignore_index=True)
        for test_index in np.arange(0, test_minDeviceNumber, 1):
            test_badDevice_selection = test_bad_device[test_badDevice_length - test_index - 1:]
            test_goodDevice_selection = test_good_device[:test_index + 1]
            test_devices_combined = pd.concat([test_goodDevice_selection,test_badDevice_selection], ignore_index=True)
            test_targetNPL1 = np.zeros([len(test_goodDevice_selection), 1])
            test_targetNPL2 = np.ones([len(test_badDevice_selection), 1])
            test_targetDF1 = pd.DataFrame(test_targetNPL1, columns=['Target'])
            test_targetDF2 = pd.DataFrame(test_targetNPL2, columns=['Target'])
            test_selectedData_target = pd.concat([test_targetDF1, test_targetDF2], ignore_index=True)

            # Grid Search
            svc = make_pipeline(scaler, SVC(kernel=kernelType,cache_size=1950, class_weight=class_weight, gamma=gamma, C=cvalue, degree=degree))
            #svc = SVC(kernel="poly",cache_size=1900, class_weight='balanced', gamma=gamma, C=cvalue)
            svc.fit((train_selectedData), np.ravel(train_selectedData_target))
            svc_predict = svc.predict(test_devices_combined)
            test_accuracyScore = accuracy_score(y_true=test_selectedData_target, y_pred=svc_predict)

            accuracyMatrix[len(train_badDevice5) - 1 - train_index, test_index] = test_accuracyScore

    return accuracyMatrix


def SVM_normal_pre(train_goodDevice1, train_goodDevice2, train_goodDevice3, train_goodDevice4, train_badDevice5,
               test_bad_device, test_good_device, gamma, cvalue, degree, class_weight, scaler, kernelType):
    train_goodData = pd.concat([train_goodDevice1, train_goodDevice2, train_goodDevice3, train_goodDevice4],
                               ignore_index=True)
    train_goodData_target = pd.DataFrame(np.zeros([len(train_goodData), 1]), columns=["Target"])
    test_minDeviceNumber = min(len(test_bad_device), len(test_good_device))
    test_badDevice_length = len(test_bad_device)

    accuracyMatrix = 2 * np.ones([len(train_badDevice5), test_minDeviceNumber])
    predictionMatrix = np.ndarray([len(train_badDevice5), test_minDeviceNumber], dtype=object)


    for train_index in (np.arange(len(train_badDevice5) - 1, -1, -1)):
        train_badDeviceSelection = train_badDevice5[len(train_badDevice5) - train_index - 1:]
        train_badDeviceSelection_target = pd.DataFrame(np.ones([len(train_badDeviceSelection), 1]), columns=["Target"])
        train_selectedData = pd.concat([train_goodData, train_badDeviceSelection], ignore_index=True)
        train_selectedData_target = pd.concat([train_goodData_target, train_badDeviceSelection_target],
                                              ignore_index=True)
        for test_index in np.arange(0, test_minDeviceNumber, 1):
            test_badDevice_selection = test_bad_device[test_badDevice_length - test_index - 1:]
            test_goodDevice_selection = test_good_device[:test_index + 1]
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

            accuracyMatrix[len(train_badDevice5) - 1 - train_index, test_index] = test_accuracyScore
            predictionMatrix[len(train_badDevice5) - 1 - train_index, test_index] = svc_predict


    return predictionMatrix, accuracyMatrix






def SVM_normal_pre_full(train_goodDevice1, train_goodDevice2, train_goodDevice3, train_goodDevice4, train_badDevice5,
               test_bad_device, test_good_device, gamma, cvalue, degree, class_weight, scaler, kernelType):
    train_goodData = pd.concat([train_goodDevice1, train_goodDevice2, train_goodDevice3, train_goodDevice4],
                               ignore_index=True)
    train_goodData_target = pd.DataFrame(np.zeros([len(train_goodData), 1]), columns=["Target"])
    test_minDeviceNumber = min(len(test_bad_device), len(test_good_device))
    test_badDevice_length = len(test_bad_device)

    accuracyMatrix = 2 * np.ones([len(train_badDevice5), len(test_bad_device)])
    predictionMatrix = np.ndarray([len(train_badDevice5), len(test_bad_device)], dtype=object)
    print("----------------------------------------------------------------------------------")
    print("Good length: " + str(len(test_good_device))+"Bad Length: "+str(test_badDevice_length))
    glength = len(test_good_device)
    blength = test_badDevice_length

    for train_index in (np.arange(len(train_badDevice5) - 1, -1, -1)):
        train_badDeviceSelection = train_badDevice5[len(train_badDevice5) - train_index - 1:]
        train_badDeviceSelection_target = pd.DataFrame(np.ones([len(train_badDeviceSelection), 1]), columns=["Target"])
        train_selectedData = pd.concat([train_goodData, train_badDeviceSelection], ignore_index=True)
        train_selectedData_target = pd.concat([train_goodData_target, train_badDeviceSelection_target],
                                              ignore_index=True)

        for test_index in np.arange(0, test_badDevice_length, 1):
            test_badDevice_selection = test_bad_device[test_badDevice_length - test_index - 1:]
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

            accuracyMatrix[len(train_badDevice5) - 1 - train_index, test_index] = test_accuracyScore
            predictionMatrix[len(train_badDevice5) - 1 - train_index, test_index] = svc_predict


    return glength, blength, predictionMatrix, accuracyMatrix




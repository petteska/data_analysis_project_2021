# This file contains the functionality for feature extraction from the data. 
# Each feature function is specified, and functions for extracting features from data objects are also described.

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy
from scipy.signal import savgol_filter

import sys

sys.path.append('./utils')
# My files
import utils

#==============================
#       Features        
#==============================


def mean(signal):
    """ returns the mean of an array
    args:
        data: np-array

    returns:
        float
    """
    return np.mean(signal)

def std(signal):
    """ returns the std. deviation of an array
    args:
        data: np-array

    returns:
        float
    """
    return np.std(signal)

def median(signal):
    """ returns the median of an array
    args:
        data: np-array

    returns:
        float
    """
    return np.median(signal)


def rms(signal):
    """ returns the root mean square of an array
    args:
        data: np-array

    returns:
        float
    """
    N = len(signal)
    square = 0

    for i in signal:
        square += i**2
    
    return np.float(np.sqrt(square/N))

def sem(signal):
    """ returns the standard error of the mean value of an array
    args:
        data: np-array

    returns:
        float
    """
    N = len(signal)

    return std(signal)/np.sqrt(N)

def fracjumps(signal):
    """ Find the precentage of instances where consecutive measurements differ with more than 50 ms.
    Arg:
        signal  -   (np.array)
    Return:
        float - between 0 and 1
    """
    jumps =[]
    for i in range(len(signal)-1):
        if abs(signal[i + 1] - signal[i]) >= 0.050: # 50 ms was taken from Rebeccas thesis
            jumps.append(True)
        else:
            jumps.append(False)
    
    if len(jumps) == 0:
        return 0
    
    return sum(jumps)/len(jumps) 


def max(signal):

    return np.max(signal)

def min(signal):
    return np.min(signal)

def slope(signal, data_type):
    sampling_freq = utils.get_sampling_freq(data_type)
    signal_length = utils.get_signal_length(signal, sampling_freq)

    time = np.arange(start= 0, stop= signal_length, step= 1/sampling_freq)

    # print(f"SLOPE:\n Sampling freq: {sampling_freq}\n signal_length: {len(signal)}\n time_length: {len(time)}")
    trend_coef = np.polyfit(time, signal, deg=1)

    return trend_coef[0]


def kurtosis(signal):
    return scipy.stats.kurtosis(signal)[0]
    
def skewness(signal):
    return scipy.stats.skew(signal)[0]


def fd_rms(signal, data_type):
    """ returns the root mean square of the first derivative of an array
    args:
        data: np-array

    returns:
        float
    """
    sampling_freq = utils.get_sampling_freq(data_type)
    time_step = 1/sampling_freq
    der = savgol_filter(signal, window_length=5, polyorder=3, deriv=1, delta=time_step, mode="nearest")
    
    return rms(der)

def fd_mean(signal, data_type):
    """ returns the root mean square of the first derivative of an array
    args:
        data: np-array

    returns:
        float
    """
    sampling_freq = utils.get_sampling_freq(data_type)
    time_step = 1/sampling_freq
    der = savgol_filter(signal, window_length=5, polyorder=3, deriv=1, delta=time_step, mode="nearest")
    
    return mean(der)

def sd_rms(signal, data_type):
    """ returns the root mean square of the first derivative of an array
    args:
        data: np-array

    returns:
        float
    """
    sampling_freq = utils.get_sampling_freq(data_type)
    time_step = 1/sampling_freq
    der = savgol_filter(signal, window_length=5, polyorder=3, deriv=2, delta=time_step, mode="nearest")

    return rms(der)


def sd_mean(signal, data_type):
    """ returns the root mean square of the first derivative of an array
    args:
        data: np-array

    returns:
        float
    """
    sampling_freq = utils.get_sampling_freq(data_type)
    time_step = 1/sampling_freq
    der = savgol_filter(signal, window_length=5, polyorder=3, deriv=2, delta=time_step, mode="nearest")

    return mean(der)

def mode(signal):
    """ returns the mode of an array
    args:
        data: np-array

    returns:
        float
    """
    [val, count] = scipy.stats.mode(signal)
    return val

#==============================
#       Functionality
#==============================

def get_feature_list(data_type):
    """ Defines the list of features that are to be extracted from each data set. 
    args:
        data_type:  (string)    -   "ACC", "BVP", "EDA", "HR", "IBI" or "TEMP"
    returns:
        list of strings -   List of the features to be extracted. The link between the strings and the corresponding functions is defined in `extract_features_e4()`
    """
    if data_type.upper() == "ACC":
        return []
    elif data_type.upper() == "BVP":
        return ["MEAN", "STD", "MEDIAN"]
    elif data_type.upper() == "EDA":
        return ["MEAN", "STD", "SLOPE", "MAX", "MIN", "KURTOSIS", "SKEWNESS", "FD_RMS", "FD_MEAN", "SD_RMS", "SD_MEAN"]
    elif data_type.upper() == "HR":
        return ["MEAN", "STD"]
    elif data_type.upper() == "IBI":
        return ["MEAN", "STD", "MEDIAN", "SEM", "FRACJUMPS", "RMS"]
    elif data_type.upper() == "TEMP":
        return ["MEAN", "STD", "MIN", "MAX", "MODE"]
    else:
        return np.NaN


def extract_features_e4(e4_data_object):
    """ Extracts features from the different measurements of one single experiment and returns a struct as described.
        The input should be preprocessed using the function preprocess_e4() from file preprocessing.py
    args:
        e4_data_object: dict  -   See file organize_data function get_e4_data_object().
        features_to_leave_out: dict - 

    returns:
        struct  -   
    """
    print("Extracting features")
    e4_feature_object = {}
    for data_type, emotion_list in e4_data_object.items():
        # print(f"Data type: {data_type}")
        feature_list = get_feature_list(data_type)
        e4_feature_object[data_type] = {}
        for emotion, data in emotion_list.items():
            if emotion == "header":
                continue
            # print(f"Emotion: {emotion}")
            e4_feature_object[data_type][emotion] = {}
            for feature in feature_list:
                # print(f"Feature: {feature}")
                if feature == "MEAN":
                    feature_data = mean(data)
                elif feature == "STD":
                    feature_data = std(data)
                elif feature == "MIN":
                    feature_data = min(data)
                elif feature == "MAX":
                    feature_data = max(data)
                elif feature == "MEDIAN":
                    feature_data = median(data)
                elif feature == "MODE":
                    feature_data == mode(data)
                elif feature == "SEM":
                    feature_data = sem(data)
                elif feature == "FRACJUMPS":
                    feature_data = fracjumps(data)
                elif feature == "RMS":
                    feature_data = rms(data)
                elif feature == "SLOPE":
                    feature_data == slope(data, data_type)
                elif feature == "KURTOSIS":
                    feature_data = kurtosis(data)
                elif feature == "SKEWNESS":
                    feature_data = skewness(data)
                elif feature == "FD_RMS":
                    feature_data = fd_rms(data, data_type)
                elif feature == "FD_MEAN":
                    feature_data = fd_mean(data, data_type)
                elif feature == "SD_RMS":
                    feature_data = sd_rms(data, data_type)
                elif feature == "SD_MEAN":
                    feature_data = sd_mean(data, data_type)
                else:
                    print(f"Feature {feature} is not yet defined!")

                e4_feature_object[data_type][emotion][feature] = feature_data
    print("Finished extracting features")
    return e4_feature_object

def get_e4_feature_list_header(e4_feature_object):
    header = []
    for data_type, emotions in e4_feature_object.items():
        # print(f"fata type: {data_type}")
        features = emotions[list(emotions.keys())[0]]
        # print(f"emotions.keys(): {emotions.keys()}")
        # print(f"features.keys(): {features.keys()}")
        for feature, feature_data in features.items():
            # print(f"feature: {feature}")
            header.append(str(data_type) + "_" + str(feature))
    return header

def get_e4_feature_list(e4_feature_objects, include_baseline = False):
    # feature_list = pd.dataframe
    feature_lists = {}
    feature_headers = get_e4_feature_list_header(e4_feature_objects[0])
    
    for emotion in utils.get_list_emotions():
        if emotion == "baseline" and not include_baseline:
            continue

        feature_data = {}
        for feature_header in feature_headers:
            # print(feature_header)
            [data_type, feature] = feature_header.split("_", 1)
            feature_data[feature_header] = []
            for feature_object in e4_feature_objects:
                feature_data[feature_header].append(feature_object[data_type][emotion][feature])
            feature_data[feature_header] = np.array(feature_data[feature_header])
        feature_lists[emotion] = feature_data
        # test = pd.DataFrame(data=feature_data)
        # print(feature_data.keys())
    return feature_lists

# def remove_baseline(e4_feature_object):
#     """
#         args:
#             e4_feature_object   :   (struct)    -   Object returned from function "extract_features_e4()"

#         returns:
#             struct  -   Same format as e4_feature_object, but with the baseline results removed from each feature. 
#     """
#     # e4_feature_object[data_type][emotion][feature] = feature_data
#     features_wo_baseline = {}
#     for data_type, emotion_list in e4_feature_object.items():
#         features_wo_baseline[data_type] = {}
#         for emotion, feature_list in emotion_list.items():
#             if emotion == "baseline":
#                 continue
#             features_wo_baseline[data_type][emotion] = {}
#             for feature, feature_data in feature_list.items():
#                 features_wo_baseline[data_type][emotion][feature] = feature_data - e4_feature_object[data_type]["baseline"][feature]
#     return features_wo_baseline



#==============================
#           Test
#==============================

# def test():
#     dt = 0.01
#     time = np.arange(0,10,dt)

#     signal = 2.3*time + np.sin(1*np.pi*time)

#     fd = savgol_filter(signal, window_length=21, polyorder=3, deriv=1, delta=dt)

#     print(fd_rms(signal))


#     plt.figure()
#     plt.plot(time, signal)
#     plt.plot(time, fd)
#     plt.show()


# test()
import numpy as np
from scipy.signal import  medfilt
import copy

import sys

sys.path.append('./utils')
# My files
import utils

#=========================
#       Utility
#=========================

def get_min_max(e4_data_object):
    """Find minimum and maximum values of the signals of the different datatypes.
    
    Args:
        e4_data_object: (dictonary)     -   defined in organiize_data.py

    Returns:
        (dict)
            {
                "ACC"   :   {
                    "min"   :   (float),
                    "max"   :   (float)
                },
                "BVP"   :   {...},
                "EDA"   :   {...},
                "HR"    :   {...},
                "IBI"   :   {...},
                "TEMP"  :   {...}
            }

    """
    min_max = {}
    for data_type, emotion_list in e4_data_object.items():
        min = 9999999
        max = -9999999
        min_max[data_type] = {}
        for emotion, data in emotion_list.items():
            if emotion == "header":
                continue
            if np.min(data) < min:
                min = np.min(data)
            if np.max(data)> max:
                max = np.max(data)
        
        min_max[data_type]["min"] = min
        min_max[data_type]["max"] = max
    
    return min_max


#=====================================
#       Preprocessing steps
#=====================================

def crop_signal(signal, sampling_freq, crop_settings):
    """ Crop signal according to crop_settings
    Args:
        signal:         (np.array)  -   signal to trim
        sampling_freq:  (float)     -   sampling frequency of the signal
        crop_settings:  (list)      -   [crop_length (float), crop_before (bool), crop_after (bool)]
    Return:
        (np.array)  -   trimed signal
    """

    # Crop signal according to crop_settings
    try:
        crop_length = crop_settings[0]
        crop_before = crop_settings[1]
        crop_after = crop_settings[2]
    except:
        print("The crop settings were not entered correctly!")
        return False

    signal_length = utils.get_signal_length(signal, sampling_freq)

    if crop_before and crop_after:
        trim_from = int(np.floor((signal_length - crop_length)/2 * sampling_freq))
        trim_to = int(np.floor(signal_length - (signal_length - crop_length)/2 * sampling_freq))

    elif crop_before:
        trim_from = int(np.floor((signal_length - crop_length) * sampling_freq))
        trim_to = int(signal_length*sampling_freq)
    elif crop_after:
        trim_from = 0
        trim_to = int(np.floor(crop_length*sampling_freq))
    else:
        return False
   
    return signal[trim_from:trim_to]


def normalize_wrt_baseline(signal, baseline):
    """ Normalize the signal with respect to baseline
    Args:
        Signal:     (np.array)
        baseline:   (np.array)
    
    Return:
        normalized signal (np.array)
    """
    return (signal - np.mean(baseline))/np.std(baseline)


def min_max_scale(signal, min, max):
    """ Scale and center signal using min-max scaling.
    Args:
        signal: (np.ndarray)    -   signal to scale
        min:    (float)         -   minimum of entire signal for the experiment
        max:    (float)         -   maximum of entire signal for the experiment
    
    Returns:
        (np.ndarray)    -   scaled signal
    
    """
    return (signal - min)/(max - min)


def filter_signal(signal, signal_type):
    """
    Args:
        filter_settings -   (list)  -   [filter_type]
    """

    s = signal.shape

    if signal_type == "BVP":
        window_length = 351
    else:
        window_length = 3
    filtered_signal = medfilt(np.ravel(signal), window_length)
    return filtered_signal.reshape(s)


def get_preprocess_pipeline(data_type):
    """ Defines how each of the different signals are to be preprocessed
    Args:
        data_type:  (string)    -   data type
    
    Returns:
        (list)  -   list of strings describing the preprocessing pipeline
    """

    if data_type.upper() == "ACC":
        return []

    elif data_type.upper() == "BVP":
        return ["CROP", "FILTER", "RM_BASELINE"]

    elif data_type.upper() == "EDA":
        return ["CROP", "RM_BASELINE"]

    elif data_type.upper() == "HR":
        return ["CROP","RM_BASELINE"]

    elif data_type.upper() == "IBI":
        return []

    elif data_type.upper() == "TEMP":
        return ["CROP", "RM_BASELINE"]

    else:
        return np.NaN


#=====================================
#       Preprocessing
#=====================================

def preprocess_e4(e4_data_object, remove_baseline = True):
    """ Preprocesses the E4 data of a single experiment from the pipeline defined in get_preprocess_pipeline(data_type).
    args:
        e4_data_object: (dict)  -   See file organize_data.py function get_e4_data_object()

    returns:
        (dict)  -   On the same form as the e4_data_object, but with preprocessed data.
            {
                "ACC"   :   {
                    "header"    :   (list), 
                    "baseline"  :   (np.ndarray), 
                    "sad"       :   (np.ndarray), 
                    "relaxed"   :   (np.ndarray),
                    "excited"   :   (np.ndarray),
                    "afraid"    :   (np.ndarray)
                },
                "BVP"   :   {...},
                "EDA"   :   {...},
                "HR"    :   {...},
                "IBI"   :   {...},
                "TEMP"  :   {...}
            }
    """
    print("Preprocessing data")

    # State crop settings
    base_length = 3*60 #seconds
    crop_before = True
    crop_after = False

    crop_settings = [base_length, crop_before, crop_after]

    min_max = get_min_max(e4_data_object)
    
    preprocessed_data_object = {}
    for data_type, emotion_list in e4_data_object.items():
        preprocessed_data_object[data_type] = {}
        baseline_data = emotion_list["baseline"]
        
        for emotion, data in emotion_list.items():
            if emotion == "header":
                continue
            preprocessed_signal = copy.deepcopy(data)
            pipeline = get_preprocess_pipeline(data_type)
            sampling_freq = utils.get_sampling_freq(data_type)
            for i in pipeline:
                if i == "CROP":
                    preprocessed_signal = crop_signal(signal=preprocessed_signal, sampling_freq=sampling_freq, crop_settings=crop_settings)
                
                if i == "FILTER":
                    preprocessed_signal = filter_signal(signal = preprocessed_signal, signal_type= data_type)
                
                if remove_baseline and i == "RM_BASELINE":
                    preprocessed_signal = normalize_wrt_baseline(data, baseline_data)
                else:
                    min = min_max[data_type]["min"]
                    max = min_max[data_type]["max"]
                    if(max <= min):
                        print("ERROR!!!")
                    preprocessed_signal = min_max_scale(data, min, max)
            preprocessed_data_object[data_type][emotion] = preprocessed_signal
            
    return preprocessed_data_object
import json
import datetime
import pandas as pd
import pathlib
from os import name

BASE_PATH_DATA = pathlib.Path("./data/")

def get_signal_length(signal, sampling_freq):
    N = signal.shape[0]
    return N/sampling_freq


def get_list_of_experiments():
    experiments = []
    for exp_dir_path in BASE_PATH_DATA.iterdir():
        experiments.append(exp_dir_path.name)
    return experiments

def experiment_name_to_path(name):
    for exp_dir_path in BASE_PATH_DATA.iterdir():
        if exp_dir_path.name == name:
            return exp_dir_path
    
    return None

def get_e4_data_path(experiment_path):
    for elt in experiment_path.iterdir():
        if elt.is_dir() and elt.name.startswith("Empatica_E4"):
            return elt
    return None

def get_e4_formated_data_path(experiment_path):
    for elt in experiment_path.iterdir():
        if elt.is_dir() and elt.name.startswith("formated_Empatica_E4"):
            return elt
    return None

def get_eeg_data_path(experiment_path):
    for elt in experiment_path.iterdir():
        if elt.is_dir() and elt.name[0:14] == "OpenBCISession":
            return elt
    return None

def get_meta_data_path(experiment_path):
    for elt in experiment_path.iterdir():
        if elt.is_file() and elt.suffix == ".txt" and elt.name[0:19] == "Experiment_metadata":
            return elt
    return None
    

def convert_to_unix_time_stamp(date_string, time_string):
    """convert date and time from string to unix time stamp

    Args:
        date_string (str) :  date on the format "yyyy-mm-dd"
        time_string (str) :  time on the format "hh:mm:ss"

    Returns:
        double: unix time stamp
    """
    time_array = time_string.split(":")
    date_array = date_string.split("-")

    # Split date
    year = int(date_array[0])
    month = int(date_array[1])
    day = int(date_array[2])

    # split time
    hour = int(time_array[0])
    minute = int(time_array[1])
    second = int(time_array[2])
    
    time = datetime.datetime(year, month, day, hour, minute, second)

    return time.timestamp()

def get_sampling_freq(signal_type):
    if signal_type.upper() == "ACC":
        return 32
    elif signal_type.upper() == "BVP":
        return 64
    elif signal_type.upper() == "EDA":
        return 4
    elif signal_type.upper() == "HR":
        return 1
    # elif signal_type.upper() == "IBI":
    #     return 
    elif signal_type.upper() == "TEMP":
        return 4
    elif signal_type.upper() == "EEG":
        return 250
    else:
        return 0

def get_list_datatypes():
    return["ACC", "BVP", "EDA", "HR", "IBI", "TEMP"]

def get_list_emotions():
    return ["baseline", "afraid", "excited", "relaxed", "sad"]

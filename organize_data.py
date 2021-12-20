# Formating of data for the Empatica E4 and the EEG headgear.
# The most important objects are

# import os
from os import name
import numpy as np
import pathlib
import shutil
import pandas as pd
import json
# import datetime
# import time
import sys

sys.path.append('./utils')
# My files
import utils

BASE_PATH_DATA = pathlib.Path("./data/")
# BASE_PATH_DATA_FORMATED = pathlib.Path("./data_formated/")

# def init_file_structure():
#     BASE_PATH_DATA_FORMATED.mkdir(exist_ok=True)



def get_time_intervals(experiment_name):
    """Gets and returns the time intervals for each of the emotions in an experiment  

    Args:
        meta_data_path (str): The file location of the meta data (.txt) file

    Returns:
        struct: struct on the form 
            {
                "baseline"  :   [start_time end_time],
                "sad"       :   [start_time end_time],
                "relaxed"   :   [start_time end_time],
                "excited"   :   [start_time end_time],
                "afraid"    :   [start_time end_time]
            }
    """
    experiment_path = utils.experiment_name_to_path(experiment_name)
    meta_data_path = utils.get_meta_data_path(experiment_path)
    # print(meta_data_path)
    # print(experiment_name)
    time_intervals = {}
    with open(meta_data_path, 'r') as f:
        meta_data_string = f.readlines()[-1]
        
        meta_data = json.loads(meta_data_string)
        
        date_string = meta_data["start_date"]

        for emotion, data in meta_data["emotion_data"].items():
            start_time_string = data["emotion_activation"]["start_time"]
            end_time_string = data["emotion_activation"]["end_time"]
            
            start_time = utils.convert_to_unix_time_stamp(date_string, start_time_string)
            end_time = utils.convert_to_unix_time_stamp(date_string, end_time_string)

            time_intervals[emotion] = [start_time, end_time]

    return time_intervals


def format_e4_data(experiment_name):
    experiment_base_path = utils.experiment_name_to_path(experiment_name)

    e4_data_path = utils.get_e4_data_path(experiment_base_path)

    # Create new directory for formated data 
    formated_e4_data_path = experiment_base_path / ("formated_" + e4_data_path.name)
    print(formated_e4_data_path)
    formated_e4_data_path.mkdir(exist_ok=True)

    # Copy files to new directory
    for file in e4_data_path.iterdir():
        if file.suffix == ".csv" and not file.name == "tags.csv":
            shutil.copy(file, formated_e4_data_path/file.name)

    for file in formated_e4_data_path.iterdir():
        print(file.name)

        content = pd.read_csv(file, header=None)    

        init_time = content[0][0]
        frequency = content[0][1]
        print(f"Start time: {init_time}")
        # print(f"Frequency: {frequency}")

        # Find the header of the file
        if file.name == "ACC.csv":
            header = ["x","y","z"]

        elif file.name == "BVP.csv":
            header = ["bvp"]

        elif file.name == "EDA.csv":
            header = ["eda"]

        elif file.name == "HR.csv":
            header = ["hr"]   

        elif file.name == "IBI.csv":
            header = ["ibi"]            

        elif file.name == "TEMP.csv":
            header = ["temp"]

        else:
            print("ERROR!: The file had a weird name!")
            return

        

        if file.name == "IBI.csv":
            # IBI file has a different form than the others
            header.insert(0, "timestamp")
            content.columns = header

            # Remove first row (the old header row)
            content = content.iloc[1:,:]

            # Convert timestamps to unix
            # content.insert(0,"timestamp", np.ones(len(content.index)))
            for i, row in content.iterrows():
                time_from_start = row["timestamp"]
                content.at[i,"timestamp"] = init_time + time_from_start

                # break


        else:
            content.columns = header
            # Remove two first rows:
            content = content.iloc[2:,:]
            # Add timestamps        
            content.insert(0,"timestamp", np.zeros(len(content.index)))
            for i, row in content.iterrows():
                content.at[i,"timestamp"] = init_time + i/frequency


        
        content.to_csv(file, index = False)


def is_formated(experiment_name):
    experiment_base_path = utils.experiment_name_to_path(experiment_name)

    for dir in experiment_base_path.iterdir():
        if dir.name.startswith("formated_Empatica_E4"):
            return True
    
    return False



def get_e4_data_object(experiment_name):
    """Sorts and returns the e4 data according to the time_intervals

    Args:
        directory_path (str): The directory location of the e4 data
        time_intervals (struct): object returned from the function get_time_intervals()

    Returns:
        struct: struct with the ordered e4 data on the form 
                {
                    "acc_data" : {
                        "header"    :   (list), 
                        "baseline"  :   (np.ndarray), 
                        "sad"       :   (np.ndarray), 
                        "relaxed"   :   (np.ndarray),
                        "excited"   :   (np.ndarray),
                        "afraid"    :   (np.ndarray)
                    },
                    "bvp_data"  : {...},
                    "gsr_data"  : {...},
                    "hr_data"   : {...},
                    "ibi_data"  : {...},
                    "tmp_data"  : {...}
                }
    """

    experiment_base_path = utils.experiment_name_to_path(experiment_name)

    e4_data_path = utils.get_e4_data_path(experiment_base_path)

    if not is_formated(experiment_name):
        print("The data is being formated")
        format_e4_data(experiment_name)
        # print("You have to format the E4 data before you can retreive the object")
    print("e4_data is now formated")

    e4_formated_data_path = utils.get_e4_formated_data_path(experiment_base_path)

    time_intervals = get_time_intervals(experiment_name)

    e4_data = {}
    for file_path in e4_formated_data_path.iterdir():
        if not file_path.is_file() and file_path.suffix == ".csv":
            print("One of the files was not a csv file.")
            continue
             
        name = file_path.stem
        file = pd.read_csv(file_path)
        header = file.keys()
        
        e4_data[name] = {"header": header}

        data = file.to_numpy()
        for emotion, time_interval in time_intervals.items():
            e4_data[name][emotion] = []

            row = 0
            start_time = time_interval[0]
            end_time = time_interval[1]

            while row < data.shape[0]:
                
                if round(data[row][0]) > end_time:
                    break

                if round(data[row][0]) >= start_time:
                    e4_data[name][emotion].append(data[row][1:]) # Do not include the time-stamps

                row += 1

            e4_data[name][emotion] = np.array(e4_data[name][emotion])
    print("Data was succeccfully extracted")
    return e4_data

def get_eeg_data_object(experiment_name):
    """Sorts and returns the eeg data according to the time_intervals

    Args:
        directory_path (str): The directory location of the e4 data
        time_intervals (struct): object returned from the function get_time_intervals()

    Returns:
        struct: struct with the ordered eeg data on the form 
                {
                    "header"    :   (list), 
                    "baseline"  :   (np.ndarray), 
                    "sad"       :   (np.ndarray), 
                    "relaxed"   :   (np.ndarray),
                    "excited"   :   (np.ndarray),
                    "afraid"    :   (np.ndarray)
                },
    """

    experiment_base_path = utils.experiment_name_to_path(experiment_name)

    eeg_data_path = utils.get_eeg_data_path(experiment_base_path)

    time_intervals = get_time_intervals(experiment_name)

    eeg_data = {}

    wanted_columns = [" Timestamp", " EXG Channel 0", " EXG Channel 1", " EXG Channel 2", " EXG Channel 3", " EXG Channel 4", " EXG Channel 5", " EXG Channel 6", " EXG Channel 7", " Accel Channel 0", " Accel Channel 1", " Accel Channel 2"]
    for file_path in eeg_data_path.iterdir():
        if not file_path.is_file() and file_path.suffix == ".txt":
            print("One of the files was not a txt file.")
            continue
        
        file = pd.read_csv(file_path, skiprows=4)
        file = pd.DataFrame(file)
        file = file[wanted_columns]

        header = file.keys()

        eeg_data["header"] = header

        data = file.to_numpy()
        
        for emotion, time_interval in time_intervals.items():
            eeg_data[emotion] = []

            start_time = time_interval[0]
            end_time = time_interval[1]

            row = 0
            while row < data.shape[0]:
                
                if data[row][0].round() > end_time:
                    break

                if data[row][0].round() >= start_time:
                    eeg_data[emotion].append(data[row][0:])

                row += 1
            
            eeg_data[emotion] = np.array(eeg_data[emotion])
    return eeg_data


def test():
    experiments = utils.get_list_of_experiments()
    print(experiments[0])

    format_e4_data(experiments[0])


# test()


# def main():
#     # Create new directory
#     init_file_structure()
    

#     # Create a copy of the wanted data from each experiment into the new folder
#     for exp_dir in BASE_PATH_DATA.iterdir():
#         if not exp_dir.is_dir():
#             continue
        
#         exp_dir_formated_path = pathlib.Path.joinpath(BASE_PATH_DATA_FORMATED, exp_dir.name)
        
#         exp_dir_formated_path.mkdir(exist_ok=True)
        
#         print("Created folder " + str(exp_dir_formated_path))
        
#         # Get path to e4 and eeg data
#         e4_dir = pathlib.Path()
#         eeg_dir = pathlib.Path()
        
#         time_intervals = {} # Will contain the from and to time for all the emotions

#         for data_dir in exp_dir.iterdir():
#             if not data_dir.is_dir():
#                 if data_dir.is_file() and data_dir.name[0:19] == "Experiment_metadata":
#                     time_intervals = get_time_intervals(data_dir)
#                 continue

#             if data_dir.name[0:11] == "Empatica_E4":
#                 e4_dir = data_dir

#             elif data_dir.name[0:14] == "OpenBCISession":
#                 eeg_dir = data_dir
        

#         if not bool(time_intervals): # No .txt file was found
#             print("ERROR!! There seems to be a .txt file missing in the experiment" + exp_dir.name)
        
#         eeg = get_eeg_data_object(eeg_dir,time_intervals)

#         print(eeg["header"])
        
#         # Sort e4 data
#         # e4_data = {}
#         # for file_path in e4_dir.iterdir():
#         #     if not file_path.is_file() and file_path.suffix == ".csv":
#         #         print("One of the files was not a csv file.")
#         #         continue
            
            
#         #     name = file_path.stem
#         #     file = pd.read_csv(file_path)
#         #     header = file.keys()
#         #     print(header)
#         #     data = file.to_numpy()
#         #     print(data.shape)
#         #     for emotion, time_interval in time_intervals.items():
#         #         e4_data[emotion] = []

#         #         row = 0
#         #         start_time = time_intervals[emotion][0]
#         #         end_time = time_intervals[emotion][1]

#         #         while row < data.shape[0]:
                    
#         #             if data[row][0].round() > end_time:
#         #                 break

#         #             if data[row][0].round() >= start_time:
#         #                 e4_data[emotion].append(data[row][0:])

#         #             row += 1
#         #     print(name)

#         # print("Saving json object to dirpath " + str(exp_dir_formated_path))
#         # save_object_as_json(e4_data, exp_dir_formated_path, "e4_data")
#         # print(e4_data.keys())







# main()


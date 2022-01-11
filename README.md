# data_analysis_project_2021
This repository consists of the code used for the data analysis part of the 5-th year project in the course TTK 4550. In short, the project aims to classify different emotions from data extracted from an experimental protocol. This is explained in further detail in the project thesis and in https://github.com/petteska/Experiment_data_collector.

## Overview
This repository has four main responsibility areas:
1. Extract and organize the sensor data from the different experiments.
2. Analyze the metadata from the experiments
3. Preprocess the sensor data
4. Construct a classification scheme using PCA and SVM

The organization and processing algorithms are defined in the different python files and visualization of the data is done in the jupyter notebook files.

Note! The specifics of each of the functions described in this file can be found in the comments in the code files.

## Files and functions
### Data handling
To be able to extract the data, it must be stored in the folder ´data´ with the following file structure

- `data`
    - `Experiment_{ID}`
        - `Empatica_E4_{ID}`
            - `ACC.csv`
            - `BVP.csv`
            - ...
        - `Experiment_metadata_{ID}.txt`

Here the ´.txt´ file is the file extracted from the Experiment Data Collector, see https://github.com/petteska/Experiment_data_collector.
The ´.csv´ files are the files extracted from the Empatica E4 connect web server.

#### Organization of the data (organize_data.py)
All the code used to organize the sensor data is implemented in the file `organize_data.py`. 
The organization consists of
1. Cleaning up the E4 data files
2. Adding timestamps to the files
3. Dividing the data into the intervals described in the meta data file.

This file has four main functions:
- **`get_time_intervals(experiment_name)`** - Extracts the time intervals for when the different emotions where triggered in the experiment from the metadata file.
- **`format_e4_data(experiment_name)`** - Creates a copy of the e4 data, formates it and saves it in under `data/Experiment_{ID}/formated_Empatica_E4_{ID}`. The formating consists of cleaning up the data table for easier extraction and adding timestamps.
    - `is_formated(experiment_name)` - Checks whether or not the experiment data has been formated.
- **`get_e4_data_object(experiment_name)`** - Returns a struct where the data is divided into the different emotions according to the time-intervals from the meta data file. The format of the struct is described below.
```
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
```
- `get_eeg_data_object(experiment_name)` - Same as the function `get_e4_data_object(experiment_name)` only for the EEG data
```
{
    "header"    :   (list), 
    "baseline"  :   (np.ndarray), 
    "sad"       :   (np.ndarray), 
    "relaxed"   :   (np.ndarray),
    "excited"   :   (np.ndarray),
    "afraid"    :   (np.ndarray)
}
```
### Metadata


### Preprocessing (preprocessing.py)
This file is responsible for all the functionality defining the preprocessing of the data.

The main function in this file is `preprocess_e4(e4_data_object, remove_baseline = True)`, which takes in an e4 data object, as described under organization of the data, and returns a copy of the object with processed data.

The different parts of the preprocessing pipeline is described in the function `get_preprocess_pipeline`.


### Feature extraction (feature_extraction.py)


### Data analysis (data_analysis.py)


## Full pipeline
The file `Data_analysis.ipynb` shows en example of how to perform a simple data analysis scheme with the framework described in this repository. It should be well readable :)


"""
Developed by Sheri Mickelson 
AMS 2019 Short Course 
"""

import numpy as np
import pandas as pd
import xarray as xr
from glob import glob

def extract_csv_data(input_data_path,input_vars,label_vars):
    """
    Extracts csv data from a given set of files. Returns datasets 
    containing the input variables and output variables. 
    
    Args:

    :param input_data_path: string pointing towards the dataset directory
    :param input_vars: 1-D list of variables input to the ML models
    :param label_vars: 1-D list of labels for training the ML models

    returns: Input and label data (examples, 32, 32, number of variables), 
             valid dates and run times (examples) 
    """
    data_files = sorted(glob(input_data_path + "*.nc"))
    
    in_data = []
    out_data = []
    valid_times = []
    run_times = []
    
    for files in data_files:
        data = xr.open_dataset(files)
        in_data.append(np.stack([data[v].values for v in input_vars], axis=-1))
        out_data.append(np.stack([data[v].values for v in label_vars], axis=-1))
        valid_times.append(data["time"].values)
        run_time = pd.Timestamp(files.split("/")[-1].split("_")[1])
        run_times.append([run_time] * in_data[-1].shape[0])
        data.close()
    
    all_in_data = np.vstack(in_data)
    all_out_data = np.vstack(out_data)
    all_valid_times = np.concatenate(valid_times)
    all_run_times = np.concatenate(run_times)
    
    del in_data[:], out_data[:], run_times[:], valid_times[:]
    del in_data, out_data, run_times, valid_times
    
    return all_in_data, all_out_data, all_run_times, all_valid_times

"""
Developed by Amanda Burke
Based off methods by Sheri Mickelson, AMS 2019 
AMS 2020 Short Course 
"""

import numpy as np
import pandas as pd
import xarray as xr
from glob import glob

# Input variables for the extract_csv_data() function
csv_input_variables = ['REFL_COM_mean', 'REFL_COM_max', 'REFL_COM_min', 'REFL_COM_std', 'REFL_COM_percentile_10', 
'REFL_COM_percentile_25', 'REFL_COM_percentile_50', 'REFL_COM_percentile_75', 'REFL_COM_percentile_90', 
'U10_mean', 'U10_max', 'U10_min', 'U10_std', 'U10_percentile_10', 'U10_percentile_25', 'U10_percentile_50', 
'U10_percentile_75', 'U10_percentile_90', 'V10_mean', 'V10_max', 'V10_min', 'V10_std', 'V10_percentile_10', 
'V10_percentile_25', 'V10_percentile_50', 'V10_percentile_75', 'V10_percentile_90', 'T2_mean', 'T2_max', 
'T2_min', 'T2_std', 'T2_percentile_10', 'T2_percentile_25', 'T2_percentile_50', 'T2_percentile_75', 
'T2_percentile_90', 'area', 'eccentricity', 'major_axis_length', 'minor_axis_length', 'orientation']
# Label variable for the extract_csv_data() function
csv_label_variable = ['RVORT1_MAX-future_max'] 

# Input variables for the extract_nc_data() function
nc_input_variables = ["REFL_COM_curr", "U10_curr", "V10_curr"]
# Label variable for the extract_nc_data() function
nc_label_variable = ["RVORT1_MAX_future"]


def extract_csv_data(input_data_path):
    """
    Extracts csv data from a given set of files. Returns datasets 
    containing the predictor and label variables. 
    
    Args:
    input_data_path (str): path to dataset directory
    returns: Predictor, label, and valid date data (# of datafiles,). 
             
    """
    # Find all csv files from given directory
    data_files = sorted(glob(input_data_path + "*.csv"))
    
    in_data = []
    out_data = []
    valid_times = []
     
    for files in data_files:
        # Read in csv data
        data = pd.read_csv(files)
        #Append the predictor and label variables
        in_data.append(data.loc[:,csv_input_variables].values)
        out_data.append(data.loc[:,csv_label_variable].values)
        #Append daily timestamps 
        valid_24_hour_date = data.loc[:,"Valid_Date"].values
        valid_times.append(pd.Timestamp(valid_24_hour_date[0][:10]))
    
    return in_data, out_data, valid_times


def extract_nc_data(input_data_path):
    """
    Extracts netcdf data from a given set of files. Returns datasets 
    containing the input variables and output variables. 
    
    Args:
    input_data_path (str): path to dataset directory
    returns: Predictor and label data (examples, 32, 32, number of variables), 
             valid dates (examples,). 
    """
    # Find all netcdf files from given directory
    data_files = sorted(glob(input_data_path + "*.nc"))
    
    in_data = []
    out_data = []
    valid_times = []
    
    for files in data_files:
        # Read in netcdf data
        data = xr.open_dataset(files)
        #Append the daily predictor and label variables 
        in_data.append(np.stack([data[v].values for v in nc_input_variables], axis=-1))
        out_data.append(np.stack([data[v].values for v in nc_label_variable], axis=-1))
        #Append daily timestamps 
        date = pd.Timestamp(files.split("/")[-1].split("_")[1])
        valid_times.append([date] * in_data[-1].shape[0])
        data.close()
    
    # Concatenate/stack data from lists of arrays to a single array
    all_in_data = np.vstack(in_data)
    all_out_data = np.vstack(out_data)
    all_valid_times = np.concatenate(valid_times)
    
    # Delete lists to save memory
    del in_data[:], out_data[:],valid_times[:]
    del in_data, out_data, valid_times
    
    return all_in_data, all_out_data, all_valid_times
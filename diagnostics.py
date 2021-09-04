
import pandas as pd
import numpy as np
import timeit
import os
import json
import pickle
import subprocess
from flask import Flask, session, jsonify, request

##################Load config.json and get environment variables
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 
test_data_path = os.path.join(config['test_data_path'])
model_path = os.path.join(config['prod_deployment_path'])

##################Function to get model predictions
def model_predictions(model_path, test_data_path):
    #read the deployed model and a test dataset, calculate predictions
    model = pickle.load(open(os.path.join(os.getcwd(), model_path, "trainedmodel.pkl"), "rb"))
    
    testdata_filepath = os.path.join(os.getcwd(), test_data_path, "testdata.csv")
    testdata = pd.read_csv(testdata_filepath)
    
    X_test = testdata[["lastmonth_activity", "lastyear_activity", "number_of_employees"]].values.reshape(-1,3)
    
    y_pred = list(model.predict(X_test))

    return y_pred

##################Function to get summary statistics
def dataframe_summary():
    #calculate summary statistics here
    numeric_columns = ["lastmonth_activity", "lastyear_activity", "number_of_employees", "exited"]
    
    data = pd.read_csv(os.path.join(os.getcwd(), dataset_csv_path, "finaldata.csv"))
    
    all_stat = []
    for col in numeric_columns:
        stat_value = []
        stat_value.append(np.mean(data[col]))
        stat_value.append(np.median(data[col]))
        stat_value.append(np.std(data[col]))
        
        all_stat.append(stat_value)

    return all_stat
    
##################Function to count missing values
def missing_values():
    #calculate percentage of the missing values by columns
    data = pd.read_csv(os.path.join(os.getcwd(), dataset_csv_path, "finaldata.csv"))
    pct_missing = list(data.isna().sum(axis=1)/data.shape[0])

    return pct_missing
    


##################Function to get timings
def execution_time():
    #calculate timing of training.py and ingestion.py
    python_files = ["ingestion.py", "training.py"]
    timings = []
    for comd in python_files:
        starttime = timeit.default_timer()
        response = subprocess.run(["python", comd])
        timing = timeit.default_timer() - starttime
        
        timings.append(timing)

    return timings

##################Function to check dependencies
def outdated_packages_list():
    #get a list of
    outdated = subprocess.check_output(['pip', 'list','--outdated'])

    with open(os.path.join(os.getcwd(), "terminal_output.txt"), "wb") as f:
        f.write(outdated)
    
    with open(os.path.join(os.getcwd(), "terminal_output.txt"), "rb") as f:
        packages = f.read()
    
    packages_list = packages.decode("utf-8").split("\n")
    
    columns = packages_list[0].split(" ")
    columns = [col for col in columns if col != ""][:3]
    output_df = pd.DataFrame(columns = columns)
    for i in range(2,len(packages_list)):
        values = packages_list[i].split(" ")
        values = [value for value in values if value != ""][:3]
        if len(values) == 3:
            value_series = pd.Series(values, index = output_df.columns)
            output_df = output_df.append(value_series, ignore_index=True)
    
    return output_df


if __name__ == '__main__':
    model_predictions(model_path, test_data_path)
    dataframe_summary()
    missing_values()
    execution_time()
    outdated_packages_list()





    

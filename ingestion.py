import pandas as pd
import numpy as np
import os
import json
from datetime import datetime




#############Load config.json and get input and output paths
with open('config.json','r') as f:
    config = json.load(f) 

input_folder_path = config['input_folder_path']
output_folder_path = config['output_folder_path']



#############Function for data ingestion
def merge_multiple_dataframe():
    #check for datasets, compile them together, and write to an output file
    dir = os.path.join(os.getcwd(), input_folder_path)

    filenames = os.listdir(dir)
    
    final_df = pd.DataFrame(columns = ["corporation", "lastmonth_activity",
                                    "lastyear_activity", "number_of_employees",
                                    "exited"])
    for filename in filenames:
        temp_df = pd.read_csv(os.path.join(dir,filename))
        temp_df.drop_duplicates(inplace=True)
        final_df =final_df.append(temp_df, ignore_index=True)
       
    
    output_filepath = os.path.join(os.getcwd(), output_folder_path)
    if not os.path.exists(output_filepath):
        os.makedirs(output_filepath)
    
    with open(os.path.join(output_folder_path,"ingestedfiles.txt"), "w") as f:
        for filename in filenames:
            f.write(filename+'\n')
    output_filepath = os.path.join(output_filepath, "finaldata.csv")
    final_df.to_csv(output_filepath, index=False)

            
if __name__ == '__main__':
    merge_multiple_dataframe()

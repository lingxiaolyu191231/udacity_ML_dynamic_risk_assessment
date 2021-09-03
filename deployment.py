from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
import os
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import json



##################Load config.json and correct path variable
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 
prod_deployment_path = os.path.join(config['prod_deployment_path']) 
model_path = os.path.join(config['output_model_path'])
output_model_path = os.path.join(os.getcwd(), model_path, "trainedmodel.pkl")
model = pickle.load(open(output_model_path, "rb"))

####################function for deployment
def store_model_into_pickle(model):
    #copy the latest pickle file, the latestscore.txt value, and the ingestedfiles.txt file into the deployment directory
    
    prod_path = os.path.join(os.getcwd(), prod_deployment_path)
    if not os.path.exists(prod_path):
        os.makedirs(prod_path)
    
    # copy model pkl file
    pickle.dump(model, open(os.path.join(prod_path, "trainedmodel.pkl"), "wb"))
    
    # copy latestscore.txt
    output_model_score = os.path.join(os.getcwd(), model_path, "latestscore.txt")
    with open(output_model_score, "r") as f:
        scores = f.read()
    
    with open(os.path.join(prod_path, "latestscore.txt"), "w") as f:
        f.write(scores)
    
    # open ingestfiles.txt
    output_ingestdata = os.path.join(os.getcwd(), dataset_csv_path, "ingestedfiles.txt")
    with open(output_ingestdata, "r") as f:
        ingestdata = f.read()
    
    with open(os.path.join(prod_path, "ingestedfiles.txt"), "w") as f:
        f.write(ingestdata)
    

if __name__ == "__main__":
    store_model_into_pickle(model)
    
        
        
        


from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
import json
import os
from scoring import score_model
from diagnostics import dataframe_summary, execution_time, missing_values



######################Set up variables for use in our script
app = Flask(__name__)
app.secret_key = '1652d576-484a-49fd-913a-6879acfa6ba4'

with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 

prediction_model = pickle.load(open(os.path.join(os.getcwd(), config['prod_deployment_path'], "trainedmodel.pkl"), "rb"))


#######################Prediction Endpoint
@app.route("/prediction", methods=['POST','OPTIONS'])
def predict():        
    #call the prediction function you created in Step 3
    inputdatapath = request.args.get("inputdata")
    inputdata = pd.read_csv(inputdatapath)
    X = inputdata[["lastmonth_activity", "lastyear_activity", "number_of_employees"]].values.reshape(-1,3)
    preds = prediction_model.predict(X).tolist()
    return str(preds)

#######################Scoring Endpoint
@app.route("/scoring", methods=['GET','OPTIONS'])
def scoring():
    #check the score of the deployed model
    model_f1_score = score_model()

    return str(model_f1_score)

#######################Summary Statistics Endpoint
@app.route("/summarystats", methods=['GET','OPTIONS'])
def stats():        
    #check means, medians, and modes for each column
    summary_stat = dataframe_summary()

    return str(summary_stat)

#######################Diagnostics Endpoint
@app.route("/diagnostics", methods=['GET','OPTIONS'])
def diagnostics():
    #check timing and percent NA values
    timing = execution_time()
    missing_value = missing_values()
    
    return str((timing, missing_values))

if __name__ == "__main__":    
    app.run(host='0.0.0.0', port=8000, debug=True, threaded=True)

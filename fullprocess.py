import os
import sys
import json
import subprocess
import pandas as pd

from training import train_model
from  scoring import score_model
from deployment import store_model_into_pickle
import diagnostics
from reporting import report_confusion_matrix
from ingestion import merge_multiple_dataframe
from apicalls import get_responses, write_apireturn


with open('config_prod.json', 'r') as f:
    config = json.load(f)

##################Check and read new data
#first, read ingestedfiles.txt
def check_new_data():
    with open(os.path.join(os.getcwd(), config['output_folder_path'], 'ingestedfiles.txt'), 'r') as f:
        ingestedfiles = f.read()

        # load newd data files
    dir = os.path.join(os.getcwd(), config['input_folder_path'])
    filenames = os.listdir(dir)
    ingestedfiles = [ingestedfile.replace('\n','') for ingestedfile in ingestedfiles]
    all_datafiles = list(set(filenames + ingestedfiles))

    #second, determine whether the source data folder has files that aren't listed in ingestedfiles.txt
    have_new_data = False
    if len(all_datafiles) > len(ingestedfiles):
        have_new_data = True
        
    return have_new_data

##################Deciding whether to proceed, part 1
#if new data found, you should proceed. otherwise, do end the process here
    
##################Checking for model drift
#check whether the score from the deployed model is different from the score from the model that uses the newest ingested data

def check_model_drift(new_df):

    latestscore_txt = os.path.join(os.getcwd(), config['prod_deployment_path'], "latestscore.txt")

    with open(latestscore_txt, 'r') as f:
        prod_model_score = f.read()

    prod_model_score = float(prod_model_score)
    new_model_score = score_model(config['production_deployment'], new_df)

    ##################Deciding whether to proceed, part 2
    #if you found model drift, you should proceed. otherwise, do end the process here
    model_drift=False

    if new_model_score < prod_model_score:
        model_drift = True

    return model_drift

##################Upload-new-data
def update_data():

    outdated_finaldata = pd.read_csv(os.path.join(os.getcwd(), config['output_folder_path'], "finaldata.csv"))

    new_df = merge_multiple_dataframe(config['input_folder_path'], config['output_folder_path'])

    new_final_df = outdated_finaldata.append(new_df, ignore_index=True)
    new_final_df.to_csv(os.path.join(os.getcwd(),config['output_folder_path'],"finaldata.csv"), index=False)

    # update ingestedfiles.txt
    with open(os.path.join(os.getcwd(), config['output_folder_path'], 'ingestedfiles.txt'), 'w') as f:
        for filename in all_datafiles:
            f.write(filename+'\n')
    
    return new_final_df, new_df


##################Re-deployment
#if you found evidence for model drift, re-run the deployment.py script
def redeploy_model():
    prod_model_path = os.path.join(os.getcwd(), 'production_deployment','trainedmodel.pkl')

    new_model_path = os.path.join(config['output_model_path'])

    # check whether /models folder exists
    if not os.path.exists(os.path.join(os.getcwd(), new_model_path)):
        os.makedirs(os.path.join(os.getcwd(), new_model_path))
        
    new_model = train_model(new_final_df, new_model_path)

    test_data_path = os.path.join(config['test_data_path'])
    testdata = pd.read_csv(os.path.join(os.getcwd(), test_data_path, "testdata.csv"))
    new_model_score = score_model(new_model_path, testdata)

    store_model_into_pickle(config['production_deployment'], new_model, os.path.join(config['output_folder_path']))


##################Diagnostics and reporting
#run diagnostics.py and reporting.py for the re-deployed model
test_data_path = os.path.join(config['test_data_path'])
model_path = os.path.join(config['prod_deployment_path'])
plot_path = os.path.join(config['output_model_path'])

def diagnostics_reporting():
    
    test_data_path = os.path.join(config['test_data_path'])
    model_path = os.path.join(config['prod_deployment_path'])
    plot_path = os.path.join(config['output_model_path'])
    
    report_confusion_matrix(plot_path, test_data_path, model_path, "confusionmatrix2.png")
    
    URL =  "http://127.0.0.1:8000/"
    get_responses = get_responses(URL)
    write_apireturn("apireturns2.txt", model_path, responses)
    
    

if __name__=="__main__":
    have_new_data = check_new_data()
    if not have_new_data:
        sys.exit()
    
    # if new data found, update data
    new_final_df, new_df = update_data()

    if not check_model_drift(new_df):
        sys.exit()
    
    # if model drift is detected, then redeploy model
    redeploy_model()
    # and the plot confusion matrix and report api
    diagnostics_reporting()
    
    
    




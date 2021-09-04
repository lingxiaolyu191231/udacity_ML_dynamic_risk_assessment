import pickle
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
from diagnostics import model_predictions
from sklearn.metrics import plot_confusion_matrix



###############Load config.json and get path variables
with open('config.json','r') as f:
    config = json.load(f) 

test_data_path = os.path.join(config['test_data_path'])
model_path = os.path.join(config['prod_deployment_path'])
plot_path = os.path.join(config['output_model_path'])
filename = "confusionmatrix.png"

##############Function for reporting
def report_confusion_matrix(plot_path, test_data_path, model_path, filename):
    #calculate a confusion matrix using the test data and the deployed model
    y_preds = model_predictions(model_path, test_data_path)
    
    data = pd.read_csv(os.path.join(os.getcwd(), test_data_path, "testdata.csv"))
    X = data[["lastmonth_activity", "lastyear_activity", "number_of_employees"]].values.reshape(-1,3)
    y = data["exited"].values.reshape(-1,1).ravel()
    
    model = pickle.load(open(os.path.join(os.getcwd(), model_path, "trainedmodel.pkl"), "rb"))
    
    plot_confusion_matrix(model, X, y)
    
    #write the confusion matrix to the workspace
    plt.savefig(os.path.join(os.getcwd(), plot_path, filename))


if __name__ == '__main__':
    report_confusion_matrix(plot_path, test_data_path, model_path, filename)

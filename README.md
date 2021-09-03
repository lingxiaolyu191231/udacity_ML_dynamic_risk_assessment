# udacity_ML_dynamic_risk_assessment
author: Lingxiao Lyu
date: September 3, 2021

This project is the Udacity-Machine-Learning-DevOps-Nano-Master degree project. The goal of this project is to build an automatic training and deploying machine learning models using crontab and cronjob.

## Files
1. ingestion.py - this file is used to ingest multiple dataset under multiple folders and concat all files in one panda dataframe.
2. training.py - this file is used to train the final data with a logistic regression model.
3. scoring.py - this file is used to produce the F1 Score for the trained logistic regression model using testdata.csv under testdata folder.
4. deployment.py - this file is used to record the production model pkl file and production model metrics including model scores, confusion matrix plot.
5. diagnostics.py - this file is used to generate model diagnostics including feature statistic summary, prediction summary, check missing values, and calculating execution time of running ingestion.py and training.py
6. reporting.py - this file is used to report model performance - generating confusion matrix plot
7. app.py and apicalls.py - these two files are used to generate real-time api and get real-time responses.
8. fullprocess.py - this file is used for future model maintaince and monitoring, checking whether new data comes and whether there is model drift

## Instruction
1. Run ingestion.py
2. Run training.py
3. Run scoring.py
4. Run deployment.py
5. Run fullprocess.py

## Note
1. This project is already set up with periodic model re-training using cronjob with a period of every 10 minutes. Once every 10 minutes, the system will detect whether there is new data and if there is whether the model performs well on the new data. If a model drift is detected, redeploying, diagnostics, and reporting will happen accordingly.


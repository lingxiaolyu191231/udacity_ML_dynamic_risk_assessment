import requests
import os
import json

#Specify a URL that resolves to your workspace
URL = "http://127.0.0.1:8000/"
with open('config.json','r') as f:
    config = json.load(f)

#Call each API endpoint and store the responses
response1 = requests.get(URL+'/prediction?inputdata=testdata/testdata.csv').content
response2 = requests.get(URL+'/scoring').content
response3 = requests.get(URL+'/summarystats').content
response4 = requests.get(URL+'/diagnostics').content

#combine all API responses
responses = [response1, response2, response3, response4]

with open(os.path.join(os.getcwd(), config['output_model_path'], "apireturns.txt"), "w") as f:
    f.write(str(responses))




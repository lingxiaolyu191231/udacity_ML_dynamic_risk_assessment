import requests
import os
import json

#Specify a URL that resolves to your workspace
URL = "http://127.0.0.1:8000/"
with open('config.json','r') as f:
    config = json.load(f)

#Call each API endpoint and store the responses
def get_responses(URL):
    response1 = requests.post(URL+'/prediction?inputdata=testdata/testdata.csv').content
    response2 = requests.get(URL+'/scoring').content
    response3 = requests.get(URL+'/summarystats').content
    response4 = requests.get(URL+'/diagnostics').content

    #combine all API responses
    responses = [response1, response2, response3, response4]
    return responses
    
def write_apireturn(filename, output, responses):
    with open(os.path.join(os.getcwd(), output, filename), "w") as f:
        f.write(str(responses))

if __name__ == "__main__":
    output = config['output_model_path']
    responses = get_responses(URL)
    write_apireturn('apireturns.txt', output, responses)




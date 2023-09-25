# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import json
import pickle
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd

from transformers import DataFormatterTransformer, LabelEncoderTransformer 
    
app = FastAPI(debug = True)

class Item(BaseModel):
    features:list

@app.post("/predict")
def predict(item: Item):
    print('starting_prediction...')
    
    print('Getting first row of data')
    print(item.features[0])
    
    sample_request = json.dumps(item.features[0])
    sample_request_json =   json.loads(sample_request)
    sample_request_json_df = pd.json_normalize(sample_request_json)
    
    # Convert the input features to a numpy array
    #input_data = np.array(input_features).reshape(1, -1)
    
    # Make predictions using the pre-trained model
    #'''
    print('loading model')
    with open("models/code_exercise_model.pkl", "rb") as model_file:
        model = pickle.load(model_file)
    
    
    print('making prediction')
    prediction = model.predict(sample_request_json_df)
    
    return {"prediction": prediction.tolist()[0]}

def build_dataset():
    data = [json.loads(x) for x in open("data/MLA_100k_checked_v3_30k.jsonlines")]
    target = lambda x: x.get("condition")
    N = -20
    X_test = data[N:]
    y_test = [target(x) for x in X_test]
    for x in X_test:
        del x["condition"]
    return X_test, y_test

def test_batch():
    X_test,y_test = build_dataset()
    X_test = pd.json_normalize(X_test)
    
    with open("models/code_exercise_model.pkl", "rb") as model_file:
        model = pickle.load(model_file)      
    return model.predict(X_test)
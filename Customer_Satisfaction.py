from flask import Flask
from flask import jsonify, request
app = Flask(__name__)
import json
import os
from io import BytesIO
import numpy as np
import pandas as pd
import joblib
import torch
import transformers as ppb
import warnings
warnings.filterwarnings('ignore')
from keras.preprocessing.sequence import pad_sequences

# For DistilBERT:
print("STEP 1")
model_class, tokenizer_class, pretrained_weights = (ppb.DistilBertModel, ppb.DistilBertTokenizer, 'distilbert-base-uncased')

# Load pretrained model/tokenizer
print("STEP 2")
tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
print("STEP 3")
model1 = model_class.from_pretrained(pretrained_weights)
print("STEP 4")
RFC = joblib.load('Dbert.pkl')
print("Ready for Call")

@app.route('/predict', methods=['POST'])
def dbert():
    data = request.data
    dataDict = json.loads(data)
    text = dataDict['text']
    text = [text]
    pda = pd.DataFrame()
    pda['text'] = text
    tokenized_test = pda['text'].apply((lambda x: tokenizer.encode(x, add_special_tokens=True)))
    max_len = 0
    for i in tokenized_test.values:
        if len(i) > max_len:
            max_len = len(i)

    padded_test = pad_sequences(tokenized_test.values, max_len, padding='post')
    attention_mask = np.where(padded_test != 0, 1, 0)
    input_ids1 = torch.tensor(padded_test)
    attention_mask = torch.tensor(attention_mask)

    with torch.no_grad():
        last_hidden_states = model1(input_ids1.long(), attention_mask=attention_mask)

    features_test = last_hidden_states[0][:, 0, :].numpy()
    predictions = RFC.predict(features_test)

    if predictions == 0 :
        sample = "Neutral"
    elif predictions == 1 :
        sample = "Satisfied"
    else :
        sample = "Unsatisfied/Frustrated"


    return jsonify({
        'status' : 'success',
        'Prediction' : sample
    })

if __name__ == "__main__":
    # port = int(os.environ.get("PORT", 5000))
    # app.run(host='0.0.0.0', port=port)
    app.run()
    #print('comes after')
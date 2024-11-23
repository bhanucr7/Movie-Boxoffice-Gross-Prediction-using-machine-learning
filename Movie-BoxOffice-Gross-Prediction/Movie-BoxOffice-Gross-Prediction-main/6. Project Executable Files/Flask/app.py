import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd
import os

app = Flask (__name__)
filepath="model_movies.pkl"
model=pickle.load(open(r'C:\Users\Brahmam\Downloads\Movie_Gross_predict\model_movies (1).pkl', 'rb')) 
scalar=pickle.load(open(r'C:\Users\Brahmam\Downloads\Movie_Gross_predict\scalar_movies (1).pkl','rb'))

@app.route('/')
def home():
  return render_template('home.html')

@app.route('/predict', methods=['POST', 'GET'])
def predict():
   return render_template('input.html')
@app.route('/submit', methods=['POST','GET'])
def submit():
   input_feature=[float(x) for x in request.form.values() ]
   features_values=[np.array(input_feature)]
   feature_name=['budget','genres', 'popularity', 'runtime', 'vote_average', 'vote_count', 'release_month', 'release_DOW']

   x_df=pd.DataFrame (features_values,columns=feature_name) 
   x=scalar.transform(x_df)

   # predictions using the loaded model file prediction-model.predict(x)
   prediction=model.predict(x)
   print("Prediction is:", prediction) 
   return render_template("output.html",result=prediction[0])
if __name__ =="__main__":
  port=int(os.environ.get('PORT', 5000))
  app.run (debug=False)
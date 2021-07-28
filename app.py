import numpy as np
import pandas as pd
from flask import Flask,request,jsonify,render_template
import pickle

app = Flask(__name__)
model=pickle.load(open('storage.pkl','rb'))


@app.route("/")
def hello_world():
    return "<p>BRUHHHHHHHHHH</p>"

@app.route("/predict",methods=['POST'])
def predict_api():
    request_data = request.get_json()
    loan_amnt = request_data['loan_amnt']
    emp_length = request_data['emp_length']
    annual_inc = request_data['annual_inc']
    delinq_2yrs = request_data['delinq_2yrs']
    inq_last_6mths = request_data['inq_last_6mths']
    mths_since_last_delinq = request_data['mths_since_last_delinq']
    mths_since_last_record = request_data['mths_since_last_record']
    open_acc = request_data['open_acc']
    pub_rec = request_data['pub_rec']
    revol_bal = request_data['revol_bal']
    revol_util = request_data['revol_util']
    total_acc = request_data['total_acc']
    purpose = request_data['purpose']
    debt2income = request_data['debt2income']
    age_on_file = request_data['age_on_file']
    array=np.array([loan_amnt,emp_length,annual_inc,delinq_2yrs,inq_last_6mths,mths_since_last_delinq,mths_since_last_record,open_acc,
                    pub_rec,revol_bal,revol_util,total_acc,purpose,debt2income,age_on_file])
    df = pd.DataFrame([array], 
             columns=['loan_amnt','emp_length','annual_inc','delinq_2yrs','inq_last_6mths','mths_since_last_delinq','mths_since_last_record','open_acc',
                    'pub_rec','revol_bal','revol_util','total_acc','purpose','debt2income','age_on_file'])
    y_pred=model.predict(df)
    if(y_pred==[1]):
        print(y_pred)
        return "Success !"
        
    else:
        print(y_pred)
        return "Failed !"
        
    
    

if __name__=="__main__":
    app.run(debug=True,port=8200)
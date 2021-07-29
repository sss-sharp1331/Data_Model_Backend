import numpy as np
import pandas as pd
from flask import Flask,request,jsonify,render_template
import pickle

app = Flask(__name__)
model=pickle.load(open('save.p','rb'))


@app.route("/")
def hello_world():
    return "<h1>PREDICTION</h1>"

@app.route("/predict",methods=['POST'])
def predict_api():
    BData=pd.read_csv("BureauData.csv")
    request_data = request.get_json()
    SSN=request_data['SSN']
    SSN_IN=int(SSN)
    loan_amnt = request_data['loan_amnt']
    emp_length = request_data['emp_length']
    annual_inc = request_data['annual_inc']
    purpose=0
    res=0
    respD=""
    if(request_data['purpose']=='car' or request_data['purpose']=='credit_card' or request_data['purpose']=='debt_consolidation' or request_data['purpose']=='home_improvement' or request_data['purpose']=='house' or request_data['purpose']=='major_purchase' or request_data['purpose']=='other' or request_data['purpose']=='vacation' or request_data['purpose']=='wedding'):
        purpose=1
    
    RowData=BData.loc[BData['id']==SSN_IN]
    
    if(RowData.size==0):
        respD="INVALID SSN NUMBER"
        resp_data={
        "CRED_SCORE":res,
        "RES":respD
        }
        
        return jsonify(resp_data)
    
    delinq_2yrs =RowData['delinq_2yrs'][1]
    inq_last_6mths = RowData['inq_last_6mths'][1]
    mths_since_last_delinq = RowData['mths_since_last_delinq'][1]
    mths_since_last_record = RowData['mths_since_last_record'][1]
    open_acc = RowData['open_acc'][1]
    pub_rec = RowData['pub_rec'][1]
    revol_bal = RowData['revol_bal'][1]
    revol_util = RowData['revol_util'][1]
    total_acc = RowData['total_acc'][1]
    debt2income = float(emp_length)/float(annual_inc)
    age_on_file = request_data['age_on_file']
    array=np.array([loan_amnt,emp_length,annual_inc,delinq_2yrs,inq_last_6mths,mths_since_last_delinq,mths_since_last_record,open_acc,
                    pub_rec,revol_bal,revol_util,total_acc,purpose,debt2income,age_on_file])    
    print(array)
    y_pred=model.decision_function([array])
    res=int(y_pred[0])
    res=str(res)
    if(y_pred[0]>=2):
        respD="STATUS_ACCEPT"
    else:
        respD="STATUS_REJECT"
    resp_data={
        "CRED_SCORE":res,
        "RES":respD
    }
    return jsonify(resp_data)
        
    
    

if __name__=="__main__":
    app.run(debug=True,port=8200)
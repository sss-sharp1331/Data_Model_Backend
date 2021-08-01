import numpy as np
import pandas as pd
from flask import Flask,request,jsonify,render_template
import pickle

app = Flask(__name__)
model=pickle.load(open('save.p','rb'))
def scale_loan_amnt(x):
    return (x-10000)/9500
def scale_emp_length(x):
    return (x-4)/6
def scale_annual_inc(x):
    return (x-59000)/41437
def scale_delinq_2yrs(x):
    return (x)
def scale_inq_last_6mths(x):
    return (x-1)
def scale_mths_since_last_delinq(x):
    return (x-120)/72
def scale_mnths_since_last_record(x):
    return (x-125.89)/19.08
def scale_open_acc(x):
    return (x-9)/6
def scale_pub_rec(x):
    return (x-0.053)/0.233
def scale_revol_bal(x):
    return (x-8901)/13401.5
def scale_revol_util(x):
    return (x-0.49)/0.46
def scale_total_acc(x):
    return (x-21)/15
def scale_debt2income(x):
    return (x-0.16)/0.1564
def scale_age_on_file(x):
    return (x-278)/93


@app.route("/")
def hello_world():
    return "<h1>PREDICTION</h1>"

@app.route("/predict",methods=['POST'])
def predict_api():
    BData=pd.read_csv("BureauData.csv")
    request_data = request.get_json()
    SSN=request_data['ssn']
    SSN_IN=int(SSN)
    loan_amnt = float(request_data['amount'])
    emp_length = float(request_data['workExp'])
    annual_inc = float(request_data['annualSalary'])
    purpose=0
    res=0
    respD=""
    if(request_data['purpose']=='car' or request_data['purpose']=='credit_card' or request_data['purpose']=='debt_consolidation' or request_data['purpose']=='home_improvement' or request_data['purpose']=='house' or request_data['purpose']=='major_purchase' or request_data['purpose']=='other' or request_data['purpose']=='vacation' or request_data['purpose']=='wedding'):
        purpose=1
    
    RowData=BData.loc[BData['id']==SSN_IN]
    
    if(loan_amnt==0 or loan_amnt<0):
        respD="REJECTED"
        resp_data={
        "CRED_SCORE":res,
        "CRED_APPROVAL_STATUS":respD,
        "resOrig":"Invalid Loan Amount"
        }
        
        return jsonify(resp_data)
    
    if(annual_inc==0 or annual_inc<0):
        respD="REJECTED"
        resp_data={
        "CRED_SCORE":res,
        "CRED_APPROVAL_STATUS":respD,
        "resOrig":"Invalid Annual Income"
        }
        
        return jsonify(resp_data)
    
    if(RowData.size==0):
        respD="REJECTED"
        resp_data={
        "CRED_SCORE":res,
        "CRED_APPROVAL_STATUS":respD,
        "resOrig":"Invalid SSN Number"
        }
        
        return jsonify(resp_data)
    
    if(loan_amnt<10000):
        respD="REJECTED"
        resp_data={
        "CRED_SCORE":res,
        "CRED_APPROVAL_STATUS":respD,
        "resOrig":"Loan Amount less than 10000"
        }
        
        return jsonify(resp_data)
    
    delinq_2yrs =RowData['delinq_2yrs'][RowData.index]
    inq_last_6mths = RowData['inq_last_6mths'][RowData.index]
    mths_since_last_delinq = RowData['mths_since_last_delinq'][RowData.index]
    mths_since_last_record = RowData['mths_since_last_record'][RowData.index]
    open_acc = RowData['open_acc'][RowData.index]
    pub_rec = RowData['pub_rec'][RowData.index]
    revol_bal = RowData['revol_bal'][RowData.index]
    revol_util = RowData['revol_util'][RowData.index]
    total_acc = RowData['total_acc'][RowData.index]
    debt2income = float(emp_length)/float(annual_inc)
    RowData['earliest_cr_line'] = pd.to_datetime(RowData['earliest_cr_line'])
    RowData['age_on_file']=((2021 - RowData['earliest_cr_line'].dt.year)*12 + 7 - RowData['earliest_cr_line'].dt.month)
    age_on_file=RowData['age_on_file'][RowData.index]
    loan_amnt=scale_loan_amnt(float(loan_amnt))
    emp_length=scale_emp_length(float(emp_length))
    annual_inc=scale_annual_inc(float(annual_inc))
    delinq_2yrs=scale_delinq_2yrs(float(delinq_2yrs))
    inq_last_6mths=scale_inq_last_6mths(float(inq_last_6mths))
    mths_since_last_delinq=scale_mths_since_last_delinq(float(mths_since_last_delinq))
    mths_since_last_record=scale_mnths_since_last_record(float(mths_since_last_record))
    open_acc=scale_open_acc(float(open_acc))
    pub_rec=scale_pub_rec(float(pub_rec))
    revol_bal=scale_revol_bal(float(revol_bal))
    revol_util=scale_revol_util(float(revol_util))
    total_acc=scale_total_acc(float(total_acc))
    debt2income=scale_debt2income(float(debt2income))
    age_on_file=scale_age_on_file(float(age_on_file))
    
    array=np.array([loan_amnt,emp_length,annual_inc,delinq_2yrs,inq_last_6mths,mths_since_last_delinq,mths_since_last_record,open_acc,
                    pub_rec,revol_bal,revol_util,total_acc,purpose,debt2income,age_on_file])    
    print(array)
    y_pred=model.decision_function([array])
    res=float(y_pred[0])
    resOrig=""
    if(y_pred[0]>=1.2):
        respD="Approved"
        resOrig="Approved"
        
    else:
        respD="REJECT"
        resOrig="Low Credit Score"
    y_max=12.58
    y_min=-1.21
    
    res=((y_max-res)*350+(res-y_min)*850)/(y_max-y_min)
    res=str(int(res))
    resp_data={
        "CRED_SCORE":res,
        "CRED_APPROVAL_STATUS":respD,
        "resOrig":resOrig
    }
    return jsonify(resp_data)
        
    
    

if __name__=="__main__":
    app.run(debug=True,port=8200)
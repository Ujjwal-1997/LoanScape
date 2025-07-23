from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('home.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        
        in_income = request.form['income']
        in_education = request.form['selectlist']
        in_dependents = request.form['dependents']
        self_emp = request.form['selectlist1']
        in_loan_amt = request.form['loan_amt']
        in_loan_term = request.form['loan_term']
        in_c_score = request.form['c_score']
        in_res_assets = request.form['res_asset']
        in_com_assets = request.form['com_asset']
        in_lux_assets = request.form['lux_asset']
        in_bank_assets = request.form['bank_asset']

        if request.form['button'] == 'Suggest':
            pkl_filename = "encoder.pkl"
            with open(pkl_filename, 'rb') as file:
                encoder = pickle.load(file)

            pkl_filename = "scaler.pkl"
            with open(pkl_filename, 'rb') as file:
                scaler = pickle.load(file)

            cat_col = [in_education,self_emp]
            cat_col_array = np.array(cat_col).reshape(1,-1)
            num_col = [in_dependents,in_income,in_loan_amt,in_loan_term,in_c_score,in_res_assets,in_com_assets,in_lux_assets,in_bank_assets]
            num_col_array = np.array(num_col).reshape(1,-1)
            X_encoded = encoder.transform(cat_col_array)
            X_scaled = scaler.transform(num_col_array)
            X_processed = np.hstack([X_scaled, X_encoded])

            # Load the LR model
            pkl_filename = "loan_pred_model.pkl"
            with open(pkl_filename, 'rb') as file:
                model = pickle.load(file)

            y_pred = model.predict(X_processed)

            op = "There are 95'%' chances of your loan being {}.".format(y_pred)

            if y_pred[0] == 'Rejected':
                avg_dependents = 2.5
                avg_income = 5025903.6
                avg_loan_amt = 15247251.5
                avg_loan_term = 10.4
                avg_c_score = 703.5
                avg_reg_assets_value = 7399811.7
                avg_com_assets_value = 5001355.4
                avg_luxury_assets_value = 15016603.9
                avg_bank_assets_value = 4959525.6

                data = {
                    'Parameters': ['No. of dependents', 'Annual income', 'Loan amount', 'Loan term', 'Cibil score', 'Residential assets', 'Commercial assets', 'Luxury assets', 'Bank assets'],
                    'Average Bank Metrics': [avg_dependents, avg_income, avg_loan_amt, avg_loan_term, avg_c_score, avg_reg_assets_value, avg_com_assets_value, avg_luxury_assets_value, avg_bank_assets_value],
                    'Your Metrics': [in_dependents, in_income, in_loan_amt, in_loan_term, in_c_score, in_res_assets, in_com_assets, in_lux_assets, in_bank_assets]
                }

                df = pd.DataFrame(data)
    
                # Pass the data to the template
                return render_template('predict_rej.html', parameters=df['Parameters'].tolist(), 
                           avg_bank_metrics=df['Average Bank Metrics'].tolist(), 
                           your_metrics=df['Your Metrics'].tolist(),
                           suggested_loan = op, income = in_income, edu = in_education, dep = in_dependents, emp = self_emp, loan_amt = in_loan_amt, loan_term = in_loan_term, c_score = in_c_score, res_asset = in_res_assets, com_asset = in_com_assets, lux_asset = in_lux_assets, bank_asset = in_bank_assets)

            return render_template('predict.html', suggested_loan = op, income = in_income, edu = in_education, dep = in_dependents, emp = self_emp, loan_amt = in_loan_amt, loan_term = in_loan_term, c_score = in_c_score, res_asset = in_res_assets, com_asset = in_com_assets, lux_asset = in_lux_assets, bank_asset = in_bank_assets)
    else:
        return render_template('predict.html')

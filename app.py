from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler

#Load the trained model
model_path = "best_loan_model.pkl"
model = joblib.load(model_path)

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get user input from the form
        Gender = request.form['Gender']
        Married = request.form['Married']
        Dependents = request.form['Dependents']
        Education = request.form['Education']
        Self_Employed = request.form['Self_Employed']
        ApplicantIncome = float(request.form['ApplicantIncome'])
        CoapplicantIncome = float(request.form['CoapplicantIncome'])
        LoanAmount = float(request.form['LoanAmount'])
        Loan_Amount_Term = int(request.form['Loan_Amount_Term'])
        Credit_History = int(request.form['Credit_History'])
        Property_Area = request.form['Property_Area']

        # Compute Total Income
        TotalIncome = ApplicantIncome + CoapplicantIncome

        # Manual Label Encoding
        category_mappings = {
            "Gender": {"Male": 1, "Female": 0},
            "Married": {"No": 0, "Yes": 1},
            "Education": {"Graduate": 0, "Not Graduate": 1},
            "Self_Employed": {"No": 0, "Yes": 1},
            "Property_Area": {"Rural": 0, "Semiurban": 1, "Urban": 2},
            "Dependents": {"0": 0, "1": 1, "2": 2, "3+": 3}
        }

        # Apply encoding
        encoded_data = [
            category_mappings["Gender"][Gender],
            category_mappings["Married"][Married],
            category_mappings["Dependents"][Dependents],
            category_mappings["Education"][Education],
            category_mappings["Self_Employed"][Self_Employed],
            ApplicantIncome,
            CoapplicantIncome,
            LoanAmount,
            Loan_Amount_Term,
            Credit_History,
            category_mappings["Property_Area"][Property_Area],
            TotalIncome
        ]

        # Convert to NumPy array
        input_data = np.array([encoded_data])

        # Standardize numerical values (on-the-fly scaling)
        scaler = StandardScaler()
        numerical_indices = [5, 6, 7, 8, 9, 11]  # Indices of numerical columns
        input_data[:, numerical_indices] = scaler.fit_transform(input_data[:, numerical_indices])

        # Predict on user input
        prediction = model.predict(input_data)[0]
        result = "Approved" if prediction == 1 else "Not Approved"
        
        return render_template('index.html', result=result)

    return render_template('index.html', result=None)

if __name__ == '__main__':
    app.run(debug=True)
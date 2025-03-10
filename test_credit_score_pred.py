import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Load the pre-trained model and scaler
with open('balanced_credit_score_model.pkl', 'rb') as f:
    balanced_model = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('selected_features.pkl', 'rb') as f:
    selected_features = pickle.load(f)



# Define diverse test cases to check for bias
test_cases = [
    # Strong "Good" Credit Score Scenario
    {
        'Month': 11, 'Age': 50, 'Annual_Income': 150000, 'Monthly_Inhand_Salary': 12500, 
        'Num_Bank_Accounts': 6, 'Num_Credit_Card': 8, 'Interest_Rate': 2.5, 
        'Num_of_Loan': 1, 'Delay_from_due_date': 0, 'Num_of_Delayed_Payment': 0, 
        'Changed_Credit_Limit': 7000, 'Num_Credit_Inquiries': 1, 'Outstanding_Debt': 2000, 
        'Credit_Utilization_Ratio': 0.1, 'Credit_History_Age': 25, 
        'Total_EMI_per_month': 800, 'Amount_invested_monthly': 1000, 
        'Monthly_Balance': 5000, 'Credit_Mix_Bad': 0, 'Credit_Mix_Good': 1, 
        'Credit_Mix_Standard': 0, 'Payment_of_Min_Amount_Yes': 1
    },
    
    # Close-to-Good but Still Standard Scenario
    {
        'Month': 8, 'Age': 40, 'Annual_Income': 80000, 'Monthly_Inhand_Salary': 6666, 
        'Num_Bank_Accounts': 4, 'Num_Credit_Card': 5, 'Interest_Rate': 4.0, 
        'Num_of_Loan': 2, 'Delay_from_due_date': 1, 'Num_of_Delayed_Payment': 1, 
        'Changed_Credit_Limit': 2000, 'Num_Credit_Inquiries': 2, 'Outstanding_Debt': 10000, 
        'Credit_Utilization_Ratio': 0.3, 'Credit_History_Age': 15, 
        'Total_EMI_per_month': 500, 'Amount_invested_monthly': 400, 
        'Monthly_Balance': 2000, 'Credit_Mix_Bad': 0, 'Credit_Mix_Good': 0, 
        'Credit_Mix_Standard': 1, 'Payment_of_Min_Amount_Yes': 1
    },
    
    # High-Risk Scenario Bordering on Poor Credit Score
    {
        'Month': 4, 'Age': 30, 'Annual_Income': 25000, 'Monthly_Inhand_Salary': 2000, 
        'Num_Bank_Accounts': 2, 'Num_Credit_Card': 3, 'Interest_Rate': 12.0, 
        'Num_of_Loan': 4, 'Delay_from_due_date': 7, 'Num_of_Delayed_Payment': 5, 
        'Changed_Credit_Limit': -1000, 'Num_Credit_Inquiries': 8, 'Outstanding_Debt': 25000, 
        'Credit_Utilization_Ratio': 0.7, 'Credit_History_Age': 6, 
        'Total_EMI_per_month': 1000, 'Amount_invested_monthly': 100, 
        'Monthly_Balance': 200, 'Credit_Mix_Bad': 1, 'Credit_Mix_Good': 0, 
        'Credit_Mix_Standard': 0, 'Payment_of_Min_Amount_Yes': 0
    },
    
    # Edge Case Between Standard and Poor Credit Score
    {
        'Month': 10, 'Age': 38, 'Annual_Income': 55000, 'Monthly_Inhand_Salary': 4583, 
        'Num_Bank_Accounts': 3, 'Num_Credit_Card': 4, 'Interest_Rate': 7.0, 
        'Num_of_Loan': 3, 'Delay_from_due_date': 5, 'Num_of_Delayed_Payment': 3, 
        'Changed_Credit_Limit': 500, 'Num_Credit_Inquiries': 5, 'Outstanding_Debt': 15000, 
        'Credit_Utilization_Ratio': 0.6, 'Credit_History_Age': 8, 
        'Total_EMI_per_month': 700, 'Amount_invested_monthly': 250, 
        'Monthly_Balance': 800, 'Credit_Mix_Bad': 0, 'Credit_Mix_Good': 0, 
        'Credit_Mix_Standard': 1, 'Payment_of_Min_Amount_Yes': 1
    },
    
    # Scenario with Minimal Risk Indicators
    {
        'Month': 1, 'Age': 22, 'Annual_Income': 15000, 'Monthly_Inhand_Salary': 1250, 
        'Num_Bank_Accounts': 1, 'Num_Credit_Card': 1, 'Interest_Rate': 10.0, 
        'Num_of_Loan': 1, 'Delay_from_due_date': 1, 'Num_of_Delayed_Payment': 1, 
        'Changed_Credit_Limit': 0, 'Num_Credit_Inquiries': 1, 'Outstanding_Debt': 1000, 
        'Credit_Utilization_Ratio': 0.2, 'Credit_History_Age': 1, 
        'Total_EMI_per_month': 200, 'Amount_invested_monthly': 50, 
        'Monthly_Balance': 300, 'Credit_Mix_Bad': 0, 'Credit_Mix_Good': 0, 
        'Credit_Mix_Standard': 1, 'Payment_of_Min_Amount_Yes': 0
    }
]

# Prepare the test cases as a DataFrame
test_df = pd.DataFrame(test_cases)

# Define custom thresholds for classification
thresholds = {
    'Credit_Score_Good': 0.3,    # Lowered threshold
    'Credit_Score_Poor': 0.4,    # Lowered threshold
    'Credit_Score_Standard': 0.5 # Remains the same
}

# Function to predict using custom thresholds
def predict_with_custom_thresholds(probs, thresholds):
    class_labels = ['Credit_Score_Good', 'Credit_Score_Poor', 'Credit_Score_Standard']
    predictions = []
    for prob in probs:
        class_prediction = 'Credit_Score_Standard' # Default class
        for i, threshold in enumerate([thresholds['Credit_Score_Good'], thresholds['Credit_Score_Poor'], thresholds['Credit_Score_Standard']]):
            if prob[i] >= threshold:
                class_prediction = class_labels[i]
                break
        predictions.append(class_prediction)
    return predictions

# Test the new prediction method with test cases
for i, test_input in enumerate(test_cases, 1):
    input_df = pd.DataFrame([test_input])
    pred_probs = balanced_model.predict_proba(input_df)[0]
    pred_label = predict_with_custom_thresholds([pred_probs], thresholds)[0]
    print(f"Test Case {i}:")
    print(f"Input: {test_input}")
    print(f"Predicted Credit Score: {pred_label}")
    print(f"Prediction Probabilities: {pred_probs}")
    print("-" * 50)

# Summarize the predictions
prediction_summary = pd.DataFrame([balanced_model.predict_proba(pd.DataFrame([test_input]))[0] for test_input in test_cases],
                                  columns=['Credit_Score_Good', 'Credit_Score_Poor', 'Credit_Score_Standard'])
print("\nPrediction Summary:")
print(prediction_summary.describe())

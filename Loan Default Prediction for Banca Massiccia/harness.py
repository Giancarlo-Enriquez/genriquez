import argparse
import pandas as pd
import numpy as np
import pickle
import statsmodels.api as sm
from prediction import preprocess_holdout_data, feature_selection, prediction_harness
import joblib

# Takes the input and outout CSV arguments
parser = argparse.ArgumentParser()
parser.add_argument("--input_csv", type=str, help="Path to the input CSV file.")
parser.add_argument("--output_csv", type=str, help="Path to the output CSV file.")
args = parser.parse_args()

# Load the final model
# with open('gb_model_11_final.pkl', 'rb') as file:
with open('GB_model_2007to2012_final.pkl', 'rb') as file:
    model = joblib.load(file)
# # # GRADIENT BOOST # # # 
print(f"\n\n\n{model}\n\n\n\n")

# Process holdout data
print("Loading input data...")
data = preprocess_holdout_data(args.input_csv) # gives the features to be used for the chosen model

# View the pre-processed data
print(data.shape)
print(data.columns)
print('\n')
print(data)

# Select final set of features
X = feature_selection(data, num_features = 11)

# Calibration parameters
logit_calibration_params_11 = np.array([-0.27109075, -4.93992847,  0.27309075])
logit_calibration_params_8 = np.array([-0.29952795, -4.62569469,  0.30152795])
gb_calibration_params_11 = np.array([-0.26850218, -4.05455449,  0.27050218])
print("Running inference...")

# Obtain calibrated predictions on the holdout set
model_type = 'gb'
print('*** GRADIENT BOOST ***')
predictions = prediction_harness(X, model, model_type, calibration_params = gb_calibration_params_11)
preds = pd.Series(predictions)   

# Write to output file
preds.to_csv(args.output_csv, index=False, header = False)
print(f"Predictions saved to {args.output_csv}")
print("\n Harness Complete.")

# IGNORE
"""
if model_type == 'logit':
    X = sm.add_constant(X, has_constant='add')
    print('after adding constant:')
    print(X)

if model_type == 'logit':
    print('*** LOGIT ***')
    predictions = prediction_harness(X, model, model_type, calibration_params = logit_calibration_params_11)
    preds = pd.Series(predictions)

OLD MODEL
[0.02452957 0.0308569  0.03837941 0.11428075 0.00959686 0.02147247
 0.00262312 0.00534979 0.01261059 0.00257285]

NEW MODEL
[0.01869708 0.03317564 0.03838324 0.09271784 0.00931543 0.01978023
 0.00247718 0.00489219 0.01121485 0.00238808]
"""

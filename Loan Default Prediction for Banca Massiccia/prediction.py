import numpy as np
import pickle as pickle
import joblib
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from scipy.stats import boxcox
from scipy.stats import zscore
import scipy.optimize as optimize
from sklearn.preprocessing import PowerTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
from feature_engine.transformation import YeoJohnsonTransformer

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
"""
PRE-PROCESSING THE HOLDOUT SET FUNCTIONS & FEATURE SELECTION
"""
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

# STEP 1: load the harness DF, summary-train DF, and growth-2012 DF

def load_harness_df(holdout_df):

  df_harness = pd.read_csv(holdout_df)
  if 'Unnamed: 0' in df_harness.columns:
    df_harness = df_harness.drop(columns=['Unnamed: 0'])

  # return
  return df_harness


# STEP 2: load the summary-train DF & growth-2012 DF

def load_helper_dfs():

  # read in dfs
  fp_summary_train = 'data/NOV17-FINAL-preprocessed_train_mean_std_percentiles.csv'
  fp_growth_2012 = 'data/unprocessed_2012_growth_data.csv'
  df_summary_train = pd.read_csv(fp_summary_train, index_col=0)
  df_growth_2012 = pd.read_csv(fp_growth_2012)

  # remove unnecessary columns
  if 'Unnamed: 0' in df_summary_train.columns:
    df_summary_train = df_summary_train.drop(columns=['Unnamed: 0'])
  if 'Unnamed: 0' in df_growth_2012.columns:
    df_growth_2012 = df_growth_2012.drop(columns=['Unnamed: 0'])

  # return
  return df_summary_train, df_growth_2012


# STEP 3: calculate growth-over-time features

''' def calculate_one_growth(current, prior, first_year): '''
def calculate_one_growth(current, prior):

  # # # COMMENTING OUT OLD CODE - we suspect first_year was causing leakage
  '''
  # first year: set growth to 0
  if first_year:
    return 0
  '''

  # if prior is NaN: set to NaN
  if pd.isna(prior):
    return float('nan')

  # if prior is 0: return Nan
  elif prior == 0:
    return float('nan')

  # growth calculation
  else:
    return (current - prior) / prior


def calc_growth_harness(harness_df, growth_2012_df):

  df_growth = harness_df.copy()

  # # # OLD CODE: CONTAINS LEAKAGE (oop)
  """
  df_growth_concat = pd.concat([growth_2012_df, df_growth_1], ignore_index=True)
  df_growth = df_growth_concat.sort_values(['id', 'fs_year'])  # # # ERROR #2: wrong order of final output

  # obtain prior year's assets, profit, revenue
  df_growth['prior_year_assets'] = df_growth.groupby('id')['asst_tot'].shift(1)
  df_growth['prior_year_profit'] = df_growth.groupby('id')['profit'].shift(1)
  df_growth['prior_year_revenue'] = df_growth.groupby('id')['rev_operating'].shift(1)

  # identify whether current row is a first year of ID
  df_growth['first_year'] = (df_growth['id'] != df_growth['id'].shift(1)).astype(int)

  # calculate growth features
  df_growth['asset_growth'] = df_growth.apply(lambda x: calculate_one_growth(x['asst_tot'], x['prior_year_assets'], x['first_year']), axis=1)
  df_growth['profit_growth'] = df_growth.apply(lambda x: calculate_one_growth(x['profit'], x['prior_year_profit'], x['first_year']), axis=1)
  df_growth['revenue_growth'] = df_growth.apply(lambda x: calculate_one_growth(x['rev_operating'], x['prior_year_revenue'], x['first_year']), axis=1)

  # drop unnecessary columns
  df_growth.drop(columns=['prior_year_assets', 'prior_year_profit', 'prior_year_revenue', 'first_year'], inplace=True)

  # filter to only have holdout set data
  df_growth = df_growth[df_growth['fs_year'] == 2013]  # # # ERROR #3: empty DataFrame
  """

  df_growth['asset_growth'] = float('nan')
  df_growth['profit_growth'] = float('nan')
  df_growth['revenue_growth'] = float('nan')

  # TO ELIMINATE LEAKAGE, we're iterating row-by-row through the holdout so no influence is occurring
  for idx, row in df_growth.iterrows():
    current_id = row['id']
    current_assets = row['asst_tot']
    current_profit = row['profit']
    current_revenue = row['rev_operating']

    # check if the holdout set's firm is in 2012 data
    firm_2012_data = growth_2012_df[growth_2012_df['id'] == current_id]

    # if it is: calculate growth
    if not firm_2012_data.empty:
      prior_assets = firm_2012_data.iloc[0]['asst_tot']
      prior_profit = firm_2012_data.iloc[0]['profit']
      prior_revenue = firm_2012_data.iloc[0]['rev_operating']

      asset_growth = calculate_one_growth(current_assets, prior_assets)
      profit_growth = calculate_one_growth(current_profit, prior_profit)
      revenue_growth = calculate_one_growth(current_revenue, prior_revenue)

      # put this info into holdout set DF
      df_growth.at[idx, 'asset_growth'] = asset_growth
      df_growth.at[idx, 'profit_growth'] = profit_growth
      df_growth.at[idx, 'revenue_growth'] = revenue_growth

    # else: it's a first year so impute 0
    else:
      df_growth.at[idx, 'asset_growth'] = 0
      df_growth.at[idx, 'profit_growth'] = 0
      df_growth.at[idx, 'revenue_growth'] = 0

  return df_growth


# STEP 4: calculate financial ratios & handle missing values

"""
Function to calculate financial ratios for DF
"""

def calculate_financial_ratios(df):

  # set-up feature set
  feature_set = pd.DataFrame()
  feature_set['fs_year'] = df['fs_year']
  feature_set['id'] = df['id']
  feature_set['stmt_date'] = df['stmt_date']
  feature_set['legal_struct'] = df['legal_struct']
  feature_set['ateco_sector'] = df['ateco_sector']

  # size
  feature_set['total_assets'] = df['asst_tot']
  feature_set['total_equity'] = df['eqty_tot']

  # debt coverage
  feature_set['solvency_debt_ratio'] = (df['debt_st'] + df['debt_lt']) / df['asst_tot']
  feature_set['debt_to_equity_ratio'] = (df['debt_st'] + df['debt_lt']) / df['eqty_tot']
  feature_set['interest_coverage_ratio'] = df['ebitda'] / df['exp_financing']
  feature_set['debt_service_coverage_ratio'] = df['ebitda'] / (df['debt_st'] + df['debt_lt'])

  # leverage
  feature_set['leverage_ratio'] = df['asst_tot'] / df['eqty_tot']
  feature_set['lt_debt_to_capitalization_ratio'] = df['debt_lt'] / (df['debt_lt'] + df['eqty_tot'])

  # profitability
  feature_set['profit_margin_ratio'] = df['profit'] / df['rev_operating']
  feature_set['return_on_assets'] = df['roa']
  feature_set['return_on_equity'] = df['roe']
  feature_set['organizational_profitability_ratio'] = df['ebitda'] / df['asst_tot']

  # liquidity
  feature_set['current_ratio'] = df['asst_current'] / (df['AP_st'] + df['debt_st'])
  feature_set['quick_ratio'] = (df['cash_and_equiv'] + df['AR']) / (df['AP_st'] + df['debt_st'])
  feature_set['cash_ratio'] = df['cash_and_equiv'] / df['asst_tot']

  # activity
  feature_set['receivables_turnover_ratio'] = df['rev_operating'] / df['AR']
  feature_set['asset_turnover_ratio'] = df['rev_operating'] / df['asst_tot']
  feature_set['inventory_turnover_ratio'] = df['COGS'] / (df['asst_current'] - df['AR'] - df['cash_and_equiv'])

  # growth
  if 'asset_growth' in df.columns:
    feature_set['asset_growth'] = df['asset_growth']
  if 'profit_growth' in df.columns:
    feature_set['profit_growth'] = df['profit_growth']
  if 'revenue_growth' in df.columns:
    feature_set['revenue_growth'] = df['revenue_growth']

  return feature_set


def process_ratios_harness(harness_df, summary_train_df, features):

  # turn the 'mean' index row of summary_train_df into a dict with columns as keys and row values as value
  summary_train_dict = summary_train_df.loc['mean'].to_dict()
  print(summary_train_dict)

  # step 1: prepare data and handle ROE and ROA calculations
  df = harness_df.replace('NaT', np.nan).copy()

  # calculate ROE for missing values where 'eqty_tot' is not zero
  df.loc[df['roe'].isnull() & (df['eqty_tot'] != 0), 'roe'] = (df['profit'] / df['eqty_tot']) * 100
  # calculate ROA for missing values
  df.loc[df['roa'].isnull(), 'roa'] = (df['prof_operations'] / df['asst_tot']) * 100

  # step 2: generate the feature set using the predefined function
  feature_set = calculate_financial_ratios(df)

  # step 3: replace infinite values with NaN
  feature_set.replace([np.inf, -np.inf], np.nan, inplace=True)

  # step 4: select necessary features only
  necessary_features = ['fs_year', 'id', 'stmt_date', 'legal_struct', 'ateco_sector'] + features
  print('features:', necessary_features)
  feature_set = feature_set[necessary_features]

  # step 5: impute feature_set with mean values from summary_train_dict
  for feature in features:
    feature_set[feature] = feature_set[feature].fillna(summary_train_dict[feature])

  return feature_set


# STEP 5: ateco sector rates

def obtain_sector_rates(harness_df):

  temp_feature_set = harness_df.copy()

  sector_mapping = {
    'Industrials': [(1, 3), (5, 9), (10, 33), (35, 35), (36, 39), (49, 53)],
    'Trade': [(45, 47)],
    'Construction': [(41, 43)],
    'Services': [(55, 56), (58, 63), (64, 66), (69, 75), (77, 82), (84, 84), (85, 85), (86, 88), (90, 93), (94, 96), (97, 98), (99, 99)],
    'Real Estate': [(68, 68)]
  }

  # function to categorize sector
  def get_sector_name(ateco_sector):
    for sector_name, ranges in sector_mapping.items():
      for start, end in ranges:
        if start <= ateco_sector <= end:
          return sector_name
    return 'Unknown'

  # apply function to create sector_name column
  temp_feature_set['sector_name'] = temp_feature_set['ateco_sector'].apply(get_sector_name)

  # default rates from train set
  sector_name_dict = {'Construction': 0.013976, 'Industrials': 0.013949, 'Real Estate': 0.008326, 'Services':0.012611, 'Trade': 0.015449, 'Unknown': 0}
  temp_feature_set['sector_rate'] = temp_feature_set['sector_name'].map(sector_name_dict)

  # drop sector_name
  temp_feature_set.drop(columns=['sector_name'], inplace=True)

  return temp_feature_set


# STEP 6: bound outliers by 1% and 99% from summary train DF

def bound_outliers_harness(harness_df, summary_train_df, features):

  for col in features:
    lower_bound = summary_train_df.loc['perc_1', col]
    upper_bound = summary_train_df.loc['perc_99', col]
    harness_df[col] = harness_df[col].clip(lower=lower_bound, upper=upper_bound)

  return harness_df


# STEP 7-backup: in the case that the new transformation doesn't work: it will resort to this transformation function

def old_transformations(harness_df, features):

  df = harness_df.copy()

  return df


# STEP 7: apply new transformation to features (except sector_rate)

def new_transformations(harness_df, features, mode = 'infer'):
  feat = ['total_equity', 'debt_service_coverage_ratio', 'leverage_ratio', 'return_on_assets', 'return_on_equity', 'receivables_turnover_ratio', 'asset_growth', 'profit_growth', 'revenue_growth', 'total_assets']

  df = harness_df.copy()

  if mode == 'train':
    # Initializing and fitting the Yeo-Johnson PowerTransformer
    transformer = PowerTransformer(method='yeo-johnson')
    feature_set_yjt = transformer.fit_transform(df[feat])
    df[feat] = feature_set_yjt

    # Saving the fitted transformer to a file
    joblib.dump(transformer, './data/yeo_johnson_transformer.pkl')

  # # # FOR HARNESS: MODE IS ALWAYS 'INFER' - we load train transformer
  else:
    yjt = joblib.load('./data/yeo_johnson_transformer.pkl')
    feature_set_yjt = yjt.transform(df[feat])
    df[feat] = feature_set_yjt
    
  return df

# STEP 8: standardize the harness DF

from scipy.stats import zscore
def standardize_harness(harness_df, summary_train_df, features):

  df = harness_df.copy()

  # write a function which takes in mean and std for a feature and standardizes that feature based on it
  def standardize_feature(feature, mean, std):
    df[feature] = (df[feature] - mean) / std
    return df

  # using train mean & std for standardization
  for col in features:
    mean = summary_train_df.loc['mean_standardize', col]
    std = summary_train_df.loc['std_standardize', col]
    df = standardize_feature(col, mean, std)

  return df


# STEP 9: output the final holdout dataset

def output_final_harness(harness_df, features):

  df = harness_df.copy()

  # only include the 11 features the model takes in
  df = df[features]
  return df


# helper function

def p(step):
  print(f'Step {step} Complete\n')


# HARNESS FUNCTION FOR ALL PRE-PROCESSING OF HOLDOUT DATA

def preprocess_holdout_data(holdout_data):

  final_features = ["total_equity", "debt_service_coverage_ratio", "leverage_ratio",
  "return_on_assets", "return_on_equity",  "receivables_turnover_ratio",
  "asset_growth", "profit_growth", "revenue_growth", "total_assets"]

  final_standardize_features = ["total_equity", "debt_service_coverage_ratio", "leverage_ratio",
  "return_on_assets", "return_on_equity",  "receivables_turnover_ratio",
  "asset_growth", "profit_growth", "revenue_growth", "total_assets", "sector_rate"]

  # STEP 1: obtain harness data
  harness_df = load_harness_df(holdout_data)
  p(1)

  # STEP 2: obtain helper data (summary of train & 2012 growth)
  summary_train_df, growth_2012_df = load_helper_dfs()
  p(2)

  # STEP 3: calculate growth-over-time features (using train DF)
  harness_growth_df = calc_growth_harness(harness_df, growth_2012_df)
  p(3)
  print(harness_growth_df)

  # STEP 4: calculate financial ratios & handle missing values (using train DF)
  harness_feature_set = process_ratios_harness(harness_growth_df, summary_train_df, final_features)
  p(4)
  print(harness_feature_set)

  # STEP 5: add ateco sector rates
  harness_feature_set_5 = obtain_sector_rates(harness_feature_set)
  p(5)
  print(harness_feature_set_5)

  # STEP 6: bound outliers by 1% and 99% (using train DF)
  harness_feature_set_6 = bound_outliers_harness(harness_feature_set_5, summary_train_df, final_features)
  p(6)
  pd.set_option('display.max_columns', None)
  print(harness_feature_set_6)

  # STEP 7: apply Yeo Johnson transformation (& log transform if it fails)
  try:
    harness_feature_set_7 = new_transformations(harness_feature_set_6, final_features)
    print('new transformation, YJ \n')
  except:
    harness_feature_set_7 = harness_feature_set_6
  p(7)
  print(harness_feature_set_7)

  # STEP 8: standardize the feature set (using train DF)
  harness_feature_set_8 = standardize_harness(harness_feature_set_7, summary_train_df, final_standardize_features)
  p(8)
  print(harness_feature_set_8)

  # STEP 9: output the final feature set
  final_holdout_set = output_final_harness(harness_feature_set_8, final_standardize_features)
  p(9)
  print(final_holdout_set)

  return final_holdout_set


# Function to obtain certain features

def obtain_features(num_features=11):

  # features found through univariate & multivariate analysis

  if num_features == 11:
    selected_features = ["total_assets", "sector_rate", "total_equity", "debt_service_coverage_ratio", "leverage_ratio", "return_on_assets", "return_on_equity", "receivables_turnover_ratio", "asset_growth", "profit_growth", "revenue_growth"]
  elif num_features == 9:
    selected_features = ["total_equity", "debt_service_coverage_ratio", "leverage_ratio", "return_on_assets", "return_on_equity",  "receivables_turnover_ratio", "asset_growth", "profit_growth", "revenue_growth"]
  elif num_features == 8:
    selected_features = ["total_equity", "debt_service_coverage_ratio", "leverage_ratio", "return_on_assets", "return_on_equity",  "receivables_turnover_ratio", "total_assets", "sector_rate"]
  elif num_features == 6:
    selected_features = ["total_equity", "debt_service_coverage_ratio", "leverage_ratio", "return_on_assets", "return_on_equity",  "receivables_turnover_ratio"]
  else:
    selected_features = ["total_equity", "debt_service_coverage_ratio", "leverage_ratio", "return_on_assets", "return_on_equity",  "receivables_turnover_ratio", "asset_growth", "profit_growth", "revenue_growth", "total_assets", "sector_rate"]

  return selected_features


# Function to only use selected features for our dataset based on model


def feature_selection(df, num_features=11):

  # obtain features depending on the model
  features = obtain_features(num_features)

  # crop feature set
  df_selected = df[features]

  return df_selected

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
"""
INFERENCE FUNCTIONS
"""
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

# Function to flatten ground truth and predictions lists across years

def flatten_pred(preds_probs):
    if isinstance(preds_probs, float):
        preds_flattened = [preds_probs]  # Wrap it in a list
    else:
        preds_flattened = [item for sublist in preds_probs for item in sublist]
  # return lists
    return preds_flattened


# Function which implements the exponential function with calibration params

def calibration_function(params, x):

  a = float(params[0])
  b = float(params[1])
  c = float(params[2])
  y = a * np.exp(b * x) + c

  return y


# Function which adjusts calibrated parameters so p > 0 + epsilon

def adjust_calibration_params(params, epsilon=0.002):

  a, b, c = params

  # at x = 0, y = a + c
  if a + c < epsilon:
    c = epsilon - a
  new_params = np.array([a, b, c])

  return new_params


# Function which maps predicted probabilities to calibrated probabilities

def obtain_calibrated_probs(preds_probs, params):

  calibrated_probs = []

  # map each predicted prob to calibrated prob
  for i in preds_probs:
    #original_prob = i
    prob = calibration_function(params, i)
    calibrated_probs.append(prob)

  return calibrated_probs


# Function which maps predicted probs to calibrated probs for each year

def obtain_calibrated_probs_years(preds_probs, params):

  cal_probs_years = []

  # iterate through each year
  for i in range(len(preds_probs)):
    curr_probs = [j for j in preds_probs[i]]

    # calibrate probabilities
    cal_probs = obtain_calibrated_probs(curr_probs, params)
    cal_probs_years.append(cal_probs)

  return cal_probs_years


# PREDICTION HARNESS: takes in a preprocessed feature set, outputs calibrated predictions

def prediction_harness(preprocessed_feature_set, model, model_type, calibration_params):

  # step 1: predict
  if model_type == 'logit':
    preds_probs = model.predict(preprocessed_feature_set)
    print('preds type:', type(preds_probs))
    print(preds_probs.head(10))
  else:
    preds_probs = model.predict_proba(preprocessed_feature_set)[:, 1]
    print('preds type:', type(preds_probs))
    print(preds_probs[:10])

  # step 2: adjust calibration parameters so that p > 0
  calibration_params = adjust_calibration_params(calibration_params)
  print('adjusted calibration: ', calibration_params)

  # step 3: map flattened probabilities to calibrated probabilities
  calibrated_probs = obtain_calibrated_probs(preds_probs, calibration_params)

  print('after calibration:', type(calibrated_probs))

  return calibrated_probs

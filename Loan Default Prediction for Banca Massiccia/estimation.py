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
from prediction import feature_selection

"""
Function to load training data from Google Drive as a pandas DF
"""

def load_train_df(train_fp='data/train.csv'):

  # read in DF
  df = pd.read_csv(train_fp)
  df = df.drop(columns=['Unnamed: 0'])

  return df

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

"""
Function to calculate growth over time given a current & prior value
"""

def calculate_one_growth(current, prior, first_year):

  # first year: set growth to 0
  if first_year:
    return 0

  # if prior is NaN: set to NaN
  elif pd.isna(prior):
    return float('nan')

  # if prior is 0: return Nan
  elif prior == 0:
    return float('nan')

  # growth calculation
  else:
    return (current - prior) / prior
    
"""
Function to obtain growth-over-time financial ratios for feature set
"""

def calculate_growth_features(df):

  # sort DF
  df_growth = df.copy()
  df_growth = df_growth.sort_values(['id', 'fs_year'])

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
  df_growth.drop(columns=['prior_year_assets', 'prior_year_profit', 'prior_year_revenue'], inplace=True)

  return df_growth

"""
Function to replace missing or null values in the DF for each record with the mean of that column up to & including a record's year
"""

def process_financial_data(df, start_year=2007, end_year=2012, impute_method='mean'):
  # step 1: Prepare data and handle ROE and ROA calculations
  df = df.replace('NaT', np.nan).copy()

  # calculate ROE for missing values where 'eqty_tot' is not zero
  df.loc[df['roe'].isnull() & (df['eqty_tot'] != 0), 'roe'] = (df['profit'] / df['eqty_tot']) * 100
  # calculate ROA for missing values
  df.loc[df['roa'].isnull(), 'roa'] = (df['prof_operations'] / df['asst_tot']) * 100

  # step 2: Generate the feature set using the predefined function
  feature_set = calculate_financial_ratios(df)

  # step 3: Replace infinite values with NaN
  feature_set.replace([np.inf, -np.inf], np.nan, inplace=True)

  # step 4: Filter by years and sort by 'fs_year'
  feature_set = feature_set[(feature_set['fs_year'] >= start_year) & (feature_set['fs_year'] <= end_year)]
  feature_set = feature_set.sort_values('fs_year')

  # step 5: Impute missing values by year
  for year in range(start_year, end_year + 1):
    yearly_data = feature_set[feature_set['fs_year'] <= year]
    columns_to_impute = feature_set.columns.difference(['fs_year', 'id', 'stmt_date', 'legal_struct', 'ateco_sector'])

    # choose imputation method
    impute_values = yearly_data[columns_to_impute].mean() if impute_method == 'mean' else yearly_data[columns_to_impute].median()

    # fill missing values for each year
    for col in columns_to_impute:
        feature_set.loc[feature_set['fs_year'] == year, col] = feature_set.loc[feature_set['fs_year'] == year, col].fillna(impute_values[col])

  # remove sorting by year by resetting the index order
  return feature_set


"""
Function to assign labels to the DF - 1 for default, 0 for non-default
"""

def label_default(df, feature_set):

  # convert 'stmt_date' and 'def_date' to datetime format
  df['stmt_date'] = pd.to_datetime(df['stmt_date'], format='%Y-%m-%d')
  df['def_date'] = pd.to_datetime(df['def_date'], format='%d/%m/%Y', errors='coerce')

  # initialize label column with 0's
  df['label'] = 0

  for index, row in df.iterrows():
    if pd.notnull(row['def_date']):
      # calculate 6 months and 1 year 6 months after 'stmt_date'
      start_date = row['stmt_date'] + pd.DateOffset(months=6)
      end_date = row['stmt_date'] + pd.DateOffset(months=18)
      # check if 'def_date' falls within this range
      if start_date <= row['def_date'] <= end_date:
        df.at[index, 'label'] = 1

  # set labels in feature set
  feature_set['label'] = df['label']
  feature_set['fs_year'] = df['fs_year']

  return feature_set

def obtain_sector_rates(df):

  temp_feature_set = df.copy()

  sector_mapping = {
    'Industrials': [(1, 3), (5, 9), (10, 33), (35, 35), (36, 39), (49, 53)],
    'Trade': [(45, 47)],
    'Construction': [(41, 43)],
    'Services': [(55, 56), (58, 63), (64, 66), (69, 75), (77, 82), (84, 84), (85, 85), (86, 88), (90, 93), (94, 96), (97, 98), (99, 99)],
    'Real Estate': [(68, 68)]
  }

  # Function to categorize sector
  def get_sector_name(ateco_sector):
      for sector_name, ranges in sector_mapping.items():
          for start, end in ranges:
              if start <= ateco_sector <= end:
                  return sector_name
      return 'Unknown'

  # Apply function to create sector_name column
  temp_feature_set['sector_name'] = temp_feature_set['ateco_sector'].apply(get_sector_name)

  sector_name_dict = {'Construction': 0.013976, 'Industrials': 0.013949, 'Real Estate': 0.008326, 'Services':0.012611, 'Trade': 0.015449}
  temp_feature_set['sector_rate'] = temp_feature_set['sector_name'].map(sector_name_dict)

  # drop sector_name
  temp_feature_set.drop(columns=['sector_name'], inplace=True)

  return temp_feature_set

"""
Function to obtain a list of numeric features in DF to apply transformations on
"""

def features_to_transform(df):

  # obtain features from DF
  features = list(df.columns)
  features = [feature for feature in features if feature not in ('fs_year', 'id', 'stmt_date', 'legal_struct', 'ateco_sector', 'label',
                                                                 'sector_name', 'sector_rate')]

  return features

"""
Function to bound outliers in DF
"""

def handle_outliers(df, columns):

    # clip outliers under 1% or greater than 99%
    for col in columns:
        lower_bound = df[col].quantile(0.01)
        upper_bound = df[col].quantile(0.99)
        df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)

    return df

"""
Function to remove records that don't meet accounting principles (i.e. where assets < equity)
"""

def remove_errors(df):

    # remove records where assets < equity
    if 'leverage_ratio' in df.columns:
        cond = (df['leverage_ratio'] < 1) & (df['leverage_ratio'] > 0)
        new_df = df[~cond]

    return new_df

"""
Function to apply log / sqrt transformations to given features in DF
"""

def apply_transformations(df, features):

  df = df.copy()

  # all features
  '''
  'total_assets', 'total_equity', 'solvency_debt_ratio', 'debt_to_equity_ratio', 'interest_coverage_ratio', 'debt_service_coverage_ratio', 'leverage_ratio',
  'lt_debt_to_capitalization_ratio', 'profit_margin_ratio', 'return_on_assets', 'return_on_equity', 'organizational_profitability_ratio', 'current_ratio',
  'quick_ratio', 'cash_ratio', 'receivables_turnover_ratio', 'asset_turnover_ratio', 'inventory_turnover_ratio'
  '''

  # transformations
  for feature in features:

    # handling values less than 0 in transformation
    min_value = df[feature].min()
    df[feature] = df[feature] - min_value + 1

    # now: applying transformations
    if feature == 'total_assets':
      df[feature] = np.log1p(df[feature])
    elif feature == 'total_equity':
      df[feature] = np.sqrt(df[feature])
    elif feature == 'solvency_debt_ratio':
      df[feature] = np.sqrt(df[feature])
    elif feature == 'debt_to_equity_ratio':
      df[feature] = np.log1p(df[feature])
    elif feature == 'interest_coverage_ratio':
      df[feature] = np.sqrt(df[feature])
    elif feature == 'debt_service_coverage_ratio':
      df[feature] = np.sqrt(df[feature])
    elif feature == 'leverage_ratio':
      df[feature] = np.log1p(df[feature])
    elif feature == 'lt_debt_to_capitalization_ratio':
      df[feature] = np.log1p(df[feature])
    elif feature == 'profit_margin_ratio':
      df[feature] = np.log1p(df[feature])
    elif feature == 'return_on_assets':
      df[feature] = np.sqrt(df[feature])
    elif feature == 'return_on_equity':
      df[feature] = np.sqrt(df[feature])
    elif feature == 'organizational_profitability_ratio':
      df[feature] = np.sqrt(df[feature])
    elif feature == 'current_ratio':
      df[feature] = np.log1p(df[feature])
    elif feature == 'quick_ratio':
      df[feature] = np.log1p(df[feature])
    elif feature == 'cash_ratio':
      df[feature] = np.sqrt(df[feature])
    elif feature == 'receivables_turnover_ratio':
      df[feature] = np.log1p(df[feature])
    elif feature == 'asset_turnover_ratio':
      df[feature] = np.log1p(df[feature])
    elif feature == 'inventory_turnover_ratio':
      df[feature] = np.sqrt(df[feature])
    elif feature == 'asset_growth':
      df[feature] = np.log1p(df[feature])
    elif feature == 'profit_growth':
      df[feature] = np.log1p(df[feature])
    elif feature == 'revenue_growth':
      df[feature] = np.log1p(df[feature])
    else:
      print('feature not found')

  return df

def old_transformations(df, feat):
  df = df.copy()

  # transformations
  for feature in feat:

    # handling values less than 0 in transformation
    min_value = df[feature].min()
    df[feature] = df[feature] - min_value + 1

    # now: applying transformations
    df[feature] = np.log1p(df[feature])

  return df

def new_transformations(df, feat, mode = 'train'):

  feat = ['total_equity', 'debt_service_coverage_ratio', 'leverage_ratio', 'return_on_assets', 'return_on_equity', 'receivables_turnover_ratio', 'asset_growth', 'profit_growth', 'revenue_growth', 'total_assets']

  df = df.copy()
  # yjt = YeoJohnsonTransformer(variables=feat)
  # feature_set_yjt = yjt.fit_transform(df)

  if mode == 'train':
    # Initializing and fitting the Yeo-Johnson PowerTransformer
    transformer = PowerTransformer(method='yeo-johnson')
    feature_set_yjt = transformer.fit_transform(df[feat])
    df[feat] = feature_set_yjt

    # Saving the fitted transformer to a file
    joblib.dump(transformer, '/content/drive/MyDrive/NYU_Fall2024/ML in Finance/yeo_johnson_transformer.pkl')
  else:
    yjt = joblib.load('/content/drive/MyDrive/NYU_Fall2024/ML in Finance/yeo_johnson_transformer.pkl')
    feature_set_yjt = yjt.transform(df[feat])
    df[feat] = feature_set_yjt

  return df

"""
Function to apply standardization to given features in feature set
"""

def standardize_df(df, features):

  # standardize features
  for feature in features:
    df[feature] = zscore(df[feature])

  return df

def preprocessing_harness(train_fp='data/train.csv'):

  # step 1: load training data
  df = load_train_df(train_fp)
  print('step 1 complete')

  # step 2: calculate growth-over-time features
  df_growth = calculate_growth_features(df)
  print('step 2 complete')

  # step 3: calculate financial ratios & handle missing values
  feature_set = process_financial_data(df_growth)
  print('step 3 complete')

  # step 4: label data
  feature_set = label_default(df, feature_set)
  print('step 4 complete')

  # step 5: obtain sector rate
  feature_set = obtain_sector_rates(feature_set)
  print('step 5 complete')

  # step 6: obtain quantitative features
  features = features_to_transform(feature_set)
  print('step 6 complete')

  # step 7: bound outliers
  feature_set = handle_outliers(feature_set, features)
  print('step 7 complete')

  # step 8: remove errors
  feature_set = remove_errors(feature_set)
  print('step 8 complete')

  # step 9: NEW - apply transformations
  try:
    feature_set = new_transformations(feature_set, features)
    print('step 9 complete')
  except Exception as e:
    feature_set = old_transformations(feature_set, features)
    print('step 9 complete')

  # step 10: NEW - standardize
  std_features = [f for f in features]
  std_features.append('sector_rate')
  feature_set = standardize_df(feature_set, std_features)
  print('step 10 complete')

  return feature_set


"""
Function to train a model and run inference using walk-forward analysis
"""

def walk_forward_analysis(df, step_size=1, model='logit', thresh=0.1, version=1):

  df = df.copy()
  print(df.shape)

  # fill in lists of values for each year
  ground_truth = []
  predictions_probs = []
  predictions_binary = []

  # iterate through each year
  min_year = df['fs_year'].min()
  max_year = df['fs_year'].max()
  for year in range(min_year, max_year):
    train_df = df[df['fs_year'] <= year]
    test_df = df[df['fs_year'] == year + 1]

    # output the shape of train & test DFs
    print(year)
    print(train_df.shape)
    print(test_df.shape)
    print('\n')

    # add current year's ground truth to list
    curr_ground_truth = test_df['label'].tolist()
    ground_truth.append(curr_ground_truth)

    # set up train & test DF
    X_train = train_df.drop(columns=['label', 'fs_year'])
    y_train = train_df['label']
    X_test = test_df.drop(columns=['label', 'fs_year'])

    # feature selection
    X_train = feature_selection(X_train, model)
    X_test = feature_selection(X_test, model)

    # LOGIT
    if model == 'logit':

      # add intercept
      X_train_w_const = sm.add_constant(X_train, has_constant='add')
      X_test_w_const = sm.add_constant(X_test, has_constant='add')

      # define the logistic regression model
      logit_model = sm.Logit(y_train, X_train_w_const)

      # add L1 regularization while fitting
      result = logit_model.fit_regularized(method='l1', alpha=0.5, maxiter=1000)

      # print the summary of the model
      ''' print(result.summary()) '''

      # obtain predicted probabilities & binary labels for the test set
      predicted_probs = result.predict(X_test_w_const)
      binary_preds = (predicted_probs >= thresh).astype(int)

    # GRADIENT BOOST
    elif model == 'gb':

      # define & train the model
      gb_model = GradientBoostingClassifier(n_estimators=100, random_state=42)
      gb_model.fit(X_train, y_train)

      # obtain predicted probabilities & binary labels for the test set
      predicted_probs = gb_model.predict_proba(X_test)[:, 1]
      binary_preds = [1 if prob >= 0.1 else 0 for prob in predicted_probs]

    # RANDOM FOREST
    elif model == 'rf':

      # define & train the model
      rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
      rf_model.fit(X_train, y_train)

      # obtain predicted probabilities & binary labels for the test set
      predicted_probs = rf_model.predict_proba(X_test)[:, 1]
      binary_preds = [1 if prob >= 0.1 else 0 for prob in predicted_probs]

    # NOT DEFINED
    else:
      print('model not found')

    # append predictions to lists
    predictions_probs.append(predicted_probs)
    predictions_binary.append(binary_preds)

  # save the model to .pkl file
  model_dict = {'logit': 'Logit', 'gb': 'GradientBoost', 'rf': 'RandomForest'}
  model_name = model_dict[model]
  filepath = f'/content/drive/MyDrive/ML_Finance/'
  with open(filepath+f'{model}_model_v{version}.pkl', 'wb') as f:
    if model == 'logit':
      pickle.dump(result, f)
    elif model == 'gb':
      pickle.dump(gb_model, f)
      #joblib.dump(gb_model,"gb_model")
    elif model == 'rf':
      pickle.dump(rf_model, f)
    else:
      print('model not found')

  # return ground truth & predictions
  return ground_truth, predictions_binary, predictions_probs


"""
Function to train our model & obtain predictions for each year
"""

def call_walk_forward(df, model='logit'):

  # drop unnecessary columns
  if 'id' in df.columns:
    df = df.drop(columns=['id'])
  if 'stmt_date' in df.columns:
    df = df.drop(columns=['stmt_date'])
  if 'legal_struct' in df.columns:
    df = df.drop(columns=['legal_struct'])
  if 'ateco_sector' in df.columns:
    df = df.drop(columns=['ateco_sector'])

  # call the walk-forward analysis function
  ground_truth, preds_binary, preds_probs = walk_forward_analysis(df, model=model,version=2)

  # return output of analysis
  return ground_truth, preds_binary, preds_probs


"""
Function to output metrics for each year
"""

def output_metrics(ground_truth, preds_probs):

  year_list = [2008, 2009, 2010, 2011, 2012]

  # iterate through each year
  for i in range(len(ground_truth)):
    print(year_list[i])

    # AUC
    print('AUC:', roc_auc_score(ground_truth[i], preds_probs[i]))

    # binary preds
    if isinstance(preds_probs[i], pd.Series):
      binary_preds = (preds_probs[i] >= 0.1).astype(int)
    else:
      binary_preds = [1 if prob >= 0.1 else 0 for prob in preds_probs[i]]

    # metrics
    print('accuracy:', accuracy_score(ground_truth[i], binary_preds))
    print('precision:', precision_score(ground_truth[i], binary_preds))
    print('recall:', recall_score(ground_truth[i], binary_preds))
    print('f1:', f1_score(ground_truth[i], binary_preds))
    print('\n')


def main():

  df = preprocessing_harness("data/train.csv")
  model = 'gb'
  if model == 'logit':
    ground_truth, preds_binary, preds_probs = call_walk_forward(df, model='logit')
    
  if model == 'gb':
    ground_truth, preds_binary, preds_probs = call_walk_forward(df, model='gb')
    
  if model == 'rf':
    ground_truth, preds_binary, preds_probs = call_walk_forward(df, model='rf')
  
  metrics = output_metrics(ground_truth, preds_probs)
  print(metrics)


if __name__ == "__main__":
  main()

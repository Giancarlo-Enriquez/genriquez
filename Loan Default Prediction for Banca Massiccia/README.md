# Loan Default Prediction for Banca Massiccia 

## Problem Statement

Banca Massiccia, a large Italian bank, aims to strengthen their loan underwriting approaches by using machine learning techniques to predict the likelihood that a potential borrower will default on a principal or interest payment for a prospective loan over the next 12 months. Loan underwriting is the process of which the lender decides whether a borrower is creditworthy and should receive a loan. 
Understanding this process is important since failing to capture important financial indicators for our borrower’s data, like leverage and profitability, may result in missed opportunities for the bank or more importantly the unexpected approval of loans that are likely to default. In the past, traditional credit risk models in finance relied on financial ratios and demographic factors, while machine learning approaches have explored algorithms like decision trees and K-means clustering to improve default prediction, often overlooking financial literacy factors. Using Banca Massiccia’s extensive historical data, our approach aims to bridge the gap between traditional and machine learning models by combining financial knowledge and advanced data mining techniques to uncover patterns and relationships that determine the 12-month probability of default.  

## Project Workflow
 
* **estimation.py:** this file pre-processes the training set in a series of steps (explained in prediction.py) and trains either a logistic regression, gradient boost, or random forest model. The final model was a Gradient Boost model trained on 11 features. It was trained using walk-forward analysis, and we output metrics.

* **prediction.py:** in this file, the holdout set is preprocessed by calculating growth features, calculating financial ratios, handling missing values, bounding outliers, transforming features, & standardizing features, feature selection. And none of these calculations depend on the holdout set at all. Instead, the mean, std, 1%, and 99% of the training set from 2007-2012, as well as asset revenue & profit from 2012, are contained in our data folder to help with calculations. THe saved gradient boost model takes in the pre-processed holdout and runs inference. It uses hardcoded calibration parameters which were found by fitting an estimation function on predictions of the Gradient Boost model to obtain calibrated probabilities. 

* **harness.py:** this file will take in input.csv and output.csv as arguments, load the Gradient Boost model, preprocess holdout data, run the holdout data through our model, & get back calibrated probabilities which are saved to output.csv.

## Data Files

* **NOV17-FINAL-preprocessed_train_mean_std_percentiles.py:** this is a CSV of 5 rows & 13 columns (11 final features + label + fs_year). The 'label' and 'fs_year' columns are not used at all. 
    - The first row contains the mean value of each feature in the train set up to 2012. This is used for *handling missing values* in the holdout set.
    - The second & third rows contain the 1% and 99% of each feature in the train set up to 2012. This is used for *bounding outliers* in the holdout set.
    - The fourth & fifth rows contain the mean & standard deviation of each feature in the train set up to 2012 *after* transforming the train set with the Yeo Johnson transform (it is calculated after transformations because we standardize the holdout set after transforming it). This is used for *standardizing* the holdout set.

* **unprocessed_2012_growth_data.csv:** this is a CSV of ~180k rows (all of the 2012 data in the train set) & 5 columns (id, fs_year, asst_tot, profit, rev_operating). This is used for *calculating growth features* - specifically growth in total assets, profit, and revenue from the year 2012 to 2013. 

* **yeo_johnson_transformer.pkl:** this is a scikit-learn PowerTransformer which was used to tranform training data to be more Gaussian-like. We save this model so that no calculations have to be done using holdout data for transformation. 

## Usage

The harness can be executed using the following command:

```
python harness.py --input_csv holdout.csv --output_csv output.csv
```

## Data Fixes
In the evening of Nov. 20th we found out there were 3 issues with this harness:
1. **Leakage** is occurring (datapoints in holdout set are affecting calculations for other holdout datapoints)
2. The output predictions not being outputted in a **different order** that they were coming in
3. At some point up until Step 6 (bounding outliers), there is at least 1 case in which the output is an **empty DF**

We will now talk through how we addressed each of these errors.

#### Leakage

1. When we handed in this harness, we thought all possible causes for leakage had been addressed. To handle missing values in the holdout, we use training data means. To bound outliers in the holdout, we use training data 1-percentile & 99-percentile cutoffs. To transform holdout data, we load a transformer which was trained on training data only & saved to this harness. To standardize the features in the holdout, we use training data means & standard deviations. 

However, when we looked back at the calculation of the *growth features* we noticed that there is potential leakage occurring. The reason for this is because we set up our function to run in optimal speed by using the *shift* function. More background: we first stack 2012 data on top of holdout data & sort by firm-id and year (this is also incorrect and we will talk about this in a bit). Then, to obtain the *previous year's* value of assets, profit, & revenue, we use the shift function. Unfortunately, when we shift values down, if there is at least one firm-id in the holdout which isn't present in the 2012 data, then the row before it will be another holdout set row. So the value of one holdout datapoint will enter the row of another. Additionally, we handle firms differently if it is their "first year" in the dataset. The issue is *how* we determine if it's a firm's first year. Our (unfortunate) logic was to check whether the previous row's ID matches this row's ID. Since they're sorted by both ID and year, if they match then we know that 2013 is not the firm's first year. If they do not match, we know it's the firm's first year. However, in a *roundabout* way, this IS leakage because the previous row will be a holdout row if it is the first year for the current row's firm. Overall, there were multiple logic mishaps which led to calculations where leakage occurred in the growth features calculation function (Step 3 in pre-processing).

To fix this error, we no longer stack 2012 data on top of the holdout set. Additionally, we no longer process the holdout set together. Instead, we now run through the holdout data iteratively, row-by-row. For each row: we look at the current row's ID. We then check if that ID is present in the 2012 set of IDs. 
- If it is present, then we know that the holdout year is not the first year of this firm's presence. We take the 2012 row for that ID and use its value of assets, profit, & revenue as "prior assets", "prior profit", and "prior revenue". The current holdout row's for assets, profit, & revenue are the current values. Using the function (current-prior)/prior for growth, we then calculate growth for THIS row of the holdout set only. 
- If it is not present, then we know that the holdout year IS the first year of this firm's presence. We set asset growth, profit growth, & revenue growth for THIS row of the holdout set only to be 0.

This eliminates any leakage by processing holdout datapoints one-by-one to ensure that no datapoint in the holdout is influencing the calculation of any other holdout datapoint features. Since the code is less optimized now (there is a for-loop of higher computational complexity) it does take a while longer to run. However, it very much meets the requirements mentioned by the Professor of at least 250 rows/sec. We timed the fixed harness on a fake holdout set of 100,000 datapoints and it took approx. 20.97 seconds to run, giving an average of ~4k rows/sec.

#### Different Order

2. The issue causing datapoints to be outputted in a different order than they are inputted was from the same function as above. We had previously sorted the stacked DF by firm ID and year by calling the function *.sort_values()*, causing the holdout set to be re-arranged and thus for predictions to be outputted in a different order. We have since fixed this issue by removing all sorting and re-ordering on the holdout set. 

#### Empty DataFrame

3. This was a tough problem to crack. Step 6 bounds outliers by using the lower & upper bounds saved from the train set for each feature. *No value* of the lower & upper bounds are NaN. We tested this function inputting a fake harness of different data types, of 1 row only, of 1 row where every value was miles below the lowest lower bound, of 1 row with all NaNs (except for fs_year) -- nothing was causing the function to output an empty DF. 

The *only* place filtering out datapoints is occurring anywhere in pre-processing is, yet again, inside the growth features calculation function. Since we stacked 2012 growth data on top of the holdout set in our previous version, at the end of calculating growth we remove training data by filtering for rows where fs_year = 2013. When we tested a fake holdout set of 1 row with fs_year = 2014, our harness breaks. An empty DF is returned by Step 6 - as well as Steps 3-5, because the output of the growth feature calculation function is an empty DF after the filtering occurs. 

We handle this error by removing the part of the code which filters out datapoints from the holdout set. This was able to be removed because, as explained in (1), we changed our logic in that function now to iterate row-by-row through the harness DF to remove leakage. Thus, no rows are added to the holdout, which removes the need for rows to be filtered out.

#### Final Thoughts

4. We apologize for not having noticed these issues before submitting our harness. In the professional world, that wouldn't fly! So we want to thank you for providing us 24 hours to try to locate & eliminate errors. Hope we got all of them. 

## Contributors 

* Isha Slavin
* Giancarlo Enriquez
* Poojitha Kolli
* Sheel Patel


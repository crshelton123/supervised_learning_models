# Supervised Learning Models

## Overview of Analysis

The purpose of this analysis was to identify the most useful supervised machine learning model that could most accurately predict whether a loan was going to be classified as healthy or risky. This machine learning model could potentially help credit monitors make financial decisions about loan applicants.  The independent variables were things like income, loan size, interest rate, DTI ratio, derogatory marks, etc. The target variable is either "0" healthy or "1" risky. For this problem we used Logistic Regression, which is very good with binary results. 

The original dataset contains 75,036 healthy examples and only 2500 risky examples. This makes this an imbalanced problem. I isolated the dependent variable as the target "y", then created a dataframe with the other variables. I completed a train-test-split of the raw data. 

Then I initiated the Logistic Regression model, fit it with the training data, and predicted on the X-test dataset. When I compared the predictions to the y_test results, it yielded the below results. 

I ran the machine learning model twice, once with the original data and again after oversampling the data in order to compare results. In order to oversample I had to import and initiate the RandomOversampler module, then fit it with the raw training data. After oversampling, I have a balanced dataset of 56,271 of each outcome. Then I was able to run the Logisitic Regression model again with the oversampled data and it yielded the below results. 

---
## Results

LogisticRegression Model on original dataset:
* Precision
    * Healthy "0" - 1.00
    * Risky "1" - 0.85
* Accuracy
     * 0.95
* Recall
     * Healthy "0" - 0.99
     * Risky "1" - 0.91
     
LogisticRegression on oversampled dataset:
* Precision
    * Healthy "0" - 1.00
    * Risky "1" - 0.84
* Accuracy
     * 0.99
* Recall
     * Healthy "0" - 0.99
     * Risky "1" - 0.99

---
## Summary
The original model has an overall accuracy of 0.95, which seems not bad but accuracy scores can be susceptible to imbalanced data. The model also has a precision of 1.0 on healthy loans, which means there were not any false positives. This means not any risky loans labeled as healthy. The recall also shows almost perfect at 0.99, which means almost no false negatives. The model is not as strong with the predictions of "1" or risky loans, due to the imbalanced data. The precision is only .85 and the recall 0.91. These are not bad, but could be better. 

The model fit with the oversampled data has an improved overall accuracy of 0.99. It also still has a perfect precision score for "0" or healthy loans. The precision for high risk loans is 0.84, essentially the same as before(0.85). However, the recall score is much better than previously, going from 0.91 to 0.99, meaning there were fewer false negatives.

Overall if I had to choose one of thse models I would go with the oversampled model due to fewer false negatives. However, I noticed that this assignment's starter code did not include any directions for scaling the data. With such large difference between independent variables like loan amount(a large number) and debt-to-income ratio(decimal) that could be throwing this model off. I would suggest scaling the dataset using the StandardScaler and then rerunning the oversampler and using the Logistic Regression model with that data. I would be interested to see an improvement in the model results using the scaled and oversampled data.  
# Sales of Summer Clothes in E-commerce Wish

This repository contains the code and documentation for the data mining project titled "Sales of Summer Clothes in E-commerce Wish" as part of the Data Mining course at Vietnam International University - Ho Chi Minh City. The project involves data pre-processing, analysis, and prediction using machine learning algorithms.

## Table of Contents

1. [Introduction](#introduction)
2. [Data Pre-processing](#data-pre-processing)
    - [Data Cleaning](#data-cleaning)
    - [Data Analysis](#data-analysis)
3. [Prediction Algorithms](#prediction-algorithms)
4. [Model Evaluation](#model-evaluation)
    - [Naive Bayes Evaluation Results](#naive-bayes-evaluation-results)
    - [Random Forest Evaluation Results](#random-forest-evaluation-results)
    - [Comparison](#comparison)
5. [Conclusion](#conclusion)
6. [References](#references)

## Introduction

In recent years, e-commerce platforms have emerged as significant disruptors, transforming traditional physical transactions into online interactions. This project aims to analyze sales data of summer clothes on an e-commerce platform and predict sales performance using machine learning algorithms.

The dataset used in this project contains 43 columns, including product listings, ratings, and sales performance metrics.

## Data Pre-processing

### Data Cleaning

1. **Dropping Unnecessary Columns**: Removed irrelevant columns from the dataset.
2. **Removing Duplicate Columns**: Eliminate duplicate columns to avoid redundant information.
3. **Handling Missing Values**: Addressed missing values by filling numerical values with the mean and categorical values with the most frequent value or 'Unknown'.
4. **Normalizing**: Standardized values in categorical columns to ensure consistency.
5. **Normalizing Color Variations**: Standardized color names and created binary columns for color representation.
![Data Sample](image.png)
### Data Analysis

- Created visualizations to analyze units sold distribution, rating distribution, and price distribution.
- Used box plots to compare rating distributions across different sales categories.

## Prediction Algorithms

Implemented prediction models using the Weka library to classify products into "slow," "stable," and "hot" categories based on their sales performance. Evaluated various algorithms, including Decision Tree, SMO, Linear Regression, Naive Bayes, and Random Forest, focusing on Naive Bayes and Random Forest due to their higher accuracy.

## Model Evaluation

### Naive Bayes Evaluation Results

- Achieved an accuracy of 81.4672%
- Kappa Statistic: 0.7106
- Mean Absolute Error (MAE): 0.1219
- Root Mean Squared Error (RMSE): 0.3347
- Runtime: 15 seconds

### Random Forest Evaluation Results

- Achieved an accuracy of 85.7143%
- Kappa Statistic: 0.7777
- Mean Absolute Error (MAE): 0.1194
- Root Mean Squared Error (RMSE): 0.2444
- Runtime: 15 seconds

### Comparison

The Random Forest model outperformed the Naive Bayes model in terms of accuracy, Kappa statistic, and error rates. However, Naive Bayes had a shorter runtime for both training and prediction.

## Conclusion

The Random Forest algorithm is the superior model for predicting sales performance in this project, offering higher accuracy and better overall performance metrics compared to Naive Bayes.

## References

1. Chittawar, P. (n.d.). Summer clothing sales prediction [Kaggle notebook]. Retrieved May 25, 2024, from [Kaggle](https://www.kaggle.com/code/parthchittawar/summer-clothing-sales-prediction)
2. Ziegler, A., & König, I. R. (2014). Mining data with random forests: current options for real-world applications. Wiley Interdisciplinary Reviews: Data Mining and Knowledge Discovery, 4(1), 55-63.
3. Singh, S. N., & Sarraf, T. (2020, January). Sentiment analysis of a product based on user reviews using random forests algorithm. In 2020 10th International conference on cloud computing, data science & engineering (Confluence) (pp. 112-116). IEEE.
4. Huang, J., Lu, J., & Ling, C. X. (2003, November). Comparing naive Bayes, decision trees, and SVM with AUC and accuracy. In Third IEEE International Conference on Data Mining (pp. 553-556). IEEE.
5. Youn, E., & Jeong, M. K. (2009). Class-dependent feature scaling method using naive Bayes classifier for text data mining. Pattern Recognition Letters, 30(5), 477-485.
6. Jiang, L., Wang, D., Cai, Z., & Yan, X. (2007). Survey of improving naive bayes for classification. In Advanced Data Mining and Applications: Third International Conference, ADMA 2007 Harbin, China, August 6-8, 2007. Proceedings 3 (pp. 134-145). Springer Berlin Heidelberg.

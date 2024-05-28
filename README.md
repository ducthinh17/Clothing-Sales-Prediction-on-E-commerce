# Clothing-Sales-Prediction-on-E-commerce
Discover insights into consumer behavior and product success in the realm of summer clothing sales with this comprehensive dataset from the Wish platform. Dive into product listings enriched with ratings and sales performance metrics, providing a unique opportunity to understand what drives the popularity of certain items during the summer season
# Sales of Summer Clothes in E-commerce Wish

## Data Mining Course: Programming Assignment

### Team Members:
- Phạm Lê Đức Thịnh (ITDSIU20085)
- Nguyễn Thanh Bình (ITDSIU20056)
- Ung Thị Hoài Thương (ITDSIU20028)
- Võ Thị Ngọc Thảo (ITDSIU20083)
- Trương Quân Bảo (ITDSIU20120)
- Nguyễn Phương Minh Ngọc (ITDSIU18045)

## Table of Contents
1. [Introduction](#introduction)
2. [Data Preprocessing](#data-preprocessing)
    - [Data Cleaning](#data-cleaning)
    - [Data Analysis](#data-analysis)
3. [Prediction Algorithms](#prediction-algorithms)
4. [Model Evaluation](#model-evaluation)
    - [Random Forest Evaluation](#random-forest-evaluation)
    - [Naive Bayes Evaluation](#naive-bayes-evaluation)
    - [Comparison](#comparison)
5. [Conclusion](#conclusion)
6. [References](#references)

## Introduction
E-commerce platforms have significantly transformed traditional physical transactions into online interactions, particularly in the developing world. Understanding market operations is critical for effective business models. This project uses the dataset "Sales of Summer Clothes in E-commerce Wish" to analyze product listings, ratings, and sales performance.

## Data Preprocessing
### Data Cleaning
1. **Dropping Unnecessary Columns**: Removing columns that are not required.
2. **Removing Duplicate Columns**: Eliminating redundant columns.
3. **Handling Missing Values**: Filling missing numerical values with the mean and categorical values with the most frequent value or 'Unknown'.
4. **Normalizing**: Standardizing the `product_variation_size_id` and `product_color` columns.
5. **Encoding**: Converting categorical variables into numerical values.
6. **Converting Currency Values**: Ensuring consistency in currency values.
7. **Normalizing Color Variations**: Removing non-alphanumeric characters, converting values to lowercase, and mapping compound color names to base colors.

### Data Analysis
- **Units Sold Distribution**: Bar chart showing the frequency distribution of units sold.
- **Price Distribution**: Histogram and KDE plot of product prices.
- **Rating Distribution Across Sales Categories**: Box plot of ratings across Slow, Stable, and Hot sales categories.

## Prediction Algorithms
Implemented prediction algorithms using the Weka library to classify product ratings into "slow", "stable", and "hot" categories. The primary algorithms evaluated were Naive Bayes and Random Forest.

## Model Evaluation
### Random Forest Evaluation
- Achieved higher accuracy and better overall performance metrics compared to Naive Bayes.
- Metrics: Accuracy, Kappa statistic, Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and runtime.

### Naive Bayes Evaluation
- Achieved a significantly shorter runtime for both training and prediction compared to Random Forest.
- Metrics: Similar to those evaluated for Random Forest.

### Comparison
- Random Forest outperformed Naive Bayes in accuracy, Kappa statistic, and error rates.
- Naive Bayes had a shorter runtime but higher error rates compared to Random Forest.

## Conclusion
- Random Forest algorithm demonstrated superior performance with higher accuracy and better overall metrics.
- Naive Bayes was faster but less accurate.

## References
1. Chittawar, P. (n.d.). Summer clothing sales prediction [Kaggle notebook]. Retrieved from [Kaggle](https://www.kaggle.com/code/parthchittawar/summer-clothing-sales-prediction)
2. Ziegler, A., & König, I. R. (2014). Mining data with random forests: current options for real‐world applications. Wiley Interdisciplinary Reviews: Data Mining and Knowledge Discovery, 4(1), 55-63.
3. Singh, S. N., & Sarraf, T. (2020). Sentiment analysis of a product based on user reviews using random forests algorithm. In 2020 10th International conference on cloud computing, data science & engineering (Confluence) (pp. 112-116). IEEE.
4. Huang, J., Lu, J., & Ling, C. X. (2003). Comparing naive Bayes, decision trees, and SVM with AUC and accuracy. In Third IEEE International Conference on Data Mining (pp. 553-556). IEEE.
5. Youn, E., & Jeong, M. K. (2009). Class dependent feature scaling method using naive Bayes classifier for text data mining. Pattern Recognition Letters, 30(5), 477-485.
6. Jiang, L., Wang, D., Cai, Z., & Yan, X. (2007). Survey of improving naive bayes for classification. In Advanced Data Mining and Applications: Third International Conference, ADMA 2007, Harbin, China, August 6-8, 2007. Proceedings, 3 (pp. 134-145). Springer Berlin Heidelberg.

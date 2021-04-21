# Lead Conversion Classification Modeling Project
This is my second practicum project for the MSDS program at Regis University.

# Introduction
The sales pipeline is an important aspect of any sales and marketing team. It is made of every step in the sales process and is unique to each organization. In general, the sales pipeline begins with lead generation, trying to draw in as many potential customers as possible through advertising, sales calls, and other forms of engagement with the public or target market. It then funnels through steps which may include lead qualificiation, engagement with the customer, quoting, negotiation, purchase, and, in optimal cases, customer loyalty. With each step in the sales pipeline, potential buyers may decide they do not want to continue with the purchase and the funnel narrows through to the last step. Because of this funneling concept, companies try to bolster their lead generation to maximize potential revenue.

![Sales Pipeline Example](https://github.com/tsgruman/regis-practicum-leads-classification/blob/main/assets/sales-pipeline.png)

Data analysis is becoming an increasingly important tool for optimizing lead conversion. Proper data collection and analysis can provide sales and marketing teams real insight into their customer base and the trajectory of their markets, allowing them to effectively target potential new business, maximize their ROI, and produce a more efficient sales pipeline.

In this project, I will analyze data pertaining to lead conversion and create various classification models to predict whether or not a lead will convert to a customer.

# Tools
This project was completed in R using RStudio. 

Libraries used include:
* tidyverse - Data manipulation and plotting
* scales - Graphics system manipulation
* xlsx - Manipulating xls files
* RRF - Feature selection for Random Forest models
* Boruta - Feature selection tools
* ggpubr - Data visualization for ggplot2
* corrplot - Correlation matrix tools
* caret - Various functions for classification and regression modeling
* randomForest - Random Forests classfication and regression modeling
* xgboost - Extreme Gradient Boosting modeling interface
* e1071 - Machine learning functions, including clustering, support vector machines, naive Bayes, and more
* pROC - ROC (receiver operating characteristics) tools, including plotting and AUC (area under the curve) calculation

# Data
The data for this project was downloaded from the Lead Scoring X Online Education Kaggle project found at https://www.kaggle.com/lakshmikalyan/lead-scoring-x-online-education?select=Leads+X+Education.csv. It is based on lead data collected by X Education, a company that "sells online courses to industry professionals." The data consists of web activity, generic customer-volunteered information, scores for each lead generated by the company, and whether or not a lead converted (i.e. purchased a course). The dataset contains over 9000 observations of a mix of 37 numeric and categorical variables.

## Data cleaning
Data cleaning consisted of converting all categorical variables to numerical values for machine learning modeling. This was done with label encoding and one-hot-encoding, splitting customer-selected dropdown values (e.g. Lead Origin, Lead Source, Specialization, etc.) to individual columns with binary 0 or 1 values to indicate the customer's selection. 

Additionally, many null values needed to be cleaned and removed. For example, customers who did not make a selection resulted in "Select" values in the data. Although a model may interpret this as a legitimate value to consider, "Select" values are essentially NULL and needed to be removed.

Finally, I dropped multiple columns that were not pertinent to my modeling needs. These included ID numbers, location data, tags, and Lead Quality, which is a score based on employee "intuition" and I deemed too subjective to use.

# Exploratory Data Analysis
To begin my exploratory data analysis, I plotted the count of my target variable Converted leads to view the distribution of converted vs. unconverted leads. There are about 1/3 more unconverted leads (value = 0) compared to converted leads.

![Count of Converted Leads](https://github.com/tsgruman/regis-practicum-leads-classification/blob/main/assets/converted-leads.png)

I wanted to know how many converted leads originated from advertising and which advertising platform. It appears most converted leads actually did not come through advertising.

![Converted Leads Ads](https://github.com/tsgruman/regis-practicum-leads-classification/blob/main/assets/ads.png)

Since the vast majority of converted leads did not come from ads, I wanted to see what else may have played a factor in lead conversion. Time on website and page views are both important search engine optimization (SEO) values, so I plotted variables against these. The following shows leads by time on website and page views per visit. It also shows lead origins by time on website and page views. 

![Time on Website and Page View stats](https://github.com/tsgruman/regis-practicum-leads-classification/blob/main/assets/converted-lead-stats.png)

The converted leads by page views revealed significant outlier data, so I examined a boxplot of page views. According to the summary print of the dataset, average page view values are about 2 per lead. The boxplot revealed page views far beyond that value, which may indicate bot activity and may not be true leads. For this reason, I removed outlier page view values to further clean the dataset.

![Page Views distribution](https://github.com/tsgruman/regis-practicum-leads-classification/blob/main/assets/page-views-boxplot.png)

# Feature Selection

# Classification Modeling 

## K-Nearest Neighbors (KNN)

## Random Forest

## Extreme Gradient Boosting (XGBoost)

## Support Vector Models (SVM)

# Discussion

# Further Research

# References

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
For classification modeling, I needed to reduce the number of features in my dataset to choose only the most important variables that may impact the output. I wanted to reduce features from the existing 126 - the number of variables grew greatly with one-hot encoding - down to between 5 - 10 variables. To achieve this, I used three methods: Boruta algorithm (Boruta package), rpart model (rpart package), and recursive feature selection (rfe from caret package). I then compared the results and selected the top features that consistently appeared in each method. 

## Comparing Results
Results from the Boruta algorithm:
```
> boruta <- Boruta(Converted ~ ., data = train, doTrace = 1)
> imp_score <- attStats(boruta)
> mps = imp_score[imp_score$decision != 'Rejected', c('meanImp', 'decision')]
> head(imps[order(-imps$meanImp), ], 10)
                                                                       meanImp  decision
Total.Time.Spent.on.Website                                           53.22589 Confirmed
Asymmetrique.Activity.Score                                           47.51818 Confirmed
Lead.Profile.Potential.Lead                                           26.75731 Confirmed
What.matters.most.to.you.in.choosing.a.course.Better.Career.Prospects 25.78960 Confirmed
What.is.your.current.occupation.Working.Professional                  25.72504 Confirmed
Lead.Origin.Lead.Add.Form                                             23.05170 Confirmed
Last.Activity.SMS.Sent                                                22.52003 Confirmed
Last.Notable.Activity.SMS.Sent                                        21.41469 Confirmed
What.is.your.current.occupation.Unemployed                            18.73294 Confirmed
Last.Notable.Activity.Modified                                        18.18502 Confirmed
```
Results from rpart: 
```
> rpart <- train(Converted ~ ., data = train, method = "rpart")
> rpart_imp <- varImp(rpart)
> plot(rpart_imp, top = 20, main = "RPart Variable Importance")
```
![RPart Variable Importance Plot](https://github.com/tsgruman/regis-practicum-leads-classification/blob/main/assets/rpart-var-imp.png)

Results from rfe: 
```
contr <- rfeControl(functions = rfFuncs, method = "repeatedcv", repeats = 5, verbose = FALSE)
lm <- rfe(x=train[,c(2:126)], y=train$Converted, rfeControl = contr)
lm$optVariables

 [1] "Total.Time.Spent.on.Website"                                            
 [2] "Asymmetrique.Activity.Score"                                            
 [3] "What.is.your.current.occupation.Working.Professional"                   
 [4] "Lead.Profile.Potential.Lead"                                            
 [5] "What.matters.most.to.you.in.choosing.a.course.Better.Career.Prospects"  
 [6] "Lead.Origin.Lead.Add.Form"                                              
 [7] "Last.Activity.SMS.Sent"                                                 
 [8] "Last.Notable.Activity.SMS.Sent"                                         
 [9] "Last.Notable.Activity.Modified"                                         
[10] "Do.Not.Email"                                                           
```
Based on these results, I have decided to select the following features:
* Total.Time.Spent.on.Website
* Asymmetrique.Activity.Score
* What.is.your.current.occupation.Working.Professional
* What.matters.most.to.you.in.choosing.a.course.Better.Career.Prospects
* Lead.Profile.Potential.Lead
* Lead.Origin.Lead.Add.Form

# Classification Modeling 

## K-Nearest Neighbors (KNN)

## Random Forest

## Extreme Gradient Boosting (XGBoost)

## Support Vector Machines (SVM)

# Discussion

# Further Research

# References

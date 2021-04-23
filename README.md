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
Based on these results, I have decided to select the following features for modeling:
* Total.Time.Spent.on.Website
* Asymmetrique.Activity.Score
* What.is.your.current.occupation.Working.Professional
* What.matters.most.to.you.in.choosing.a.course.Better.Career.Prospects
* Lead.Profile.Potential.Lead
* Lead.Origin.Lead.Add.Form

# Classification Modeling 
For classification modeling, I decided to use four different methods. I then tuned each model and compared the results to see which yielded the strongest model.

## K-Nearest Neighbors (KNN)
The first method I used was K-Nearest Neighbors, or KNN. I used the train() function from the caret package which outputs a tuned model, so I didn't need to adjust the parameters after the first go. 

After setting the control, I created the KNN model which selected k = 13 as the optimal value for the highest model accuracy. I then applied the optimized model to predict values against the test dataset.
```
> contr <- trainControl(method="repeatedcv", number = 10, repeats = 3)
> knnPred <- predict(knnFit, newdata = test, type = "prob")
#add column to convert predicted values to 0 or 1 factor
> knnPred$response <- as.factor(ifelse(knnPred$Yes > 0.5, 1, 0))
> confusionMatrix(knnPred$response, test_var_factor)

Confusion Matrix and Statistics

          Reference
Prediction   0   1
         0 795 133
         1  91 400
                                          
               Accuracy : 0.8421          
    ...
```
The KNN model's confusion matrix shows an accuracy of 84.21%. 

## Random Forest
The next method was Random Forest. I created a model with default values and then tuned the model using tuneRF() from the randomForest package to see if I could improve the model with a different mtry value, which is the number of variables sampled when splitting the tree nodes. The default value for classification modeling is the square root of the number of variables.
```
> rf <- randomForest(Converted~., data=train)
> rf_pred <- as.data.frame(predict(rf, newdata = test, type = "prob"))
> rf_pred$response <- as.factor(ifelse(rf_pred$Yes > 0.5, 1, 0))
> confusionMatrix(rf_pred$response, test_var_factor)

Confusion Matrix and Statistics

          Reference
Prediction   0   1
         0 821 154
         1  65 379
                                          
               Accuracy : 0.8457          
    ...
    
> features <- setdiff(names(train), "Converted")
> m_tune <- tuneRF(
    x = train[features],
    y = train$Converted,
    ntreeTry = 500,
    mtryStart = 2,
    stepFactor = 2,
    improve = 0.001,
    trace = FALSE
  )
```
As seen in the graph below, using a different value for mtry did not reduce the out-of-bag (OOB) error for the model. There is no need to tune this model further.

![Random Forest tuning graph](https://github.com/tsgruman/regis-practicum-leads-classification/blob/main/assets/tunerf-ob-error-01.png)

The random forest model with default values had an accuracy of 84.57%, which is slightly higher than the KNN model.

## Extreme Gradient Boosting (XGBoost)
The third classification modeling method I used was Extreme Gradient Boosting, or XGBoost, with the xgboost package. This modeling method required converting the data to a DMatrix, xgboost's internal data structure.

The first model was made with default values. I then proceeded to adjust various values to see which yielded the best results.
```
> xgb <- xgboost(data = d_train,
                 nrounds = 1000,
                 objective = "binary:logistic",
                 eval_metric = "error",
                 eval_metric = "auc",
                 early_stopping_rounds = 3)
                 
> xgb_pred <- predict(xgb, d_test, type = "response")
> xgb_pred_response <- as.data.frame(xgb_pred)
> xgb_pred_response$response <- as.factor(ifelse(xgb_pred_response > 0.5, 1, 0))
> confusionMatrix(xgb_pred_response$response, test_var_factor)

Confusion Matrix and Statistics

          Reference
Prediction   0   1
         0 778 149
         1 108 384
                                          
               Accuracy : 0.8189          
    ...

```
An accuracy score of 81.89% is a pretty good model, but it is already worse than the KNN and random forest models. For the second xgboost model, I lowered the max depth - maximum depth of the tree - value.
```
> xgb2 <- xgboost(data = d_train,
                  max.depth = 3,
                  nrounds = 1000,
                  early_stopping_rounds = 3,
                  objective = "binary:logistic",
                  eval_metric = "error",
                  eval_metric = "auc")

> xgb2_pred <- predict(xgb2, d_test)
> xgb2_pred_response <- as.data.frame(xgb2_pred)
> xgb2_pred_response$response <- as.factor(ifelse(xgb2_pred_response > 0.5, 1, 0))
> confusionMatrix(xgb2_pred_response$response, test_var_factor)

Confusion Matrix and Statistics

          Reference
Prediction   0   1
         0 791 145
         1  95 388
                                        
               Accuracy : 0.8309        
    ...
```
The second model produced better accuracy at 83.09%. The next model had the default value for max depth but lower eta, which is the step size.
```
> xgb3 <- xgboost(data = d_train,
                  eta = 0.1,
                  nrounds = 1000,
                  early_stopping_rounds = 3,
                  objective = "binary:logistic",
                  eval_metric = "error",
                  eval_metric = "auc")
                  
> xgb3_pred <- predict(xgb3, d_test)
> xgb3_pred_response <- as.data.frame(xgb3_pred)
> xgb3_pred_response$response <- as.factor(ifelse(xgb3_pred_response > 0.5, 1, 0))
> confusionMatrix(xgb3_pred_response$response, test_var_factor)

Confusion Matrix and Statistics

          Reference
Prediction   0   1
         0 786 142
         1 100 391
                                          
               Accuracy : 0.8295              
    ...
```
Although this produced a higher accuracy than the default model, it was lower than the adjusted max depth model at just 82.95%.

For the final model, I combined both adjusted parameters.
```
> xgb4 <- xgboost(data = d_train,
                  eta = 0.1,
                  max_depth = 3,
                  nrounds = 1000,
                  early_stopping_rounds = 3,
                  objective = "binary:logistic",
                  eval_metric = "error",
                  eval_metric = "auc")
                  
> xgb4_pred <- predict(xgb4, d_test)
> xgb4_pred_response <- as.data.frame(xgb4_pred)
> xgb4_pred_response$response <- as.factor(ifelse(xgb4_pred_response > 0.5, 1, 0))
> confusionMatrix(xgb4_pred_response$response, test_var_factor)

Confusion Matrix and Statistics

          Reference
Prediction   0   1
         0 787 113
         1  99 420
                                         
               Accuracy : 0.8506            
    ...
```
This produced a higher accuracy than the KNN or random forest models at 85.06%!

## Support Vector Machines (SVM)
The final method I used was support vector machines, better known as SVM. SVM allows one to choose a kernel type for modeling. For my purposes, I started with a linear SVM kernel and then tuned models for different kernal types with the tune.svm() function from the e1071 package.

Linear kernel
```
> svm.lin <- svm(formula = Converted ~ .,
                 data = train,
                 type = 'C-classification',
                 probability = TRUE,
                 kernel = 'linear')
                 
> svm_pred <- predict(svm.lin, newdata = test[-1], probability = TRUE)
> svm_response <- as.data.frame(attr(svm_pred, "probabilities"))
> svm_response$response <- as.factor(ifelse(svm_response$Yes > 0.5, 1, 0))
> confusionMatrix(svm_response$response, test_var_factor)

Confusion Matrix and Statistics

          Reference
Prediction   0   1
         0 791 157
         1  95 376
                                         
               Accuracy : 0.8224    
               
    ...
```
Polynomial kernel tuning and model
```
> poly.tune <- tune.svm(Converted ~ .,
                        data = train,
                        type = 'C-classification',
                        kernel = 'polynomial',
                        degree=c(3, 4, 5, 6),
                        coef0 = c(0.1, 0.5, 1, 2, 3))
> summary(poly.tune)

Parameter tuning of ‘svm’:

- sampling method: 10-fold cross validation 

- best parameters:
 degree coef0
      6     3

- best performance: 0.1848988 

#fit model with best parameters
> svm.poly <- svm(Converted ~ .,
                  data = train,
                  type = 'C-classification',
                  kernal = 'polynomial',
                  degree = 6,
                  coef0 = 3,
                  probability = TRUE)
> poly_pred <- predict(svm.poly, newdata = test[-1], probability = TRUE)
> poly_response <- as.data.frame(attr(poly_pred, "probabilities"))
> poly_response$response <- as.factor(ifelse(poly_response$Yes > 0.5, 1, 0))
> confusionMatrix(poly_response$response, test_var_factor)

Confusion Matrix and Statistics

          Reference
Prediction   0   1
         0 827 159
         1  59 374
                                          
               Accuracy : 0.8464      
   ...
```
Radial kernel tuning and model
```
> rad.tune <- tune.svm(Converted~.,
                       data = train,
                       kernal = 'radial',
                       gamma = c(0.1, 0.3, 0.5, 1, 2, 3))
> summary(rad.tune)

Parameter tuning of ‘svm’:

- sampling method: 10-fold cross validation 

- best parameters:
 gamma
   0.5

- best performance: 0.1791705 

#fit model with best parameters
> svm.rad <- svm(Converted ~ .,
                 data = train,
                 type = 'C-classification',
                 kernel = 'radial',
                 gamma = 0.5,
                 probability = TRUE)

> rad_pred <- predict(svm.rad, newdata = test[-1], probability = TRUE)
> rad_response <- as.data.frame(attr(rad_pred, "probabilities"))
> rad_response$response <- as.factor(ifelse(rad_response$Yes > 0.5, 1, 0))
> confusionMatrix(rad_response$response, test_var_factor)

Confusion Matrix and Statistics

          Reference
Prediction   0   1
         0 812 143
         1  74 390
                                          
               Accuracy : 0.8471      
   ...
```

# Results & Discussion

## ROC Curves

# Further Research

# References

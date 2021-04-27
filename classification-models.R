library(tidyverse)
library(caret)
library(randomForest)
library(xgboost)
library(e1071)
library(pROC)

#load train dataset
train <- read.csv("train.csv", row.names = 1)
#convert target variable to factor
train$Converted <- as.factor(train$Converted)

#convert target variable to num for class knn function
train_num <- train
train_num$Converted <- ifelse(train_num$Converted == "No", 0, 1)

#load test dataset
test <- read.csv("test.csv", row.names = 1)
#convert test variable to factor
test$Converted <- as.factor(test$Converted)

#convert test var to num for class
test_num <- test
test_num$Converted <- ifelse(test_num$Converted == "No", 0, 1)

#extract target variables as factor
target_var <- train$Converted
test_var <- test$Converted

#extract target variables as num
target_var_num <- train_num$Converted
test_var_num <- test_num$Converted

#convert target to factor
test_var_factor <- as.factor(test_var_num)

#create accuracy check function
#https://towardsdatascience.com/k-nearest-neighbors-algorithm-with-examples-in-r-simply-explained-knn-1f2c88da405c
accuracy <- function(x){sum(diag(x)/(sum(rowSums(x)))) * 100}

###

#knn modeling

#create knn model using caret package function

#start with setting control
set.seed(123)
contr <- trainControl(method="repeatedcv", number = 10, repeats = 3)

#fit model
knnFit <- train(Converted ~ ., data = train, method = "knn", trControl = contr, tuneLength = 25)
knnFit
plot(knnFit)

#predict using model
knnPred <- predict(knnFit, newdata = test, type = "prob")
knnPred
knnPred$response <- as.factor(ifelse(knnPred$Yes > 0.5, 1, 0))

confusionMatrix(knnPred$response, test_var_factor)

knncaret_mat <- table(knnPred$response, test$Converted)
knncaret_mat
accuracy(knncaret_mat)

###

#random forest modeling

#create model with default values
rf <- randomForest(Converted~., data=train)
rf

#use rf model to predict on test set
rf_pred <- as.data.frame(predict(rf, newdata = test, type = "prob"))

rf_pred$response <- as.factor(ifelse(rf_pred$Yes > 0.5, 1, 0))
head(rf_pred)

#evaluate results
confusionMatrix(rf_pred$response, test_var_factor)
rf_mat <- table(test$Converted, rf_pred$response)
rf_mat
accuracy(rf_mat)

#use tuneRF function to find optimal mtry value
features <- setdiff(names(train), "Converted")
m_tune <- tuneRF(
  x = train[features],
  y = train$Converted,
  ntreeTry = 500,
  mtryStart = 2,
  stepFactor = 2,
  improve = 0.001,
  trace = FALSE
)

#tune model with mtry = 4
rf2 <- randomForest(Converted~., data = train, mtry = 4)
rf2

rf2_pred <- as.data.frame(predict(rf2, newdata=test, type = "prob"))
rf2_pred$response <- as.factor(ifelse(rf2_pred$Yes > 0.5, 1, 0))

#evaluate results
confusionMatrix(rf2_pred$response, test_var_factor)
rf2_mat <- table(test$Converted, rf2_pred$response)
accuracy(rf2_mat) #worse model

###

#XGBoost modeling

#subset train and test data without target variable
train_new <- subset(train[,-1])
test_new <- subset(test[,-1])

#extract target variables into numeric labels
label_tr <- as.numeric(train$Converted)-1
label_ts <- as.numeric(test$Converted)-1

#convert data to matrix
mtrain <- data.matrix(train_new)
mtest <- data.matrix(test_new)

#convert data to DMatrix
d_train <- xgb.DMatrix(data = mtrain, label = label_tr)
d_test <- xgb.DMatrix(data = mtest, label = label_ts)

#create model using default parameters
xgb <- xgboost(data = d_train,
               nrounds = 1000,
               objective = "binary:logistic",
               eval_metric = "error",
               eval_metric = "auc",
               early_stopping_rounds = 3)

#predict with test data
xgb_pred <- predict(xgb, d_test, type = "response")

#add response column as factor
xgb_pred_response <- as.data.frame(xgb_pred)
xgb_pred_response$response <- as.factor(ifelse(xgb_pred_response > 0.5, 1, 0))

#evaluate model
confusionMatrix(xgb_pred_response$response, test_var_factor)

xgb_mat <- table(label_ts, xgb_pred_response$response)
accuracy(xgb_mat)

#tune XGBoost model - lower max depth
xgb2 <- xgboost(data = d_train,
                max.depth = 3,
                nrounds = 1000,
                early_stopping_rounds = 3,
                objective = "binary:logistic",
                eval_metric = "error",
                eval_metric = "auc")

xgb2_pred <- predict(xgb2, d_test)
xgb2_pred_response <- as.data.frame(xgb2_pred)
xgb2_pred_response$response <- as.factor(ifelse(xgb2_pred_response > 0.5, 1, 0))

confusionMatrix(xgb2_pred_response$response, test_var_factor)

xgb2_mat <- table(label_ts, xgb2_pred_response$response)
accuracy(xgb2_mat)

#tune XGBoost model - lower eta
xgb3 <- xgboost(data = d_train,
                eta = 0.1,
                nrounds = 1000,
                early_stopping_rounds = 3,
                objective = "binary:logistic",
                eval_metric = "error",
                eval_metric = "auc")

xgb3_pred <- predict(xgb3, d_test)
xgb3_pred_response <- as.data.frame(xgb3_pred)
xgb3_pred_response$response <- as.factor(ifelse(xgb3_pred_response > 0.5, 1, 0))

confusionMatrix(xgb3_pred_response$response, test_var_factor)

xgb3_mat <- table(label_ts, xgb3_pred_response$response)
accuracy(xgb3_mat)

#tune XGBoost model - lower eta AND max depth
xgb4 <- xgboost(data = d_train,
                eta = 0.1,
                max_depth = 3,
                nrounds = 1000,
                early_stopping_rounds = 3,
                objective = "binary:logistic",
                eval_metric = "error",
                eval_metric = "auc")

xgb4_pred <- predict(xgb4, d_test)
xgb4_pred_response <- as.data.frame(xgb4_pred)
xgb4_pred_response$response <- as.factor(ifelse(xgb4_pred_response > 0.5, 1, 0))

confusionMatrix(xgb4_pred_response$response, test_var_factor)

xgb4_mat <- table(label_ts, xgb4_pred_response$response)
accuracy(xgb4_mat)

###

#SVM modeling

#fit SVM model with linear kernel
svm.lin <- svm(formula = Converted ~ .,
                 data = train,
                 type = 'C-classification',
                 probability = TRUE,
                 kernel = 'linear')

#use model to predict values
svm_pred <- predict(svm.lin, newdata = test[-1], probability = TRUE)
svm_response <- as.data.frame(attr(svm_pred, "probabilities"))
svm_response$response <- as.factor(ifelse(svm_response$Yes > 0.5, 1, 0))

#check accuracy
confusionMatrix(svm_response$response, test_var_factor)

svm_mat <- table(test$Converted, svm_response$response)
svm_mat
accuracy(svm_mat)

#tune linear model
lin.tune <- tune.svm(Converted ~ .,
                     data = train,
                     kernel = "linear",
                     cost = c(0.001, 0.01, 0.1, 1, 5, 10))
summary(lin.tune) #cost = 1 is default so no need to tune model

#tune polynomial SVM model
#https://rpubs.com/Kushan/296706
poly.tune <- tune.svm(Converted ~ .,
             data = train,
             type = 'C-classification',
             kernel = 'polynomial',
             degree=c(3, 4, 5, 6),
             coef0 = c(0.1, 0.5, 1, 2, 3))

summary(poly.tune)

#fit poly model with tuned parameters
svm.poly <- svm(Converted ~ .,
                data = train,
                type = 'C-classification',
                kernal = 'polynomial',
                degree = 6,
                coef0 = 3,
                probability = TRUE)

poly_pred <- predict(svm.poly, newdata = test[-1], probability = TRUE)

poly_response <- as.data.frame(attr(poly_pred, "probabilities"))
poly_response$response <- as.factor(ifelse(poly_response$Yes > 0.5, 1, 0))

#check accuracy
confusionMatrix(poly_response$response, test_var_factor)

poly_mat <- table(test$Converted, poly_response$response)
accuracy(poly_mat)

#tune radial SVM model
rad.tune <- tune.svm(Converted~.,
                     data = train,
                     kernal = 'radial',
                     gamma = c(0.1, 0.3, 0.5, 1, 2, 3))

summary(rad.tune)

svm.rad <- svm(Converted ~ .,
              data = train,
              type = 'C-classification',
              kernel = 'radial',
              gamma = 0.5,
              probability = TRUE)

rad_pred <- predict(svm.rad, newdata = test[-1], probability = TRUE)

rad_response <- as.data.frame(attr(rad_pred, "probabilities"))
rad_response$response <- as.factor(ifelse(rad_response$Yes > 0.5, 1, 0))

#check accuracy
confusionMatrix(rad_response$response, test_var_factor)
rad_mat <- table(test$Converted, rad_response$response)
accuracy(rad_mat) #slightly better than poly

###

#plot results in ROC curve to compare performance based on AUC 

#plot ROC curves for all models with pROC package

#plot first ROC curve
roc(test_var_num, knnPred$Yes, plot = TRUE, col = 1,
    print.auc = TRUE, print.auc.cex = .8, print.auc.adj = c(-1, 2),
    main = "ROC Curves for All Models")

#add all other ROC curves to the first plot
plot.roc(test_var_num, rf_pred$Yes, print.auc = TRUE, col = 2,
         add = TRUE, print.auc.col = 2, print.auc.cex = .8, print.auc.adj = c(-1, 4))

plot.roc(test_var_num, rf2_pred$Yes, print.auc = TRUE, col = 3,
         add = TRUE, print.auc.col = 3, print.auc.cex = .8, print.auc.adj = c(-1, 6))

plot.roc(test_var_num, xgb_pred, print.auc = TRUE, col = 4,
         add = TRUE, print.auc.col = 4, print.auc.cex = .8, print.auc.adj = c(-1, 8))

plot.roc(test_var_num, xgb2_pred, print.auc = TRUE, col = 5,
         add = TRUE, print.auc.col = 5, print.auc.cex = .8, print.auc.adj = c(-1, 10))

plot.roc(test_var_num, xgb3_pred, print.auc = TRUE, col = 6,
         add = TRUE, print.auc.col = 6, print.auc.cex = .8, print.auc.adj = c(-1, 12))

plot.roc(test_var_num, xgb4_pred, print.auc = TRUE, col = 7,
         add = TRUE, print.auc.col = 7, print.auc.cex = .8, print.auc.adj = c(-1, 14))

plot.roc(test_var_num, svm_response[,2], print.auc = TRUE, col = 8,
         add = TRUE, print.auc.col = 8, print.auc.cex = .8, print.auc.adj = c(-1, 16))

plot.roc(test_var_num, poly_response[,2], print.auc = TRUE, col = "tomato",
         add = TRUE, print.auc.col = "tomato", print.auc.cex = .8, print.auc.adj = c(-1, 18))

plot.roc(test_var_num, rad_response[,2], print.auc = TRUE, col = "blueviolet",
         add = TRUE, print.auc.col = "blueviolet", print.auc.cex = .8, print.auc.adj = c(-1, 20))

#add legend to plot
legend("bottomright", legend = c("knn", "randforest1", "randforest2", 
                  "xgboost1", "xgboost2", "xgboost3", "xgboost4",
                  "svm-linear", "svm-polynomial", "svm-radial"),
       col=c(1,2,3,4,5,6,7,8,"tomato","blueviolet"),
       lty=1)

#split plots by model

#ROC curves - knn and random forest

roc(test_var_num, knnPred$Yes, plot = TRUE, col = 1,
    print.auc = TRUE, main = "ROC Curves - KNN and Random Forest")

plot.roc(test_var_num, rf_pred$Yes, print.auc = TRUE, col = 2,
         add = TRUE, print.auc.col = 2, print.auc.adj = c(0,3))

plot.roc(test_var_num, rf2_pred$Yes, print.auc = TRUE, col = 3,
         add = TRUE, print.auc.col = 3, print.auc.adj = c(0,5))

legend("bottomright", 
       legend = c("knn", "randforest1", "randforest2"),
       col=c(1,2,3),
       lty=1)

#ROC curves - XGBoost

roc(test_var_num, xgb_pred, plot = TRUE, print.auc = TRUE, col = 1,
    print.auc.col = 1, main = "ROC Curves - XGBoost Models")

plot.roc(test_var_num, xgb2_pred, print.auc = TRUE, col = 2,
         add = TRUE, print.auc.col = 2, print.auc.adj = c(0,3))

plot.roc(test_var_num, xgb3_pred, print.auc = TRUE, col = 3,
         add = TRUE, print.auc.col = 3, print.auc.adj = c(0,5))

plot.roc(test_var_num, xgb4_pred, print.auc = TRUE, col = 4,
         add = TRUE, print.auc.col = 4, print.auc.adj = c(0,7))

legend("bottomright", 
       legend = c("xgboost1", "xgboost2", "xgboost3", "xgboost4"),
       col=c(1,2,3,4),
       lty=1)

#ROC curves - SVM
roc(test_var_num, svm_response[,2], plot = TRUE, print.auc = TRUE, col = 1,
         print.auc.col = 1, main = "ROC Curves - SVM Models")

plot.roc(test_var_num, poly_response[,2], print.auc = TRUE, col = 2,
         add = TRUE, print.auc.col = 2, print.auc.adj = c(0,3))

plot.roc(test_var_num, rad_response[,2], print.auc = TRUE, col = 3,
         add = TRUE, print.auc.col = 3, print.auc.adj = c(0,5))

legend("bottomright", 
       legend = c("svm-linear", "svm-polynomial", "svm-radial"),
       col=c(1,2,3),
       lty=1)
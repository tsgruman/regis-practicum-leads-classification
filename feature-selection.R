library(tidyverse)
library(sjmisc)
library(caret)
library(randomForest)
library(RRF)
library(Boruta)

#load file saved from data cleaning & move target variable to first column
clean <- read.csv("clean.csv", row.names = 1) %>%
  relocate(Converted, .before = Lead.Origin.API)
str(clean)
clean$Converted <- as.factor(clean$Converted)
df <- na.omit(clean)
str(df)

#split data into train and test sets
set.seed(123)
trainInd <- createDataPartition(df$Converted, p = .7, times = 1, list = FALSE)
train <- df[trainInd,]
test <- df[-trainInd,]

#normalize train set non-binary data with z-score
train$Total.Time.Spent.on.Website <- (train$Total.Time.Spent.on.Website - mean(train$Total.Time.Spent.on.Website))/sd(train$Total.Time.Spent.on.Website)

train$Page.Views.Per.Visit <- (train$Page.Views.Per.Visit - mean(train$Page.Views.Per.Visit))/sd(train$Page.Views.Per.Visit)

train$Asymmetrique.Activity.Score <- (train$Asymmetrique.Activity.Score - mean(train$Asymmetrique.Activity.Score)) / sd(train$Asymmetrique.Activity.Score)

train$Asymmetrique.Profile.Score <- (train$Asymmetrique.Profile.Score - mean(train$Asymmetrique.Profile.Score)) / sd(train$Asymmetrique.Profile.Score)

#normalize test set non-binary data with z-score
test$Total.Time.Spent.on.Website <- (test$Total.Time.Spent.on.Website - mean(test$Total.Time.Spent.on.Website))/sd(test$Total.Time.Spent.on.Website)

test$Page.Views.Per.Visit <- (test$Page.Views.Per.Visit - mean(test$Page.Views.Per.Visit))/sd(test$Page.Views.Per.Visit)

test$Asymmetrique.Activity.Score <- (test$Asymmetrique.Activity.Score - mean(test$Asymmetrique.Activity.Score)) / sd(test$Asymmetrique.Activity.Score)

test$Asymmetrique.Profile.Score <- (test$Asymmetrique.Profile.Score - mean(test$Asymmetrique.Profile.Score)) / sd(test$Asymmetrique.Profile.Score)

#boruta feature selection
boruta <- Boruta(Converted ~ ., data = train, doTrace = 1)
#print significant attributes
boruta_sig <- getSelectedAttributes(boruta, withTentative=TRUE)
print(boruta_sig)

roughfix <- TentativeRoughFix(boruta)
boruta_sig_rf <- getSelectedAttributes(roughfix)
print(boruta_sig_rf)

#print var importance scores
imp_score <- attStats(boruta)
imps = imp_score[imp_score$decision != 'Rejected', c('meanImp', 'decision')]
head(imps[order(-imps$meanImp), ], 10)

#print var importance scores for rough fix
imp_score_rf <- attStats(roughfix)
imps_rf = imp_score_rf[imp_score_rf$decision != 'Rejected', c('meanImp', 'decision')]
head(imps_rf[order(-imps_rf$meanImp), ], 10)

#plot var importance
par(mar = c(20,5,5,1))
plot(boruta, cex.axis=.6, las=2, xlab="", main="Variable Importance")
plot(roughfix, cex.axis=.6, las=2, xlab="", main="Rough Fix Variable Importance")

#caret package - rpart
rpart <- train(Converted ~ ., data = train, method = "rpart")
rpart_imp <- varImp(rpart)
print(rpart_imp)
plot(rpart_imp, top = 20, main = "RPart Variable Importance")

#Recursive Feature Elimination (RFE)
contr <- rfeControl(functions = rfFuncs, method = "repeatedcv", repeats = 5, verbose = FALSE)
lm <- rfe(x=train[,c(2:126)], y=train$Converted, rfeControl = contr)
lm
lm$optVariables

#create subset train and test sets with selected variables
train_sub <- subset(train, select = c(Converted, Total.Time.Spent.on.Website, 
                                Asymmetrique.Activity.Score,
                                What.is.your.current.occupation.Working.Professional,
                                What.matters.most.to.you.in.choosing.a.course.Better.Career.Prospects,
                                Lead.Profile.Potential.Lead, 
                                Lead.Origin.Lead.Add.Form))

test_sub <- subset(test, select = c(Converted, Total.Time.Spent.on.Website, 
                                     Asymmetrique.Activity.Score,
                                     What.is.your.current.occupation.Working.Professional,
                                     What.matters.most.to.you.in.choosing.a.course.Better.Career.Prospects,
                                     Lead.Profile.Potential.Lead, 
                                     Lead.Origin.Lead.Add.Form))

#save train and test files locally
write.csv(train_sub, "train.csv")

write.csv(test_sub, "test.csv")

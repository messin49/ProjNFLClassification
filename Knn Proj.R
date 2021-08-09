# Data via https://www.kaggle.com/tobycrabtree/nfl-scores-and-betting-data#nfl_teams.csv

install.packages("caret")
install.packages('pROC')

library(caret)
library(MLmetrics)
library(ggplot2)
library(pROC)

set.seed(123)


df <- read.csv('spreadspoke_scoresF2.csv')


summary(df)


dummies <- dummyVars(result ~ ., data = df)
df_dummies = as.data.frame(predict(dummies, newdata = df))


df_dummies$result =  df$result



inTrain <- createDataPartition(y = df_dummies$result, p = 0.7, list = FALSE)
training <- df_dummies[inTrain,] 
testing <- df_dummies[-inTrain,] 


preprocess <- preProcess(training, method = c("center", "scale"))

train_transformed <- predict(preprocess, training)
test_transformed <- predict(preprocess, testing)


fitControl <- trainControl(method = "repeatedcv", number = 3, repeats = 3)


knn1 <- train(result ~ ., data = train_transformed, method = "knn",  trControl = fitControl)
knn1


plot(knn1)


varImp(knn1)


pred <- predict(knn1, newdata = test_transformed)


confusionMatrix(pred, testing$result)


confusionMatrix(pred, testing$result, mode = "prec_recall")


fitControl_prob <- trainControl(method = "repeatedcv", number = 3, repeats = 3,  classProbs = TRUE, summaryFunction = twoClassSummary)
knn2 <- train(result ~ ., data = train_transformed, method = "knn",  trControl = fitControl_prob, metric='ROC')
knn2


pred2 <- predict(knn2, newdata = test_transformed, type = "prob")


plot(roc((testing$result),pred2[,1]), print.auc=TRUE)

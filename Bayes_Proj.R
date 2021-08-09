# Data via https://www.kaggle.com/tobycrabtree/nfl-scores-and-betting-data#nfl_teams.csv

install.packages("caret")
install.packages('pROC')
install.packages('MLmetrics')
install.packages('tidyverse')
install.packages('skimr')
install.packages('klaR')

library(caret)
library(MLmetrics)
library(ggplot2)
library(pROC)
library(tidyverse)
library(klaR)


set.seed(123)

# Import the data
df <- read.csv('spreadspoke_scoresF3.csv')

# Create dummy variables for each of our categorical columns. 
# You will get a warning for the second line here, but everything is okay.
dummies <- dummyVars(result ~ ., data = df)
df_dummies = as.data.frame(predict(dummies, newdata = df))


df_dummies$result =  df$result



inTrain <- createDataPartition(y = df_dummies$result, p = 0.7, list = FALSE)
training <- df_dummies[inTrain,] 
testing <- df_dummies[-inTrain,] 


preprocess <- preProcess(training, method = c("center", "scale"))

train_transformed <- predict(preprocess, training)
test_transformed <- predict(preprocess, testing)


df <- read.csv('spreadspoke_scoresF3')
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

bayes <- train(result ~ ., data = train_transformed, method = "nb", trControl=fitControl)
bayes

pred <- predict(bayes, newdata = test_transformed)
confusionMatrix(pred, test_transformed$result)


varImp(bayes)


tuneControl <- data.frame(fL=1, usekernel = TRUE, adjust=1)
bayes2 <- train(result ~ ., data = train_transformed, method = "nb", tuneGrid=tuneControl)
pred2 <- predict(bayes2, newdata = test_transformed)
confusionMatrix(pred2, test_transformed$result)
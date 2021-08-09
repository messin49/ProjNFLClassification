# Data via https://www.kaggle.com/tobycrabtree/nfl-scores-and-betting-data#nfl_teams.csv

install.packages("caret")
install.packages('rpart')
install.packages('rpart.plot')
install.packages('MLmetrics')

library(caret)
library(rpart)
library(rpart.plot)
library(MLmetrics)
library(ggplot2)

set.seed(123)


df <- read.csv('spreadspoke_scoresF3.csv')


summary(df)


ggplot(data = df) + geom_histogram(mapping = aes(x = over_under_line))




inTrain <- createDataPartition(y = df$result, p = 0.7, list = FALSE)
training <- df[ inTrain,] 
testing <- df[-inTrain,] 



tree <- rpart(result ~ ., data = training, method='class', control=rpart.control(minsplit=2, cp=0))


prp(tree, under=TRUE, type=3, varlen = 0, faclen = 0, extra = TRUE)


tree$variable.importance


printcp(tree)
plotcp(tree)


print(tree)

tree <- rpart(result ~ ., data = training, method='class', control=rpart.control(cp=0))


tree.pred = predict(tree, testing, type="class")


confusionMatrix(tree.pred, testing$result)


pruned <- prune(tree, cp = 0.00623)
prp(pruned, under=TRUE, varlen = 0, faclen = 0, extra = TRUE)


tree.pruned = predict(pruned, testing, type="class")


confusionMatrix(tree.pruned, testing$result)

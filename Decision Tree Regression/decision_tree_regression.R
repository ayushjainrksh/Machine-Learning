dataset = read.csv('Position_Salaries.csv')
dataset = dataset[2:3]

#install.packages('rpart')
library(rpart)
regressor = rpart(formula = Salary ~ .,
                  data = dataset,
                  control = rpart.control(minsplit = 1))

y_pred = predict(regressor, data.frame(Level=6.5))

library(ggplot2)
X_grid = seq(min(dataset$Level),max(dataset$Level),0.01)
ggplot() +
  geom_point(aes(x = dataset$Level, y = dataset$Salary),
             color = 'red') +
  geom_line(aes(x = X_grid , y = predict(regressor,newdata = data.frame(Level = X_grid))),
             color = 'blue') +
  ggtitle('Truth or bluff(Decision Tree Regression') +
  xlab('Levels') +
  ylab('Salary')